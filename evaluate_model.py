"""
Multi-Dimensional Evaluation Framework for Chess Commentary AI
==============================================================
Three evaluation pillars (Research Paper Section 3.7):

Pillar 1: LLM Judge Score — local open-source model (Ollama) grades quality
Pillar 2: Concept Recall — engine-detected tactics vs AI mentions
Pillar 3: Semantic Match — cosine similarity: AI text vs engine analysis

Bonus:  Move Quality Agreement, Square Precision

All metrics use the ENGINE as ground truth. No human annotations required.
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import chess
import chess.pgn
import google.generativeai as genai

load_dotenv()

# ==============================================================================
# IMPORT EXACT LOGIC FROM THE APP
# ==============================================================================
from app import (
    classify_move_quality,
    build_rich_context,
    tactics_analyzer,
    STOCKFISH_PATH,
)
from stockfish import Stockfish
from sentence_transformers import SentenceTransformer, util

# ==============================================================================
# OLLAMA (LOCAL JUDGE) SETUP
# ==============================================================================
try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ==============================================================================
# GEMINI API KEYS (for commentary generation only)
# ==============================================================================
GEMINI_API_KEYS = [
    "AIzaSyDYpql0My_QlYNRIfRH9NREwWurBcwuce0",
    "AIzaSyDrKbSXjUgC8Jrk4SD6hjcz_WjVq51bbbg",
    "AIzaSyArkelQjh2K7zJ7CrWQeLISVzfQbC8K6fU",
    "AIzaSyDtypNWJIYViSEl50EGiCxqZ3o7GpRD-FY",
    "AIzaSyD5I9YxI6IHqM5X_ZIJbJQH-VEI6wHqk7Y",
    "AIzaSyAMCg7BTnVJwnPUk1HhcOKP45GVmWkVE_s",
    "AIzaSyC_OWB2uJJCjXzuiqtWOm3WIkcEs4cRr0Y",
    "AIzaSyBlIKVOTe4t7ZFEJzeGUl7Zgm987LyZZZw"
]
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]


class KeyRotator:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.current_index = 0

    def get_next_key(self) -> str:
        key = self.keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.keys)
        return key


key_rotator = KeyRotator(GEMINI_API_KEYS)


def call_gemini_with_rotation(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    """Gemini call for commentary generation (NOT for judging)."""
    genai.configure(api_key=key_rotator.get_next_key())
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"      [!] Gemini API Error: {e}")
        return ""


# ==============================================================================
# SEMANTIC MODEL
# ==============================================================================
print("Loading Semantic Evaluation Model (MPNet)...")
similarity_model = SentenceTransformer('all-mpnet-base-v2')


# ==============================================================================
# EVALUATOR
# ==============================================================================
class ExactAppEvaluator:
    """Engine-grounded evaluation using the 3-pillar framework."""

    _QUALITY_SIGNALS = {
        "blunder": ["blunder", "catastrophic", "drops", "loses", "terrible"],
        "mistake": ["mistake", "inaccurate", "misses", "error", "slip"],
        "brilliant": ["brilliant", "stunning", "spectacular", "beautiful"],
    }

    def __init__(self, judge_type: str = "ollama", judge_model: str = "qwen2.5:3b"):
        if not STOCKFISH_PATH:
            raise RuntimeError("Stockfish path not found in app.py!")
        self.engine = Stockfish(path=STOCKFISH_PATH, depth=14, parameters={"Threads": 2})
        self.square_regex = re.compile(r'\b[a-h][1-8]\b')
        self.judge_type = judge_type
        self.judge_model = judge_model

        if judge_type == "ollama" and not OLLAMA_AVAILABLE:
            print("  [!] WARNING: ollama package not installed. Install with: pip install ollama")
            print("  [!] Also install Ollama itself: https://ollama.com/download")
            print("  [!] Falling back to Gemini judge (circular evaluation warning).")
            self.judge_type = "gemini"
        if self.judge_type == "gemini":
            print("  [⚠] WARNING: Using Gemini to judge Gemini output (circular evaluation).")

    # ==================== HELPER ====================

    def _get_eval_cp(self, fen: str) -> int:
        self.engine.set_fen_position(fen)
        ev = self.engine.get_evaluation()
        return (10000 if ev["value"] > 0 else -10000) if ev["type"] == "mate" else ev["value"]

    def _run_app_analysis(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> Dict:
        analysis_after = tactics_analyzer.analyze(curr_board.fen(), prev_board.fen())
        analysis_after["attack_info"] = tactics_analyzer.detect_attack(prev_board, curr_board, move)
        analysis_after["pressure_info"] = tactics_analyzer.detect_pressure(prev_board, curr_board, move)
        is_brill, brill_det = tactics_analyzer.detect_brilliant_move(prev_board, curr_board, move, sid="local")
        analysis_after["is_brilliant"] = is_brill
        analysis_after["brilliant_details"] = brill_det
        return analysis_after

    # ==================== PILLAR 1: LLM JUDGE ====================

    def _call_judge(self, prompt: str) -> str:
        if self.judge_type == "ollama":
            return self._call_ollama(prompt)
        elif self.judge_type == "gemini":
            return call_gemini_with_rotation(prompt, model_name="gemini-2.5-flash")
        elif self.judge_type == "gemma":
            return call_gemini_with_rotation(prompt, model_name="gemma-4-31b-it")
        return ""

    def _call_ollama(self, prompt: str) -> str:
        try:
            resp = ollama_client.chat(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp["message"]["content"]
        except Exception as e:
            print(f"      [!] Ollama error: {e}")
            return ""

    def llm_judge_score(self, ply: int, fen: str, eval_text: str,
                        move_san: str, quality: str, tactics: str,
                        ai_text: str, keywords: str = "",
                        draft: str = "") -> float:
        """Pillar 1: Grade commentary via LLM judge (local or fallback)."""
        prompt = (
            "You are an expert chess commentary evaluator.\n\n"
            f"BOARD (FEN): {fen}\n"
            f"EVAL: {eval_text}\n"
            f"MOVE: {move_san} ({quality})\n"
            f"TACTICS DETECTED: {tactics or 'none'}\n"
        )
        if keywords:
            prompt += f"REQUIRED KEYWORDS: {keywords}\n"
        if draft:
            prompt += f"VERIFIED DRAFT: {draft}\n"
        prompt += (
            f"\nAI COMMENTARY: \"{ai_text}\"\n\n"
            "Grade on three criteria (0-10 each):\n"
            "1. TECHNICAL PRECISION — does commentary mention the correct pieces, squares, and tactical patterns from the detected tactics? Does it name specific squares?\n"
            "2. ENGAGEMENT — natural conversational tone, not robotic or generic? Does it flow like expert commentary?\n"
            "3. INSIGHT — explains WHY the move matters, not just WHAT happened? Mentions consequences or follow-up ideas?\n\n"
            "Award higher scores when commentary mentions specific squares, piece names, and tactical concepts.\n"
            "Think step by step. Output ONLY valid JSON:\n"
            '{"precision":<0-10>,"engagement":<0-10>,"insight":<0-10>}\n'
        )
        resp = self._call_judge(prompt)
        if not resp:
            return 0.5
        try:
            m = re.search(r'\{[\s\S]*?\}', resp)
            if m:
                d = json.loads(m.group(0))
                return (d.get("precision", 5) + d.get("engagement", 5) + d.get("insight", 5)) / 30.0
        except Exception:
            pass
        return 0.5

    # ==================== PILLAR 2: CONCEPT RECALL ====================

    def _extract_engine_concepts(self, context: Dict) -> List[str]:
        """All tactical/strategic concepts the engine detected for this move."""
        concepts = []
        tactical = context.get("tactical", {})
        analysis = context.get("analysis", {})
        verified = context.get("verified_facts", {})

        # Primary tactic
        primary = tactical.get("primary_tactic")
        if primary:
            ptype = primary.get("type", "").replace("_", " ")
            if ptype:
                concepts.append(ptype)

        # Additional tactical patterns
        for p in analysis.get("tactical_patterns", []):
            ptype = p.get("type", "").replace("_", " ")
            if ptype and ptype not in concepts:
                concepts.append(ptype)

        # Attack info
        attack = tactical.get("attack_info", {})
        if attack.get("is_attacking"):
            target = str(attack.get("target_piece", ""))
            if target:
                concepts.append(target)

        # Pressure info
        pressure = tactical.get("pressure_info", {})
        if pressure.get("is_pressure"):
            target = str(pressure.get("target_piece", ""))
            if target and target not in concepts:
                concepts.append(target)

        # Capture analysis
        capture = tactical.get("capture_analysis", {})
        cap_type = str(capture.get("type", ""))
        if cap_type in ("sacrifice", "free_capture", "winning_capture"):
            concepts.append(cap_type.replace("_", " "))
        cap_target = str(capture.get("target", ""))
        if cap_target and cap_target not in concepts:
            concepts.append(cap_target)

        # Hung piece
        hung = tactical.get("hung_piece") or {}
        if hung.get("piece"):
            concepts.append(str(hung["piece"]))

        if tactical.get("is_brilliant"):
            concepts.append("brilliant")

        # Move quality
        quality = str(context.get("quality", ""))
        if quality in ("blunder", "mistake", "brilliant") and quality not in concepts:
            concepts.append(quality)

        # Verified facts: check, checkmate, castling, captures, development
        if verified.get("gives_checkmate"):
            concepts.append("checkmate")
        elif verified.get("gives_check"):
            concepts.append("check")
        if verified.get("castling_side"):
            concepts.append("castle")
        if verified.get("captures") and "captures" not in concepts:
            concepts.append(str(verified["captures"]))
        if verified.get("develops_minor_piece"):
            concepts.append("develop")

        return concepts

    def concept_recall(self, context: Dict, ai_text: str) -> float:
        """Pillar 2: Fraction of engine-detected concepts mentioned by AI."""
        detected = self._extract_engine_concepts(context)
        if not detected:
            return 1.0
        ai_lower = ai_text.lower()
        mentioned = sum(1 for c in detected if c in ai_lower)
        return mentioned / len(detected)

    # ==================== PILLAR 3: SEMANTIC MATCH ====================

    def _build_engine_truth_text(self, context: Dict) -> str:
        """Natural-language rendering of what the engine sees (the PV truth)."""
        parts = []
        move_info = context.get("move", {})
        quality = context.get("quality", "")
        eval_info = context.get("eval", {})
        tactical = context.get("tactical", {})
        verified = context.get("verified_facts", {})
        analysis = context.get("analysis", {})

        mover = move_info.get("mover", "")
        san = move_info.get("san", "")
        piece = move_info.get("piece", "piece")
        move_from = verified.get("move_from", "")
        move_to = verified.get("move_to", "")
        if move_from and move_to:
            parts.append(f"{mover} plays {san}, moving the {piece} from {move_from} to {move_to}.")
        else:
            parts.append(f"{mover} plays {san} with the {piece}.")

        if verified.get("captures"):
            parts.append(f"This captures the {verified['captures']}.")
        if verified.get("gives_checkmate"):
            parts.append("Checkmate.")
        elif verified.get("gives_check"):
            parts.append(f"Check from {verified['gives_check']}.")
        if verified.get("castling_side"):
            parts.append(f"{verified['castling_side'].title()} castling.")
        if verified.get("is_recapture"):
            parts.append("This is a recapture.")
        if verified.get("develops_minor_piece"):
            parts.append(f"This develops the {piece}.")

        qdesc = {"blunder": "A serious blunder.", "mistake": "A mistake.",
                 "brilliant": "A brilliant move.", "best": "The best move.",
                 "excellent": "An excellent move."}
        if quality in qdesc:
            parts.append(qdesc[quality])

        fmt = eval_info.get("formatted_after", "")
        if fmt:
            parts.append(f"Evaluation: {fmt}.")

        # Primary tactic with narrative
        primary = tactical.get("primary_tactic")
        if primary:
            narr = primary.get("narrative", "")
            parts.append(narr if narr else f"There is a {primary.get('type','').replace('_',' ')}.")

        # Attack info
        attack = tactical.get("attack_info", {})
        if attack.get("is_attacking"):
            parts.append(f"The {attack.get('attacker_piece','piece')} attacks the {attack.get('target_piece','piece')} on {attack.get('target_square','')}.")

        # Pressure info
        pressure = tactical.get("pressure_info", {})
        if pressure.get("is_pressure"):
            parts.append(f"Pressure on the {pressure.get('target_piece','piece')} on {pressure.get('target_square','')}.")

        # Capture analysis
        capture = tactical.get("capture_analysis", {})
        cap_type = str(capture.get("type", ""))
        if cap_type:
            cap_target = capture.get("target", "piece")
            cap_attacker = capture.get("attacker", "piece")
            if cap_type == "sacrifice":
                parts.append(f"The {cap_attacker} is sacrificed for the {cap_target}.")
            elif cap_type == "free_capture":
                parts.append(f"The {cap_target} is captured for free.")
            elif cap_type == "winning_capture":
                parts.append(f"The {cap_target} is won for less material.")
            elif cap_type == "equal_trade":
                parts.append(f"Equal trade of {cap_attacker} for {cap_target}.")

        # Hung piece
        hung = tactical.get("hung_piece") or {}
        if hung.get("piece"):
            parts.append(f"The {hung['piece']} on {hung.get('square','')} is left hanging.")

        # Resolved threats
        resolved = tactical.get("resolved_threats") or []
        if resolved:
            parts.append(f"This saves the {resolved[0]} from danger.")

        # Strategic themes
        for theme in (analysis.get("strategic_themes") or [])[:2]:
            parts.append(str(theme).rstrip(".") + ".")

        # King safety
        king_safety = analysis.get("king_safety", {})
        for side_key in ("white", "black"):
            ks = king_safety.get(side_key, {})
            level = str(ks.get("level", ""))
            if level in ("vulnerable", "exposed"):
                parts.append(f"{side_key.title()}'s king is {level}.")

        # Engine best line
        engine_info = context.get("engine_info", {})
        best_line = engine_info.get("best_line_after", "")
        if best_line:
            parts.append(f"Best continuation: {best_line}.")

        # Missed tactics
        missed = context.get("missed_tactics") or {}
        if missed.get("missed_best_move"):
            parts.append(f"The correct move was {missed['missed_best_move']}.")

        return " ".join(parts)

    def semantic_match(self, context: Dict, ai_text: str) -> float:
        """Pillar 3: Cosine similarity between AI commentary and engine truth."""
        engine_text = self._build_engine_truth_text(context)
        if not engine_text or not ai_text:
            return 0.0
        e1 = similarity_model.encode(engine_text, convert_to_tensor=True)
        e2 = similarity_model.encode(ai_text, convert_to_tensor=True)
        return max(0.0, util.pytorch_cos_sim(e1, e2).item())

    # ==================== BONUS METRICS ====================

    def move_quality_agreement(self, quality: str, ai_text: str) -> float:
        """Did the AI correctly identify the move quality?"""
        ai_lower = ai_text.lower()
        if quality not in self._QUALITY_SIGNALS or not self._QUALITY_SIGNALS[quality]:
            negatives = ["blunder", "mistake", "error", "terrible", "catastrophic"]
            return 0.0 if any(w in ai_lower for w in negatives) else 1.0
        return 1.0 if any(w in ai_lower for w in self._QUALITY_SIGNALS[quality]) else 0.0

    def square_precision(self, context: Dict, ai_text: str) -> float:
        """Did AI mention the correct board squares?"""
        verified = context.get("verified_facts", {})
        tactical = context.get("tactical", {})
        correct_sq = set()
        for key in ("move_from", "move_to"):
            if verified.get(key):
                correct_sq.add(verified[key])
        # Include tactic squares as valid
        primary = tactical.get("primary_tactic") or {}
        for sq in primary.get("squares", []):
            if sq:
                correct_sq.add(str(sq))
        # Include attack/pressure target squares
        attack = tactical.get("attack_info", {})
        if attack.get("target_square"):
            correct_sq.add(str(attack["target_square"]))
        pressure = tactical.get("pressure_info", {})
        if pressure.get("target_square"):
            correct_sq.add(str(pressure["target_square"]))
        # Hung piece square
        hung = tactical.get("hung_piece") or {}
        if hung.get("square"):
            correct_sq.add(str(hung["square"]))

        ai_sq = set(self.square_regex.findall(ai_text.lower()))
        if not ai_sq:
            return 0.5 if correct_sq else 1.0
        if not correct_sq:
            return 1.0
        # Reward mentioning correct squares; penalize less for extra squares
        hits = len(correct_sq & ai_sq)
        if hits == 0:
            return 0.0
        # Blend precision and recall
        precision = hits / len(ai_sq)
        recall = hits / len(correct_sq)
        return 0.5 * precision + 0.5 * recall

    # ==================== COMMENTARY GENERATION (real pipeline) ====================

    def batch_generate_commentary(self, contexts: List[Dict]) -> Dict[int, str]:
        """Generate commentary using the exact same pipeline as app.py."""
        if not contexts:
            return {}

        context_lines = []
        for ctx in contexts:
            mv = ctx["move"]
            bp = ctx.get("blueprint") or ctx.get("commentary_blueprint") or {}
            draft = str(ctx.get("draft_commentary", "") or "")
            kw_text = ", ".join(str(i) for i in bp.get("keywords", []) if i) or "none"
            # Gather engine facts for richer commentary
            tactical = ctx.get("tactical", {})
            engine_info = ctx.get("engine_info", {})
            verified = ctx.get("verified_facts", {})
            tactic_type = ""
            primary = tactical.get("primary_tactic") or {}
            if primary.get("type"):
                tactic_type = primary["type"].replace("_", " ")
            best_line = engine_info.get("best_line_after", "")
            quality = str(ctx.get("quality", ""))
            move_from = verified.get("move_from", "")
            move_to = verified.get("move_to", "")
            hung = tactical.get("hung_piece") or {}
            parts = [
                f"ply={ctx['ply']}",
                f"move={mv['mover']} {mv['san']}",
                f"quality={quality}",
                f"from={move_from} to={move_to}",
                f"draft={draft}",
                f"lead={bp.get('lead', '') or '-'}",
                f"support={bp.get('support', '') or '-'}",
                f"verdict={bp.get('verdict', '') or '-'}",
                f"keywords={kw_text}",
            ]
            if tactic_type:
                parts.append(f"tactic={tactic_type}")
            if best_line:
                parts.append(f"best_line={best_line}")
            if hung.get("piece"):
                parts.append(f"hung={hung['piece']} on {hung.get('square', '')}")
            context_lines.append(" | ".join(parts))

        prompt = (
            "You are polishing chess commentary from verified engine analysis.\n"
            "The chess facts (tactics, squares, best lines) are already verified. Your job is to make them sound natural.\n\n"
            "RULES:\n"
            "1. Rewrite the draft into polished, human-sounding commentary\n"
            "2. You MUST preserve all locked keywords, piece names, and square references from the draft\n"
            "3. If a tactic is detected (fork, pin, skewer, etc.), you MUST name it in the commentary\n"
            "4. If a best_line is provided, reference it naturally (e.g., 'the engine suggests...')\n"
            "5. Always mention the destination square of the move\n"
            "6. Keep each commentary 1-3 sentences. Be specific, not generic\n"
            "7. For blunders/mistakes: name what was wrong and what was better\n"
            '8. Output ONLY valid JSON: {"commentaries":[{"ply":<int>,"text":"<commentary>"}, ...]}\n\n'
            f"PLANS:\n{chr(10).join(context_lines)}"
        )

        resp_text = call_gemini_with_rotation(prompt)
        by_ply: Dict[int, str] = {}
        if resp_text:
            match = re.search(r'\{[\s\S]*\}', resp_text)
            if match:
                try:
                    for item in json.loads(match.group(0)).get("commentaries", []):
                        ply = item.get("ply")
                        text = str(item.get("text", ""))
                        if ply and text:
                            by_ply[int(ply)] = text
                except Exception as e:
                    print(f"JSON Parse error: {e}")

        for ctx in contexts:
            ply = ctx.get("ply")
            if ply and ply not in by_ply:
                draft = str(ctx.get("draft_commentary", "") or "")
                if draft:
                    by_ply[ply] = draft
        return by_ply

    # ==================== MAIN EVALUATION LOOP ====================

    def run_evaluation(self, pgn_file_path: str, skip_llm_judge: bool = False, eval_mode: str = "ai_generate", max_games: int = 1):
        print(f"\n{'='*60}")
        print(f" 3-PILLAR EVALUATION FRAMEWORK (Engine Ground Truth)")
        print(f" Judge: {self.judge_type} ({self.judge_model})" if self.judge_type == "ollama"
              else f" Judge: {self.judge_type}")
        print(f"{'='*60}")

        all_metrics: List[Dict[str, float]] = []
        game_count = 0
        open("commented_test_game.pgn", "w", encoding="utf-8").close()

        with open(pgn_file_path, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if not game:
                    break

                game_count += 1
                if max_games > 0 and game_count > max_games:
                    break
                white = game.headers.get("White", "?")
                black = game.headers.get("Black", "?")
                print(f"\n--- Game {game_count}: {white} vs {black} ---")

                board = game.board()
                app_contexts: List[Dict] = []
                previous_blueprint: Optional[Dict[str, Any]] = None
                previous_move: Optional[chess.Move] = None

                pgn_commentaries: Dict[int, str] = {}
                for node in game.mainline():
                    ply = node.ply()
                    move = node.move
                    if node.comment:
                        pgn_commentaries[ply] = node.comment
                    cp_before = self._get_eval_cp(board.fen())
                    prev_board = board.copy()
                    board.push(move)
                    cp_after = self._get_eval_cp(board.fen())

                    quality = classify_move_quality(cp_before, cp_after, prev_board.turn == chess.WHITE)
                    analysis_after = self._run_app_analysis(prev_board, board, move)

                    missed_tactics: Dict[str, Any] = {}
                    if quality in ("blunder", "mistake"):
                        try:
                            from app import detect_missed_tactics
                            missed_tactics = detect_missed_tactics(
                                sid="local", board_before=prev_board, played_move=move,
                                cp_before=cp_before, cp_after=cp_after, quality=quality,
                                tactics_enabled=True,
                            )
                        except Exception as e:
                            print(f"      [!] missed_tactics error: {e}")

                    opening_name = game.headers.get("Opening") or game.headers.get("ECO") or "Unknown"

                    ctx = build_rich_context(
                        prev_board=prev_board, curr_board=board, move=move,
                        move_san=prev_board.san(move), analysis_after=analysis_after,
                        cp_before=cp_before, cp_after=cp_after,
                        opening_name=opening_name, missed_tactics=missed_tactics,
                        tactics_enabled=True, previous_move=previous_move,
                        previous_blueprint=previous_blueprint, sid="local",
                    )
                    ctx['ply'] = ply
                    ctx['fen_after'] = board.fen()
                    ctx['cp_after_raw'] = cp_after
                    previous_blueprint = ctx.get("blueprint")
                    previous_move = move
                    app_contexts.append(ctx)

                if eval_mode == "ai_generate":
                    # Generate AI commentary via real pipeline
                    print(f"  > Generating commentary for {len(app_contexts)} moves...")
                    ai_commentaries: Dict[int, str] = {}
                    chunk = 20
                    for i in range(0, len(app_contexts), chunk):
                        ai_commentaries.update(self.batch_generate_commentary(app_contexts[i:i+chunk]))

                    # Write to PGN
                    for node in game.mainline():
                        if node.ply() in ai_commentaries:
                            node.comment = ai_commentaries[node.ply()]
                    with open("commented_test_game.pgn", "a", encoding="utf-8") as out_f:
                        print(game, file=out_f, end="\n\n")
                else:
                    print(f"  > Using existing PGN comments for evaluation...")
                    ai_commentaries = pgn_commentaries

                # Evaluate all moves that got commentary
                print("  > Computing metrics...")
                g_llm = g_recall = g_semantic = g_qual = g_sq = 0.0
                count = 0

                for ctx in app_contexts:
                    ply = ctx['ply']
                    ai_text = ai_commentaries.get(ply, "")
                    if not ai_text:
                        continue
                    count += 1

                    # Pillar 2: Concept Recall
                    cr = self.concept_recall(ctx, ai_text)
                    g_recall += cr

                    # Pillar 3: Semantic Match
                    sm = self.semantic_match(ctx, ai_text)
                    g_semantic += sm

                    # Bonus: Move Quality Agreement
                    qa = self.move_quality_agreement(str(ctx.get("quality", "")), ai_text)
                    g_qual += qa

                    # Bonus: Square Precision
                    sp = self.square_precision(ctx, ai_text)
                    g_sq += sp

                # Pillar 1: LLM Judge (batched, expensive — do selectively)
                if not skip_llm_judge and count > 0:
                    # Judge a sample of moves (every 3rd) to save time
                    judge_plies = [ctx['ply'] for i, ctx in enumerate(app_contexts)
                                   if ai_commentaries.get(ctx['ply']) and i % 3 == 0]
                    print(f"  > LLM Judging {len(judge_plies)} sampled moves...")
                    for i, ctx in enumerate(app_contexts):
                        ply = ctx['ply']
                        if ply not in judge_plies:
                            continue
                        print(f"      [Judge] Grading ply {ply} ({judge_plies.index(ply) + 1}/{len(judge_plies)})...")
                        ai_text = ai_commentaries.get(ply, "")
                        concepts = self._extract_engine_concepts(ctx)
                        bp = ctx.get("blueprint") or ctx.get("commentary_blueprint") or {}
                        kw_text = ", ".join(str(k) for k in bp.get("keywords", []) if k)
                        draft_text = str(ctx.get("draft_commentary", "") or "")
                        score = self.llm_judge_score(
                            ply=ply,
                            fen=ctx.get('fen_after', ''),
                            eval_text=ctx.get('eval', {}).get('formatted_after', '0.00'),
                            move_san=ctx.get('move', {}).get('san', ''),
                            quality=str(ctx.get('quality', '')),
                            tactics=", ".join(concepts) if concepts else "none",
                            ai_text=ai_text,
                            keywords=kw_text,
                            draft=draft_text,
                        )
                        g_llm += score
                    llm_count = len(judge_plies)
                else:
                    llm_count = 0

                if count == 0:
                    continue

                metrics = {
                    "count": count,
                    "avg_llm": g_llm / llm_count if llm_count > 0 else 0.0,
                    "avg_recall": g_recall / count,
                    "avg_semantic": g_semantic / count,
                    "avg_qual": g_qual / count,
                    "avg_sq": g_sq / count,
                    "llm_count": llm_count,
                }
                all_metrics.append(metrics)

                # Sample output
                sample = app_contexts[0]
                sample_text = ai_commentaries.get(sample['ply'], '')
                print(f"  > [Sample Ply {sample['ply']}]")
                print(f"    AI: {sample_text[:150]}...")
                print(f"  > Scores -> LLM: {metrics['avg_llm']:.2f} | "
                      f"ConRecall: {metrics['avg_recall']:.2f} | "
                      f"Semantic: {metrics['avg_semantic']:.2f} | "
                      f"QualAgr: {metrics['avg_qual']:.2f} | "
                      f"SqPrec: {metrics['avg_sq']:.2f}")

        if not all_metrics:
            return

        # Global metrics
        tot = sum(m["count"] for m in all_metrics)
        tot_llm = sum(m["llm_count"] for m in all_metrics)

        def _wavg(key):
            return sum(m[key] * m["count"] for m in all_metrics) / tot

        def _wavg_llm():
            if tot_llm == 0:
                return 0.0
            return sum(m["avg_llm"] * m["llm_count"] for m in all_metrics) / tot_llm

        print("\n" + "=" * 60)
        print("              FINAL GLOBAL METRICS")
        print("=" * 60)
        llm_score = _wavg_llm()
        recall_score = _wavg("avg_recall")
        semantic_score = _wavg("avg_semantic")
        qual_score = _wavg("avg_qual")
        sq_score = _wavg("avg_sq")

        print(f" Pillar 1  LLM Judge Score    : {llm_score:.2f}   (qualitative — precision/engagement/insight)")
        print(f" Pillar 2  Concept Recall     : {recall_score:.2f}   (engine tactics mentioned by AI)")
        print(f" Pillar 3  Semantic Match     : {semantic_score:.2f}   (cosine sim: AI text vs engine PV)")
        print(f" Bonus     Quality Agreement  : {qual_score:.2f}   (blunder/brilliant polarity)")
        print(f" Bonus     Square Precision   : {sq_score:.2f}   (correct squares in AI text)")
        print("=" * 60)

        # Weighted composite — explicit rationale:
        #   LLM Judge 40% — most holistic, captures quality humans care about
        #   Concept Recall 25% — ensures tactical coverage (paper Pillar 2)
        #   Semantic Match 20% — alignment with engine reality (paper Pillar 3)
        #   Square Precision 15% — factual correctness of specific claims
        composite = (
            0.40 * llm_score
            + 0.25 * recall_score
            + 0.20 * semantic_score
            + 0.15 * sq_score
        )
        print(f"  Composite Score: {composite:.2f}")
        print(f"  (0.40×LLM + 0.25×Recall + 0.20×Semantic + 0.15×SqPrec)")
        print("=" * 60)


# ==============================================================================
# CLI
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess Commentary Evaluation Framework")
    parser.add_argument("--pgn", default="Grandmaster Commentry.pgn", help="PGN file to evaluate against")
    parser.add_argument("--judge", choices=["ollama", "gemini", "gemma", "none"], default="ollama",
                        help="LLM judge backend (default: ollama)")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name (default: qwen2.5:3b)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge (faster)")
    parser.add_argument("--eval-mode", choices=["ai_generate", "pgn_comments"], default="ai_generate",
                        help="Whether to generate AI commentary or evaluate existing PGN comments")
    parser.add_argument("--games", type=int, default=1, help="Number of games to evaluate (default: 1, set to 0 for all games)")
    args = parser.parse_args()

    evaluator = ExactAppEvaluator(judge_type=args.judge, judge_model=args.model)
    evaluator.run_evaluation(args.pgn, skip_llm_judge=(args.no_judge or args.judge == "none"), eval_mode=args.eval_mode, max_games=args.games)