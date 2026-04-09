import os
import json
import re
import chess
import chess.pgn
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv()

# ==============================================================================
# IMPORT EXACT LOGIC DIRECTLY FROM YOUR APP
# ==============================================================================
from app import (
    classify_move_quality,
    build_rich_context,
    tactics_analyzer,    
    STOCKFISH_PATH
)

from stockfish import Stockfish
from sentence_transformers import SentenceTransformer, util

# ==============================================================================
# 🔑 MULTIPLE API KEYS SETUP
# ==============================================================================
GEMINI_API_KEYS = [
    "AIzaSyDYpql0My_QlYNRIfRH9NREwWurBcwuce0", # Grabs your main key from .env
    "AIzaSyDrKbSXjUgC8Jrk4SD6hjcz_WjVq51bbbg",         # Add more keys here as strings
    "AIzaSyArkelQjh2K7zJ7CrWQeLISVzfQbC8K6fU"
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

def call_gemini_with_rotation(prompt: str) -> str:
    """Switches API keys sequentially before making a request."""
    genai.configure(api_key=key_rotator.get_next_key())
    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"      [!] Gemini API Error: {e}")
        return ""

# ==============================================================================

print("Loading Advanced Semantic Evaluation Model (MPNet)...")
similarity_model = SentenceTransformer('all-mpnet-base-v2')

class ExactAppEvaluator:
    def __init__(self):
        if not STOCKFISH_PATH:
            raise RuntimeError("Stockfish path not found in app.py! Ensure it is configured.")
        self.engine = Stockfish(path=STOCKFISH_PATH, depth=14, parameters={"Threads": 2})

        self.square_regex = re.compile(r'\b[a-h][1-8]\b')
        self.pieces = {"king", "queen", "rook", "bishop", "knight", "pawn"}
        self.chess_concepts = {
            "center", "develop", "pin", "fork", "skewer", "sacrifice", "initiative", 
            "blunder", "mistake", "checkmate", "attack", "defend", "castle", "hang", "trade"
        }

    def _extract_chess_entities(self, text: str) -> set:
        text = text.lower()
        squares = set(self.square_regex.findall(text))
        found_pieces = {p for p in self.pieces if p in text}
        found_concepts = {c for c in self.chess_concepts if c in text}
        return squares | found_pieces | found_concepts

    def evaluate_local_metrics(self, human_text: str, ai_text: str) -> Dict[str, float]:
        emb1 = similarity_model.encode(human_text, convert_to_tensor=True)
        emb2 = similarity_model.encode(ai_text, convert_to_tensor=True)
        semantic_score = max(0.0, util.pytorch_cos_sim(emb1, emb2).item())

        human_entities = self._extract_chess_entities(human_text)
        ai_entities = self._extract_chess_entities(ai_text)
        concept_recall = 1.0 if not human_entities else len(human_entities.intersection(ai_entities)) / len(human_entities)

        return {"semantic_similarity": semantic_score, "concept_recall": concept_recall}

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

    def batch_generate_commentary(self, contexts: List[Dict]) -> Dict[int, str]:
        if not contexts: return {}
        
        context_lines = []
        for ctx in contexts:
            move, tactical, verified = ctx["move"], ctx.get("tactical", {}), ctx.get("verified_facts", {})
            line_parts = [f"ply={ctx['ply']}", f"move={move['san']}", f"mover={move['mover']}"]

            if tactical.get("is_brilliant"): line_parts.append("brilliant=true")
            if hung := tactical.get("hung_piece"): 
                if ctx.get("quality") in ["blunder", "mistake"]: line_parts.append(f"hangs={hung['piece']}")
            if capture_info := tactical.get("capture_analysis"):
                if capture_info.get("type") == "equal_trade": line_parts.append(f"trades_for={capture_info.get('target')}")
                elif capture_info.get("type") in ["free_capture", "favorable_trade"]: line_parts.append(f"wins_material={capture_info.get('target')}")
            if resolved := tactical.get("resolved_threats"): line_parts.append(f"defends_attacked={resolved[0]}")
            if attack_info := tactical.get("attack_info"):
                if attack_info.get("is_attacking"):
                    if attack_info.get("attacker_piece", "piece") != move["piece"] and verified.get("gives_check"):
                        line_parts.append("zwischenzug=true")
                        line_parts.append(f"threatens={attack_info.get('target_piece')}")
                    elif attack_info.get("attack_type") == "trade_offer": line_parts.append(f"offers_trade={attack_info.get('target_piece')}")
                    elif attack_info.get("attack_type") == "favorable_trade": line_parts.append(f"attacks_valuable={attack_info.get('target_piece')}")
                    else: line_parts.append(f"threatens={attack_info.get('target_piece')}")
            if themes := tactical.get("strategic_themes", []): line_parts.append(f"strategy={themes[0]}")
            if verified.get("gives_checkmate"): line_parts.append("mate=true")
            if missed := ctx.get("missed_tactics"):
                if best_move := missed.get("missed_best_move"): line_parts.append(f"MISSED={best_move}")

            context_lines.append(" | ".join(line_parts))

        prompt = f"""You are a chess grandmaster commentator. Write natural, concise commentary for each move.

RULES:
1. Be CONVERSATIONAL: "White develops the knight" not "The knight is developed"
2. If "hangs=" exists, explicitly state they blundered and hung that piece
3. If "zwischenzug=true", call it a brilliant in-between check
4. If "trades_for=" or "offers_trade=", mention the trade
5. If "defends_attacked=", mention they saved the piece
6. NEVER invent or hallucinate piece locations, defenders, or attackers that are not explicitly provided.
7. Include at least one concrete chess idea
8. Output ONLY valid JSON: {{"commentaries":[{{"ply":<int>,"text":"<commentary>"}}, ...]}}

MOVES:
{chr(10).join(context_lines)}"""

        resp_text = call_gemini_with_rotation(prompt)
        by_ply = {}
        if resp_text:
            match = re.search(r'\{[\s\S]*\}', resp_text)
            if match:
                try:
                    for item in json.loads(match.group(0)).get("commentaries", []):
                        by_ply[int(item.get("ply"))] = str(item.get("text"))
                except Exception as e: print(f"JSON Parse error: {e}")
        return by_ply

    def batch_llm_judge(self, comparisons: List[Dict]) -> Dict[int, float]:
        if not comparisons: return {}
        prompt_lines = [f"Ply {c['ply']} | Human: '{c['human']}' | AI: '{c['ai']}'" for c in comparisons]
        prompt = f"""You are an expert chess evaluator. Compare the AI's commentary to the Human's actual commentary.
Grade the AI on a scale of 0.0 to 10.0 based on CHESS NARRATIVE ACCURACY. Do they describe the same chess action or consequence? Ignore stylistic differences.
Output ONLY valid JSON. Format: {{"scores": [{{"ply": 1, "score": 8.5}}, ...]}}
DATA:
{chr(10).join(prompt_lines)}"""

        scores = {}
        resp_text = call_gemini_with_rotation(prompt)
        if resp_text:
            match = re.search(r'\{[\s\S]*\}', resp_text)
            if match:
                try:
                    for item in json.loads(match.group(0)).get("scores", []):
                        scores[item["ply"]] = float(item["score"]) / 10.0 
                except Exception: pass
        for c in comparisons:
            if c['ply'] not in scores: scores[c['ply']] = self.evaluate_local_metrics(c['human'], c['ai'])['semantic_similarity']
        return scores

    def run_evaluation(self, pgn_file_path: str):
        print(f"\n{'='*60}")
        print(f" TESTING ACTUAL APP.PY LOGIC (API OPTIMIZED)")
        print(f"{'='*60}")
        
        all_games_metrics = []
        game_count = 0

        with open(pgn_file_path, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if not game: break 
                
                game_count += 1
                print(f"\n--- Analyzing Game {game_count}: {game.headers.get('White')} vs {game.headers.get('Black')} ---")

                board = game.board()
                human_comments = {}
                app_contexts = []
                
                for node in game.mainline():
                    ply = node.ply()
                    move = node.move
                    
                    # 1. Check if human actually commented on this move
                    has_human_comment = False
                    if node.comment:
                        clean_comment = re.sub(r'\[%clk[^\]]+\]', '', node.comment).strip()
                        if clean_comment: 
                            human_comments[ply] = clean_comment
                            has_human_comment = True

                    # 2. ONLY DO HEAVY PROCESSING IF THERE IS A COMMENT!
                    if has_human_comment:
                        cp_before = self._get_eval_cp(board.fen())
                        
                        prev_board = board.copy()
                        board.push(move) # Update board
                        
                        cp_after = self._get_eval_cp(board.fen())
                        quality = classify_move_quality(cp_before, cp_after, prev_board.turn == chess.WHITE)
                        
                        analysis_after = self._run_app_analysis(prev_board, board, move)
                        
                        ctx = build_rich_context(
                            prev_board=prev_board, curr_board=board, move=move, 
                            move_san=prev_board.san(move), analysis_after=analysis_after, 
                            cp_before=cp_before, cp_after=cp_after, 
                            opening_name="Opening", tactics_enabled=True
                        )
                        ctx['ply'] = ply
                        app_contexts.append(ctx)
                    else:
                        # Just push the move to keep the FEN state accurate for the next ply
                        board.push(move)

                print(f"  > Found {len(human_comments)} human annotations. Skipped API/Engine calls for all other moves.")
                if not human_comments: continue

                print(f"  > Processing {len(app_contexts)} contexts through your app's exact prompt...")
                ai_commentaries = {}
                chunk_size = 20
                for i in range(0, len(app_contexts), chunk_size):
                    ai_commentaries.update(self.batch_generate_commentary(app_contexts[i:i+chunk_size]))

                comparisons = [{"ply": p, "human": h, "ai": ai_commentaries.get(p, "")} for p, h in human_comments.items() if ai_commentaries.get(p, "")]
                if not comparisons: continue

                print("  > Evaluating results...")
                llm_scores = {}
                for i in range(0, len(comparisons), chunk_size):
                    llm_scores.update(self.batch_llm_judge(comparisons[i:i+chunk_size]))

                game_semantic, game_concept, game_llm = 0.0, 0.0, 0.0
                for c in comparisons:
                    mets = self.evaluate_local_metrics(c['human'], c['ai'])
                    game_semantic += mets['semantic_similarity']
                    game_concept += mets['concept_recall']
                    game_llm += llm_scores.get(c['ply'], 0.0)

                num = len(comparisons)
                metrics = {"count": num, "avg_sem": game_semantic/num, "avg_con": game_concept/num, "avg_llm": game_llm/num}
                all_games_metrics.append(metrics)
                
                print(f"  > [Sample Ply {comparisons[0]['ply']}]")
                print(f"    Human: {comparisons[0]['human']}\n    AI   : {comparisons[0]['ai']}")
                print(f"  > Game Score -> LLM Judge: {metrics['avg_llm']:.2f}")

        if not all_games_metrics: return

        tot = sum(m["count"] for m in all_games_metrics)
        print("\n" + "="*60 + "\n                 FINAL GLOBAL METRICS                 \n" + "="*60)
        print(f" 1. LLM Judge Score    : {sum(m['avg_llm'] * m['count'] for m in all_games_metrics) / tot:.2f}")
        print(f" 2. Concept Recall     : {sum(m['avg_con'] * m['count'] for m in all_games_metrics) / tot:.2f}")
        print(f" 3. NLP Semantic Match : {sum(m['avg_sem'] * m['count'] for m in all_games_metrics) / tot:.2f}")
        print("="*60)

if __name__ == "__main__":
    TEST_PGN = "test_game.pgn"
    evaluator = ExactAppEvaluator()
    evaluator.run_evaluation(TEST_PGN)