"""
Chess Commentary AI Application
Provides rich, human-like chess commentary with tactical analysis.
Uses Gemini for Text and local KittenTTS for lightning-fast audio.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import os
import queue
import re
import stat
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import chess
import chess.pgn
import requests
import numpy as np
from dotenv import load_dotenv
from flask import Flask, abort, render_template, request
from flask_socketio import SocketIO
from stockfish import Stockfish

from commentary_logic import (
    CommentaryBlueprint,
    blueprint_from_payload,
    build_commentary_blueprint,
    render_blueprint_draft,
)
from structured_commentary import build_structured_commentary
from tactics_analyzer import EnhancedTacticsAnalyzer

try:
    import google.generativeai as genai_text_model
except Exception:
    genai_text_model = None

try:
    from kittentts import KittenTTS
except ImportError:
    KittenTTS = None


load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "a_very_secret_key")
socketio = SocketIO(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================

class TimedLRUCache:
    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[Any, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return value

    def set(self, key: Any, value: Any) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)


def _make_executable(path: str) -> bool:
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)
        return True
    except Exception as exc:
        logger.error("Failed to make %s executable: %s", path, exc)
        return False


def _resolve_stockfish_path() -> Optional[str]:
    candidates = [
        os.getenv("STOCKFISH_PATH"),
        os.path.join("stockfish", "stockfish-ubuntu-x86-64-avx2"),
        os.path.join("stockfish", "stockfish"),
        os.path.join("stockfish", "stockfish-linux-x86-64-avx2"),
    ]
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        if not os.access(path, os.X_OK) and not _make_executable(path):
            continue
        if os.access(path, os.X_OK):
            return path
    return None


STOCKFISH_PATH = _resolve_stockfish_path()
ENGINE_THREADS = max(1, min(2, os.cpu_count() or 2))
DEFAULT_ANALYSIS_DEPTH = 18
COMMENTARY_DEPTH = 14
DEFAULT_TACTICS_ENABLED = True

if STOCKFISH_PATH:
    logger.info("Stockfish initialized at: %s", STOCKFISH_PATH)
else:
    logger.warning("Stockfish binary not available. Engine features are disabled.")


# ==================== AI INITIALIZATION ====================

tactics_analyzer = EnhancedTacticsAnalyzer(engine_path=STOCKFISH_PATH)

# 1. Text AI (Gemini)
text_model = None
if os.getenv("GEMINI_TEXT_API_KEY") and genai_text_model is not None:
    try:
        genai_text_model.configure(api_key=os.getenv("GEMINI_TEXT_API_KEY"))
        text_model = genai_text_model.GenerativeModel("gemini-2.5-flash")
    except Exception as exc:
        logger.error("Gemini init failed: %s", exc)

# 2. Local Audio AI (KittenTTS)
tts_model = None
tts_lock = threading.Lock()
TTS_VOICE = "Bruno"  # You can change to any valid KittenTTS voice
DISABLE_KITTENTTS = os.getenv("DISABLE_KITTENTTS", "").strip().lower() in {"1", "true", "yes"}

if DISABLE_KITTENTTS:
    logger.info("KittenTTS disabled via DISABLE_KITTENTTS.")
elif KittenTTS is not None:
    try:
        logger.info("Loading KittenTTS model...")
        tts_model = KittenTTS("KittenML/kitten-tts-nano-0.8-int8")
        logger.info("KittenTTS loaded successfully.")
    except Exception as exc:
        logger.error("KittenTTS init failed: %s", exc)
else:
    logger.warning("KittenTTS module not found. Run pip install kittentts.")


LICHESS_API_URL = "https://lichess.org/api"
API_HEADERS = {"User-Agent": "LichessGameViewer/1.0"}
if os.getenv("LICHESS_TOKEN"):
    API_HEADERS["Authorization"] = f"Bearer {os.getenv('LICHESS_TOKEN')}"

HTTP_TIMEOUT = (5, 20)
HTTP_SESSION = requests.Session()
GAME_CACHE = TimedLRUCache(max_size=500, ttl_seconds=30 * 60)
USER_GAMES_CACHE = TimedLRUCache(max_size=128, ttl_seconds=3 * 60)
COMMENTARY_AUDIO_CACHE = TimedLRUCache(max_size=1024, ttl_seconds=12 * 60 * 60)
OPENING_BOOK: Dict[str, str] = {}

analysis_queues: Dict[str, "queue.Queue"] = {}
worker_threads: Dict[str, threading.Thread] = {}
commentary_engines: Dict[str, Stockfish] = {}
commentary_locks: Dict[str, threading.Lock] = {}
# Stores the last CommentaryBlueprint dict per session so the real-time
# handler can pass previous_blueprint to build_rich_context, matching the
# behaviour of the batch prefetch handler.
_session_blueprints: Dict[str, Optional[Dict[str, Any]]] = {}


# ==================== UTILITY FUNCTIONS ====================

def load_openings() -> None:
    for filename in ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"]:
        try:
            with open(filename, "r", encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    if i == 0:
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        OPENING_BOOK[parts[2].strip()] = parts[1].strip()
        except FileNotFoundError:
            logger.warning("Opening file not found: %s", filename)


def close_resources() -> None:
    HTTP_SESSION.close()
    tactics_analyzer.close()


def parse_initial_time_seconds(time_control: Optional[str]) -> int:
    if not time_control or time_control in {"-", "?"}:
        return 600
    base = time_control.split("+", 1)[0]
    if "/" in base:
        base = base.split("/", 1)[-1]
    try:
        sec = int(base)
        return sec if sec > 0 else 600
    except ValueError:
        return 600


def lichess_get(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
    url = f"{LICHESS_API_URL}/{path.lstrip('/')}"
    headers = API_HEADERS.copy()
    headers["Accept"] = "application/x-chess-pgn"
    try:
        response = HTTP_SESSION.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        logger.warning("Lichess request failed for %s: %s", url, exc)
        return None


def _safe_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _tactics_enabled_from_settings(settings: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(settings, dict):
        return DEFAULT_TACTICS_ENABLED
    for key in ("tacticsEnabled", "tactics_enabled"):
        if key in settings:
            return _safe_bool(settings.get(key), DEFAULT_TACTICS_ENABLED)
    return DEFAULT_TACTICS_ENABLED


def _bool_from_multidict(source: Any, key: str, default: bool) -> bool:
    if source is None:
        return default
    try:
        values = source.getlist(key)
    except Exception:
        values = []
    if values:
        return _safe_bool(values[-1], default)
    try:
        return _safe_bool(source.get(key), default)
    except Exception:
        return default


def _cp_from_eval(eval_dict: Optional[Dict[str, Any]]) -> int:
    if not eval_dict: return 0
    if eval_dict.get("type") == "cp": return int(eval_dict.get("value", 0))
    if eval_dict.get("type") == "mate":
        mate_val = int(eval_dict.get("value", 0))
        return 10000 if mate_val > 0 else -10000
    return 0


def _format_cp(cp: int) -> str:
    if abs(cp) >= 10000:
        return "M" + ("+" if cp > 0 else "-")
    return f"{cp/100:+.2f}"


def classify_move_quality(cp_before: int, cp_after: int, is_white_move: bool) -> str:
    cp_loss = max(0, cp_before - cp_after) if is_white_move else max(0, cp_after - cp_before)
    if cp_loss <= 10: return "best"
    if cp_loss <= 30: return "excellent"
    if cp_loss <= 70: return "good"
    if cp_loss <= 150: return "inaccuracy"
    if cp_loss <= 350: return "mistake"
    return "blunder"


def sanitize_sentence(text: Any) -> str:
    raw_lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if not raw_lines: return ""
    cleaned_lines = []
    for line in raw_lines:
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"\b(?:none|null)\b", "", line, flags=re.IGNORECASE).strip(" ,;:-")
        if not line: continue
        if line[-1] not in ".!?": line += "."
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def _normalize_commentary_block(text: Any) -> str:
    raw_lines = [line.rstrip() for line in str(text or "").splitlines()]
    cleaned_lines: List[str] = []
    blank_streak = 0
    for raw in raw_lines:
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            blank_streak += 1
            if blank_streak <= 1:
                cleaned_lines.append("")
            continue
        blank_streak = 0
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def is_low_quality_commentary(text: str) -> bool:
    line = (text or "").strip().lower()
    if not line: return True
    banned_phrases = ["move quality tag", "story thread", "opening info", "practical choice", "no hallucinations", "no move trees"]
    if any(p in line for p in banned_phrases): return True
    generic_patterns = [
        r"after this move,\s+(white|black)\s+keeps?\s+(a\s+)?(slight|clear)\s+edge",
        r"after this move,\s+the position stays roughly balanced",
        r"\b(white|black)\s+keeps?\s+(a\s+)?(slight|clear)\s+edge\b",
        r"\b(white|black)\s+has\s+a\s+near-winning\s+advantage\b",
        r"\bthe position stays roughly balanced\b",
        r"^(white|black)\s+(plays|moves)\s+[a-z0-9=+#-]+[.?!]?\s+after this move,",
    ]
    if any(re.search(pattern, line) for pattern in generic_patterns): return True
    if re.fullmatch(r"(white|black)\s+(plays|moves)\s+[a-z0-9=+#-]+[.?!]?", line): return True
    generic_eval_phrases = ["slight edge", "clear edge", "near-winning advantage", "roughly balanced", "after this move"]
    if any(phrase in line for phrase in generic_eval_phrases):
        concept_hits = sum(1 for token in ["center", "central", "develop", "bishop", "knight", "rook", "queen", "king", "pawn", "file", "diagonal", "fork", "pin", "trade", "threat", "castle", "initiative", "d4", "d5", "e4", "e5"] if token in line)
        if concept_hits < 2: return True
    return len(line.split()) < 8


def _normalized_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9+#=-]+", str(text or "").lower())


def _rewrite_keyword_hit_count(text: str, keywords: List[str]) -> int:
    normalized = str(text or "").lower()
    tokens = set(_normalized_words(normalized))
    hits = 0
    for keyword in keywords:
        key = str(keyword or "").strip().lower()
        if not key:
            continue
        key_tokens = [tok for tok in re.findall(r"[a-z0-9+#=-]+", key) if tok]
        if not key_tokens:
            continue
        if len(key_tokens) == 1:
            if key in normalized or key_tokens[0] in tokens:
                hits += 1
        elif all(token in normalized or token in tokens for token in key_tokens):
            hits += 1
    return hits


def _rewrite_preserves_blueprint(candidate: str, context: Dict[str, Any]) -> bool:
    blueprint = context.get("blueprint") or {}
    draft = str(context.get("draft_commentary", "") or "")
    if not candidate or is_low_quality_commentary(candidate): return False
    draft_words, candidate_words = len(_normalized_words(draft)), len(_normalized_words(candidate))
    if candidate_words < max(9, int(draft_words * 0.8)): return False

    keywords = [str(item) for item in blueprint.get("keywords", []) if item]
    if not keywords: return True
    required_hits = min(2, len(keywords))
    return _rewrite_keyword_hit_count(candidate, keywords) >= required_hits


def _color_name(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


def _phase_from_ply(ply: int) -> str:
    if ply <= 12: return "opening"
    if ply <= 70: return "middlegame"
    return "endgame"


# ==================== ENGINE OPERATIONS ====================

def _ensure_commentary_lock(sid: str) -> threading.Lock:
    if sid not in commentary_locks: commentary_locks[sid] = threading.Lock()
    return commentary_locks[sid]


def _restart_commentary_engine(sid: str) -> Optional[Stockfish]:
    if not STOCKFISH_PATH: return None
    old = commentary_engines.pop(sid, None)
    if old:
        try: del old
        except Exception: pass
    try:
        engine = Stockfish(path=STOCKFISH_PATH, depth=COMMENTARY_DEPTH, parameters={"Threads": 1})
        commentary_engines[sid] = engine
        _ensure_commentary_lock(sid)
        return engine
    except Exception as exc:
        logger.error("Failed to restart commentary engine for %s: %s", sid, exc)
        return None


def _engine_call(sid: str, op, default):
    lock = _ensure_commentary_lock(sid)
    for _ in range(2):
        engine = commentary_engines.get(sid)
        if not engine:
            engine = _restart_commentary_engine(sid)
            if not engine: return default
        try:
            with lock: return op(engine)
        except Exception:
            _restart_commentary_engine(sid)
    return default


def get_commentary_eval(sid: str, fen: str, depth: int = COMMENTARY_DEPTH) -> int:
    def _op(engine: Stockfish) -> int:
        engine.set_depth(depth)
        engine.set_fen_position(fen)
        return _cp_from_eval(engine.get_evaluation())
    return _engine_call(sid, _op, default=0)


def get_best_line_san(sid: str, board: chess.Board, max_plies: int = 4, depth: int = COMMENTARY_DEPTH) -> str:
    def _op(engine: Stockfish) -> str:
        temp = board.copy(stack=False)
        san_line = []
        engine.set_depth(depth)
        engine.set_fen_position(temp.fen())
        for _ in range(max_plies):
            uci = engine.get_best_move()
            if not uci: break
            try: move = chess.Move.from_uci(uci)
            except ValueError: break
            if move not in temp.legal_moves: break
            try: san_line.append(temp.san(move))
            except ValueError: san_line.append(uci)
            temp.push(move)
            engine.set_fen_position(temp.fen())
        return " ".join(san_line)
    return _engine_call(sid, _op, default="")


def get_top_moves(sid: str, board: chess.Board, depth: int = COMMENTARY_DEPTH, count: int = 3) -> List[Dict[str, Any]]:
    def _op(engine: Stockfish) -> List[Dict[str, Any]]:
        engine.set_depth(depth)
        engine.set_fen_position(board.fen())
        engine.get_best_move()
        top_moves = engine.get_top_moves(count) or []
        result = []
        for m in top_moves:
            uci = m.get("Move", "")
            if not uci: continue
            try:
                move = chess.Move.from_uci(uci)
                san = board.san(move)
            except ValueError: continue
            
            mate = m.get("Mate")
            if mate is not None: cp_val = 10000 - mate * 100 if mate > 0 else -10000 - mate * 100
            else: cp_val = m.get("Centipawn", 0)
            if board.turn == chess.BLACK: cp_val = -cp_val

            result.append({"uci": uci, "san": san, "cp": cp_val, "mate": mate, "move": move})
        return result
    return _engine_call(sid, _op, default=[])


# ==================== TACTICAL ANALYSIS ====================

def _empty_tactical_analysis() -> Dict[str, Any]:
    return {
        "tactical_patterns": [],
        "strategic_themes": [],
        "attack_info": {},
        "pressure_info": {},
        "is_brilliant": False,
        "brilliant_details": {},
    }


def _build_analysis_after(
    sid: str,
    prev_board: Optional[chess.Board],
    curr_board: chess.Board,
    move: Optional[chess.Move],
    tactics_enabled: bool,
) -> Dict[str, Any]:
    if not tactics_enabled:
        return _empty_tactical_analysis()

    analysis_after = tactics_analyzer.analyze(curr_board.fen(), prev_board.fen() if prev_board else None)
    if move and prev_board:
        analysis_after["attack_info"] = tactics_analyzer.detect_attack(prev_board, curr_board, move)
        analysis_after["pressure_info"] = tactics_analyzer.detect_pressure(prev_board, curr_board, move)
        is_brilliant, brilliant_details = tactics_analyzer.detect_brilliant_move(prev_board, curr_board, move, sid)
        analysis_after["is_brilliant"], analysis_after["brilliant_details"] = is_brilliant, brilliant_details
    return analysis_after

def analyze_best_move_tactics(
    sid: str,
    board: chess.Board,
    best_move: chess.Move,
    depth: int = COMMENTARY_DEPTH,
    tactics_enabled: bool = DEFAULT_TACTICS_ENABLED,
) -> Dict[str, Any]:
    if not tactics_enabled:
        return {}
    if not best_move or best_move not in board.legal_moves: return {}
    moving_piece = board.piece_at(best_move.from_square)
    if not moving_piece: return {}

    board_after = board.copy(stack=False)
    board_after.push(best_move)

    analysis = tactics_analyzer.analyze(board_after.fen(), board.fen())

    is_capture = board.is_capture(best_move)
    captured_piece = None
    if is_capture:
        if board.is_en_passant(best_move): captured_piece = "pawn"
        else:
            cap = board.piece_at(best_move.to_square)
            captured_piece = chess.piece_name(cap.piece_type) if cap else None

    tactical_patterns = analysis.get("tactical_patterns", [])
    
    # Discovered-attack detection: a slider that was NOT already attacking the
    # valuable target before the move (i.e. the attack was "uncovered" by the
    # best move clearing the line).  The old code checked only the post-move
    # board so every pre-existing slider attack was a false positive.
    discovered_attack, discovered_target = False, None
    for attacker_sq in board_after.pieces(chess.QUEEN, board.turn) | board_after.pieces(chess.ROOK, board.turn) | board_after.pieces(chess.BISHOP, board.turn):
        if attacker_sq == best_move.to_square:
            continue  # The moved piece itself is not a discovered attacker
        if attacker_sq == best_move.from_square:
            continue  # Piece left this square; skip ghost reference
        attacker = board_after.piece_at(attacker_sq)
        if not attacker:
            continue
        for target_sq in board_after.attacks(attacker_sq):
            target = board_after.piece_at(target_sq)
            if not target or target.color == board.turn:
                continue
            if target.piece_type not in (chess.KING, chess.QUEEN, chess.ROOK):
                continue
            # Only flag as DISCOVERED if the slider did NOT already attack this
            # square before the best move was played.
            if target_sq not in board.attacks(attacker_sq):
                discovered_attack = True
                discovered_target = chess.piece_name(target.piece_type)
                break
        if discovered_attack:
            break

    attacks_valuable, attacked_piece = False, None
    for target_sq in board_after.attacks(best_move.to_square):
        target = board_after.piece_at(target_sq)
        if target and target.color != board.turn and target.piece_type in [chess.QUEEN, chess.ROOK]:
            attacks_valuable = True
            attacked_piece = chess.piece_name(target.piece_type)
            break

    is_brilliant, brilliant_details = False, {}
    if is_capture or best_move.to_square in board.attacks(best_move.from_square):
        if board_after.is_attacked_by(not board.turn, best_move.to_square):
            is_brilliant, brilliant_details = tactics_analyzer.detect_brilliant_move(board, board_after, best_move, sid)

    return {
        "move_san": board.san(best_move),
        "is_capture": is_capture,
        "captured_piece": captured_piece,
        "gives_check": board_after.is_check(),
        "gives_checkmate": board_after.is_checkmate(),
        "tactic_types": [p.get("type", "") for p in tactical_patterns],
        "discovered_attack": discovered_attack,
        "discovered_target": discovered_target,
        "attacks_valuable": attacks_valuable,
        "attacked_piece": attacked_piece,
        "has_tactics": bool(tactical_patterns) or discovered_attack,
        "attack_info": tactics_analyzer.detect_attack(board, board_after, best_move),
        "pressure_info": tactics_analyzer.detect_pressure(board, board_after, best_move),
        "is_brilliant": is_brilliant,
        "brilliant_details": brilliant_details,
    }


def detect_missed_tactics(
    sid: str,
    board_before: chess.Board,
    played_move: chess.Move,
    cp_before: int,
    cp_after: int,
    quality: str,
    tactics_enabled: bool = DEFAULT_TACTICS_ENABLED,
) -> Dict[str, Any]:
    if not tactics_enabled:
        return {}
    if quality not in ["blunder", "mistake"]: return {}
    top_moves = get_top_moves(sid, board_before, depth=18, count=3)
    if not top_moves: return {}
    
    best = top_moves[0]
    if played_move and best["move"] == played_move: return {}
    
    best_move_analysis = analyze_best_move_tactics(sid, board_before, best["move"], tactics_enabled=tactics_enabled)
    eval_loss = abs(cp_after - cp_before) if board_before.turn == chess.WHITE else abs(cp_before - cp_after)

    return {
        "missed_best_move": best["san"],
        "missed_best_cp": best["cp"],
        "eval_loss": eval_loss,
        "tactical_details": best_move_analysis,
        "summary": _generate_missed_tactic_summary(best["san"], best_move_analysis, quality),
    }


def _generate_missed_tactic_summary(best_san: str, analysis: Dict[str, Any], quality: str) -> str:
    quality_adj = "disastrous" if quality == "blunder" else "significant"
    parts = [f"A {quality_adj} mistake"]

    if analysis.get("gives_checkmate"): parts.append(f"— {best_san} was checkmate!")
    elif analysis.get("is_brilliant"): parts.append(f"— {best_san} was a brilliant sacrifice!")
    elif analysis.get("gives_check") and analysis.get("is_capture"): parts.append(f"— {best_san} wins material with check!")
    elif analysis.get("is_capture"): parts.append(f"— {best_san} wins material")
    elif analysis.get("attack_info", {}).get("is_attacking"): parts.append(f"— {best_san} forces a winning attack!")
    elif analysis.get("tactic_types"): parts.append(f"— {best_san} creates a {analysis['tactic_types'][0].replace('_', ' ')}!")
    else: parts.append(f"— {best_san} was the required move")

    return " ".join(parts)


# ==================== COMMENTARY GENERATION ====================

def get_attackers(board: chess.Board, square: int, by_color: chess.Color) -> List[str]:
    return [f"{chess.piece_name(board.piece_at(a).piece_type)} on {chess.square_name(a)}" for a in board.attackers(by_color, square) if board.piece_at(a)][:6]


def build_verified_facts(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    previous_move: Optional[chess.Move] = None,
) -> Dict[str, Any]:
    facts = {
        "move_from": None, "move_to": None, "moving_piece": None, "captures": None,
        "attacks_enemy_pieces": [], "defends_friendly_pieces": [], "attackers_of_dest": [],
        "defenders_of_dest": [], "gives_check": None, "gives_checkmate": False,
        "is_developing_piece": False, "develops_minor_piece": False, "castling_side": None,
        "central_control": [], "central_support": [], "central_pressure": [],
        "is_fianchetto_prep": False, "is_recapture": False,
    }
    if not move: return facts

    facts["move_from"] = chess.square_name(move.from_square)
    facts["move_to"] = chess.square_name(move.to_square)
    moving_piece = prev_board.piece_at(move.from_square)
    if moving_piece: facts["moving_piece"] = chess.piece_name(moving_piece.piece_type)

    if prev_board.is_capture(move):
        if prev_board.is_en_passant(move): facts["captures"] = "pawn (en passant)"
        else:
            cap_piece = prev_board.piece_at(move.to_square)
            if cap_piece: facts["captures"] = chess.piece_name(cap_piece.piece_type)

    dest_piece = curr_board.piece_at(move.to_square)
    if dest_piece:
        for target_sq in curr_board.attacks(move.to_square):
            target = curr_board.piece_at(target_sq)
            if target:
                if target.color != dest_piece.color: facts["attacks_enemy_pieces"].append(f"{chess.piece_name(target.piece_type)} on {chess.square_name(target_sq)}")
                else: facts["defends_friendly_pieces"].append(f"{chess.piece_name(target.piece_type)} on {chess.square_name(target_sq)}")

    facts["attackers_of_dest"] = get_attackers(curr_board, move.to_square, not prev_board.turn)
    facts["defenders_of_dest"] = get_attackers(curr_board, move.to_square, prev_board.turn)

    if curr_board.is_check():
        checkers = list(curr_board.checkers())
        if checkers:
            c_piece = curr_board.piece_at(checkers[0])
            if c_piece: facts["gives_check"] = f"{chess.piece_name(c_piece.piece_type)} from {chess.square_name(checkers[0])}"

    facts["gives_checkmate"] = curr_board.is_checkmate()
    if prev_board.is_castling(move):
        facts["castling_side"] = "queenside" if chess.square_file(move.to_square) < chess.square_file(move.from_square) else "kingside"

    if moving_piece and moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
        from_rank = chess.square_rank(move.from_square)
        if (prev_board.turn == chess.WHITE and from_rank == 0) or (prev_board.turn == chess.BLACK and from_rank == 7):
            facts["is_developing_piece"] = True
            facts["develops_minor_piece"] = True

    if moving_piece and moving_piece.piece_type == chess.PAWN:
        bishop_square = {
            chess.WHITE: {chess.G3: chess.F1, chess.B3: chess.C1},
            chess.BLACK: {chess.G6: chess.F8, chess.B6: chess.C8},
        }.get(moving_piece.color, {}).get(move.to_square)
        bishop = curr_board.piece_at(bishop_square) if bishop_square is not None else None
        if bishop and bishop.color == moving_piece.color and bishop.piece_type == chess.BISHOP:
            facts["is_fianchetto_prep"] = True

    if previous_move and prev_board.is_capture(move):
        facts["is_recapture"] = move.to_square == previous_move.to_square

    central_control = []
    central_support = []
    central_pressure = []
    for sq in [chess.D4, chess.D5, chess.E4, chess.E5]:
        if sq not in curr_board.attacks(move.to_square):
            continue
        sq_name = chess.square_name(sq)
        occupant = curr_board.piece_at(sq)
        if occupant and occupant.color == prev_board.turn:
            central_support.append(sq_name)
        elif occupant and occupant.color != prev_board.turn:
            central_pressure.append(sq_name)
        else:
            central_control.append(sq_name)

    facts["central_control"] = central_control
    facts["central_support"] = central_support
    facts["central_pressure"] = central_pressure

    return facts


def build_rich_context(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    move_san: str,
    analysis_after: Dict[str, Any],
    cp_before: int,
    cp_after: int,
    opening_name: str,
    missed_tactics: Optional[Dict[str, Any]] = None,
    tactics_enabled: bool = DEFAULT_TACTICS_ENABLED,
    previous_move: Optional[chess.Move] = None,
    previous_blueprint: Optional[Dict[str, Any]] = None,
    sid: str = "",
) -> Dict[str, Any]:
    mover = prev_board.turn
    verified_facts = build_verified_facts(prev_board, curr_board, move, previous_move=previous_move)

    move_info = {
        "san": move_san,
        "uci": move.uci() if move else "",
        "mover": _color_name(mover),
        "piece": verified_facts["moving_piece"] or "piece",
        "from_square": verified_facts["move_from"],
        "to_square": verified_facts["move_to"],
        "is_castling": prev_board.is_castling(move) if move else False,
        "is_check": verified_facts.get("gives_check") is not None,
        "is_checkmate": verified_facts.get("gives_checkmate", False),
    }

    eval_swing = (cp_after - cp_before) if mover == chess.WHITE else (cp_before - cp_after)
    eval_info = {"cp_after": cp_after, "formatted_after": _format_cp(cp_after), "swing": eval_swing, "is_turning_point": abs(cp_after - cp_before) >= 100}

    capture_analysis = tactics_analyzer.analyze_capture(prev_board, curr_board, move) if tactics_enabled and move else {}
    resolved_threats = tactics_analyzer.detect_resolved_threats(prev_board, curr_board, move) if tactics_enabled and move else []
    hung_piece_info = tactics_analyzer.detect_hung_piece(prev_board, curr_board, move) if tactics_enabled and move else None

    tactical_patterns = analysis_after.get("tactical_patterns", [])
    
    primary_tactic = None
    if tactical_patterns and move:
        move_squares = {chess.square_name(move.from_square), chess.square_name(move.to_square)}
        active_types = {"fork", "discovered_attack", "double_check", "removal_of_guard"}
        
        # 1. Direct involvement (highest priority)
        direct_tactics = [t for t in tactical_patterns if any(sq in move_squares for sq in t.get("squares", []))]
        
        # 2. Active tactics (even if square matching fails)
        active_tactics = [t for t in tactical_patterns if t.get("type") in active_types]
        
        if direct_tactics:
            primary_tactic = direct_tactics[0]
        elif active_tactics:
            primary_tactic = active_tactics[0]
        else:
            # If it's a passive pattern (pin/xray) and doesn't involve the moved piece, drop it
            pass

    quality = classify_move_quality(cp_before, cp_after, mover == chess.WHITE)
    tactical_context = {
        "has_tactics": bool(tactical_patterns),
        "primary_tactic": primary_tactic,
        "strategic_themes": analysis_after.get("strategic_themes", []),
        "attack_info": analysis_after.get("attack_info", {}),
        "pressure_info": analysis_after.get("pressure_info", {}),
        "is_brilliant": analysis_after.get("is_brilliant", False),
        "brilliant_details": analysis_after.get("brilliant_details", {}),
        "capture_analysis": capture_analysis,
        "resolved_threats": resolved_threats,
        "hung_piece": hung_piece_info,
    }

    context = {
        "move": move_info,
        "eval": eval_info,
        "quality": quality,
        "tactical": tactical_context,
        "analysis": {
            "side_to_move": "white" if curr_board.turn == chess.WHITE else "black",
            "game_phase": analysis_after.get("game_phase"),
            "king_safety": analysis_after.get("king_safety", {}),
            "piece_activity": analysis_after.get("piece_activity", {}),
            "pawn_structure": analysis_after.get("pawn_structure", {}),
            "weak_squares": analysis_after.get("weak_squares", {}),
            "threats": analysis_after.get("threats", []),
            "strategic_themes": analysis_after.get("strategic_themes", []),
            "tactical_patterns": tactical_patterns,
        },
        "opening_name": opening_name,
        "ply": curr_board.ply(),
        "missed_tactics": missed_tactics or {},
        "verified_facts": verified_facts,
        "previous_move": _serialize_move(prev_board, previous_move) if previous_move else {},
    }

    parsed_previous_blueprint = previous_blueprint if isinstance(previous_blueprint, CommentaryBlueprint) else blueprint_from_payload(previous_blueprint)
    commentary_blueprint = build_commentary_blueprint(
        prev_board=prev_board,
        curr_board=curr_board,
        move=move,
        context=context,
        previous_blueprint=parsed_previous_blueprint,
    )
    context["commentary_blueprint"] = commentary_blueprint.to_dict()
    context["blueprint"] = commentary_blueprint.to_dict()
    context["draft_commentary"] = commentary_blueprint.draft or render_blueprint_draft(commentary_blueprint)
    engine_info = _build_engine_commentary_info(
        sid=sid,
        prev_board=prev_board,
        curr_board=curr_board,
        move=move,
        quality=quality,
    )
    context["engine_info"] = engine_info
    context["structured_commentary"] = build_structured_commentary(
        prev_board=prev_board,
        curr_board=curr_board,
        move=move,
        context=context,
        engine_info=engine_info,
    )
    return context


def _build_engine_commentary_info(
    sid: str,
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    quality: str,
) -> Dict[str, Any]:
    if not sid or not STOCKFISH_PATH:
        return {}

    info: Dict[str, Any] = {}
    best_line_after = get_best_line_san(sid, curr_board, max_plies=4)
    if best_line_after:
        info["best_line_after"] = best_line_after
        info["best_move_after"] = best_line_after.split()[0]

    if move and quality != "best":
        best_line_before = get_best_line_san(sid, prev_board, max_plies=4)
        if best_line_before:
            info["best_line_before"] = best_line_before
            info["best_move_before"] = best_line_before.split()[0]

    return info


def _serialize_move(board: chess.Board, move: chess.Move) -> Dict[str, Any]:
    piece = board.piece_at(move.from_square)
    return {
        "san": board.san(move) if move in board.legal_moves else move.uci(),
        "uci": move.uci(),
        "piece": chess.piece_name(piece.piece_type) if piece else "",
        "from_square": chess.square_name(move.from_square),
        "to_square": chess.square_name(move.to_square),
        "mover": _color_name(board.turn),
    }


def generate_commentary_prompt(context: Dict[str, Any]) -> str:
    move = context["move"]
    blueprint = context.get("blueprint") or context.get("commentary_blueprint") or {}
    draft = str(context.get("draft_commentary", "") or "")
    locked_keywords = ", ".join(str(item) for item in blueprint.get("keywords", []) if item) or "none"
    sentence_limit = max(2, len([part for part in re.split(r"[.!?]+", draft) if part.strip()]))
    lines = [
        "You are editing chess commentary, not analyzing the move.",
        "The chess understanding is already done by the program.",
        "Rewrite the verified draft below so it sounds natural and polished, but do not add new ideas.",
        "",
        f"MOVE: {move['mover']} played {move['san']}",
        f"DRAFT: {draft}",
        f"LEAD: {blueprint.get('lead', '')}",
    ]
    if support := blueprint.get("support", ""):
        lines.append(f"SUPPORT: {support}")
    if verdict := blueprint.get("verdict", ""):
        lines.append(f"VERDICT: {verdict}")
    lines.append(f"KEYWORDS: {locked_keywords}")

    prompt = f"""{chr(10).join(lines)}

RULES:
- Rewrite only. Do not do fresh chess analysis.
- Keep the same meaning as the draft and preserve the locked keywords.
- Do not add any new move, square, line, attacker, defender, or evaluation claim.
- Keep at least as much concrete detail as the draft.
- Use active, human-sounding prose.
- No more than {sentence_limit} sentences.

Commentary:"""
    return prompt


def generate_gemini_commentary(*, context: Dict[str, Any]) -> str:
    if structured := _normalize_commentary_block(context.get("structured_commentary", "")):
        return structured
    if not text_model: return ""
    try:
        resp = text_model.generate_content(generate_commentary_prompt(context))
        line = sanitize_sentence(getattr(resp, "text", "") or "")
        return line if _rewrite_preserves_blueprint(line, context) else ""
    except Exception as exc:
        logger.debug("Gemini commentary generation failed: %s", exc)
        return ""


def _normalize_tts_text(text: Any, max_chars: int = 600) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized: return ""
    if len(normalized) <= max_chars: return normalized
    truncated = normalized[:max_chars]
    if " " in truncated: truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip() or normalized[:max_chars]


def _pcm16le_to_wav(audio_bytes: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Wraps raw PCM bytes in a valid WAV file header so the browser can play it."""
    if len(audio_bytes) % 2 == 1: audio_bytes += b"\x00"
    bits_per_sample = 16
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(audio_bytes)
    return b"".join([
        b"RIFF", (36 + data_size).to_bytes(4, "little", signed=False), b"WAVE", b"fmt ",
        (16).to_bytes(4, "little", signed=False), (1).to_bytes(2, "little", signed=False),
        channels.to_bytes(2, "little", signed=False), sample_rate.to_bytes(4, "little", signed=False),
        byte_rate.to_bytes(4, "little", signed=False), block_align.to_bytes(2, "little", signed=False),
        bits_per_sample.to_bytes(2, "little", signed=False), b"data", data_size.to_bytes(4, "little", signed=False),
    ]) + audio_bytes


def generate_tts_audio_payload(commentary: Any) -> Optional[Dict[str, str]]:
    """Generates audio using the incredibly fast local KittenTTS model."""
    if not tts_model: return None
    tts_text = _normalize_tts_text(commentary)
    if not tts_text: return None

    cache_key = f"kitten|{TTS_VOICE}|{tts_text}"
    if cached := COMMENTARY_AUDIO_CACHE.get(cache_key): return cached

    try:
        with tts_lock:
            audio_array = tts_model.generate(tts_text, voice=TTS_VOICE)
        
        # Convert float32 array [-1.0 to 1.0] to 16-bit PCM integer bytes
        audio_int16 = np.int16(audio_array * 32767)
        audio_bytes = audio_int16.tobytes()
        
        playable_bytes = _pcm16le_to_wav(audio_bytes, sample_rate=24000, channels=1)
        payload = {"audio_data": base64.b64encode(playable_bytes).decode("ascii"), "audio_mime_type": "audio/wav"}
        
        COMMENTARY_AUDIO_CACHE.set(cache_key, payload)
        return payload
    except Exception as exc:
        logger.debug("KittenTTS generation failed: %s", exc)
        return None


def emit_commentary_audio_for_room(sid: str, ply: int, commentary: Any) -> None:
    payload = generate_tts_audio_payload(commentary)
    if not payload:
        socketio.emit("ai_commentary_audio_error", {"ply": ply, "message": "TTS generation failed."}, room=sid)
        return
    socketio.emit("ai_commentary_audio_result", {"ply": ply, "audio_data": payload["audio_data"], "audio_mime_type": payload["audio_mime_type"]}, room=sid)


def generate_fallback_commentary(context: Dict[str, Any]) -> str:
    if structured := _normalize_commentary_block(context.get("structured_commentary", "")):
        return structured
    if draft := str(context.get("draft_commentary", "") or ""):
        return sanitize_sentence(draft)
    blueprint = blueprint_from_payload(context.get("blueprint") or context.get("commentary_blueprint"))
    if blueprint:
        return sanitize_sentence(render_blueprint_draft(blueprint))
    move = context["move"]
    return sanitize_sentence(f"{move['mover']} plays {move['san']} and improves the position.")


def generate_batch_gemini_commentary(contexts: List[Dict[str, Any]], chunk_size: int = 20) -> Dict[int, str]:
    structured = {
        int(ctx["ply"]): _normalize_commentary_block(ctx.get("structured_commentary", ""))
        for ctx in contexts
        if ctx.get("ply") and _normalize_commentary_block(ctx.get("structured_commentary", ""))
    }
    if structured:
        return structured
    if not text_model or not contexts: return {}
    by_ply = {}

    for start in range(0, len(contexts), chunk_size):
        chunk = contexts[start : start + chunk_size]
        context_lines = []
        context_by_ply = {}
        for ctx in chunk:
            move = ctx["move"]
            blueprint = ctx.get("blueprint") or ctx.get("commentary_blueprint") or {}
            draft = str(ctx.get("draft_commentary", "") or "")
            keyword_text = ", ".join(str(item) for item in blueprint.get("keywords", []) if item) or "none"
            line_parts = [
                f"ply={ctx['ply']}",
                f"move={move['mover']} {move['san']}",
                f"draft={draft}",
                f"lead={blueprint.get('lead', '') or '-'}",
                f"support={blueprint.get('support', '') or '-'}",
                f"verdict={blueprint.get('verdict', '') or '-'}",
                f"keywords={keyword_text}",
            ]
            context_by_ply[int(ctx["ply"])] = ctx
            context_lines.append(" | ".join(line_parts))

        prompt = f"""You are editing chess commentary, not analyzing the position.
The chess understanding is already done for you. Rewrite each verified draft into polished commentary.

RULES:
1. Rewrite only; do not add any fresh chess analysis
2. Preserve the meaning and locked keywords for every move
3. Do not add any new square, move, line, attacker, defender, or evaluation claim
4. Keep each commentary at least as concrete as the draft
5. Output ONLY valid JSON: {{"commentaries":[{{"ply":<int>,"text":"<commentary>"}}, ...]}}

PLANS:
{chr(10).join(context_lines)}"""

        try:
            resp = text_model.generate_content(prompt)
            if json_match := re.search(r'\{[\s\S]*\}', getattr(resp, "text", "") or ""):
                for item in json.loads(json_match.group(0)).get("commentaries", []):
                    ply, commentary = item.get("ply"), sanitize_sentence(item.get("text", ""))
                    ctx = context_by_ply.get(int(ply)) if ply else None
                    if ply and commentary and ctx and _rewrite_preserves_blueprint(commentary, ctx):
                        by_ply[int(ply)] = commentary
        except Exception as exc: logger.debug("Batch Gemini generation failed: %s", exc)

    return by_ply


# ==================== ANALYSIS WORKER ====================

def analysis_worker(sid: str, q: "queue.Queue") -> None:
    stockfish = None
    try:
        if not STOCKFISH_PATH: return
        stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": ENGINE_THREADS})
        while True:
            job = q.get()
            if job is None: break
            while True:
                try: pending = q.get_nowait()
                except queue.Empty: break
                if pending is None: job = None; break
                job = pending
            if job is None: break

            fen, settings, request_id = job.get("fen"), job.get("settings") or {}, job.get("requestId")
            if not fen: continue

            depth, threads = _safe_int(settings.get("depth"), DEFAULT_ANALYSIS_DEPTH, 10, 24), _safe_int(settings.get("threads"), ENGINE_THREADS, 1, ENGINE_THREADS)
            stockfish.update_engine_parameters({"Threads": threads})
            stockfish.set_depth(depth)
            stockfish.set_fen_position(fen)
            stockfish.get_best_move()

            board = chess.Board(fen)
            top_moves = []
            for m in stockfish.get_top_moves(3) or []:
                uci = m.get("Move")
                if not uci: continue
                try: san = board.san(chess.Move.from_uci(uci))
                except ValueError: san = uci
                top_moves.append({"san": san, "cp": m.get("Centipawn"), "mate": m.get("Mate"), "uci": uci})

            socketio.emit("analysis_result", {"eval": stockfish.get_evaluation(), "moves": top_moves, "requestId": request_id, "depth": depth, "fen": fen}, room=sid)
    except Exception as exc: logger.error("Analysis worker error for %s: %s", sid, exc)
    finally:
        if stockfish:
            try: del stockfish
            except Exception: pass


# ==================== ROUTES ====================

@app.after_request
def add_static_cache_headers(response):
    if request.path.startswith("/static/"): response.headers.setdefault("Cache-Control", "public, max-age=604800")
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    username = (request.form.get("username") or request.args.get("username") or "").strip()
    page = request.args.get("page", 1, type=int)
    tactics_enabled = _bool_from_multidict(request.values, "tactics_enabled", DEFAULT_TACTICS_ENABLED)
    games, error, newest_timestamp, oldest_timestamp, games_count = [], None, 0, 0, 0

    if username:
        if not re.match(r"^[A-Za-z0-9_.-]{1,50}$", username): error = "Invalid username format."
        else:
            payload = get_games_for_user(username, since=request.args.get("since", type=int), until=request.args.get("until", type=int))
            if not payload: error = f"Could not fetch games for '{username}'."
            elif not payload["games"]: error = "No games found."
            else:
                games, newest_timestamp, oldest_timestamp, games_count = payload["games"], payload["newest_timestamp"], payload["oldest_timestamp"], payload["count"]

    return render_template("index.html", games=games, username=username, error=error, page=page, newest_timestamp=newest_timestamp, oldest_timestamp=oldest_timestamp, games_count=games_count, tactics_enabled=tactics_enabled)

def get_games_for_user(username: str, max_games: int = 10, since: Optional[int] = None, until: Optional[int] = None):
    cache_key = (username.lower(), max_games, since or 0, until or 0)
    if cached := USER_GAMES_CACHE.get(cache_key):
        for gid, pgn in cached.get("pgn_by_id", {}).items(): GAME_CACHE.set(gid, pgn)
        return {"games": cached["games"], "count": cached["count"], "newest_timestamp": cached["newest_timestamp"], "oldest_timestamp": cached["oldest_timestamp"]}

    params = {"max": max_games, "opening": "true", "clocks": "true"}
    if since: params["since"] = since
    if until: params["until"] = until

    response = lichess_get(f"games/user/{username}", params=params)
    if not response or not response.text.strip(): return {"games": [], "count": 0, "newest_timestamp": 0, "oldest_timestamp": 0}

    games_data, pgn_by_id, newest, oldest, pgn_io = [], {}, 0, float("inf"), io.StringIO(response.text)

    while game := chess.pgn.read_game(pgn_io):
        game_id = game.headers.get("Site", "").split("/")[-1]
        if not game_id: continue

        try:
            from datetime import datetime, timezone
            ts_ms = int(datetime.strptime(f"{game.headers.get('UTCDate', '1970.01.01')} {game.headers.get('UTCTime', '00:00:00')}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
            newest, oldest = max(newest, ts_ms), min(oldest, ts_ms)
        except ValueError: pass

        games_data.append({"id": game_id, "white_player": game.headers.get("White", "?"), "black_player": game.headers.get("Black", "?"), "result": game.headers.get("Result", "*"), "date": game.headers.get("UTCDate", "N/A")})
        pgn_text = str(game)
        pgn_by_id[game_id] = pgn_text
        GAME_CACHE.set(game_id, pgn_text)

    if oldest == float("inf"): oldest = 0
    USER_GAMES_CACHE.set(cache_key, {"games": games_data, "count": len(games_data), "newest_timestamp": newest, "oldest_timestamp": oldest, "pgn_by_id": pgn_by_id})
    return {"games": games_data, "count": len(games_data), "newest_timestamp": newest, "oldest_timestamp": oldest}

@app.route("/game/<game_id>")
def view_game(game_id: str):
    tactics_enabled = _bool_from_multidict(request.args, "tactics_enabled", DEFAULT_TACTICS_ENABLED)
    pgn_text = GAME_CACHE.get(game_id)
    if not pgn_text:
        response = lichess_get(f"game/export/{game_id}", params={"moves": "true", "clocks": "true", "opening": "true"})
        if not response or not response.text.strip(): abort(404)
        pgn_text = response.text
        GAME_CACHE.set(game_id, pgn_text)

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game: abort(404)

    board, moves_data, opening_names, pgn_str, last_opening = game.board(), [], ["Starting Position"], "", "Starting Position"

    for node in game.mainline():
        san = board.san(node.move)
        pgn_str += f"{board.fullmove_number}. {san} " if board.turn == chess.WHITE else f"{san} "
        if hit := OPENING_BOOK.get(pgn_str.strip()): last_opening = hit
        opening_names.append(last_opening)
        moves_data.append({"san": san, "clock": node.clock(), "ply": node.ply()})
        board.push(node.move)

    return render_template("game.html", game={"white": {"name": game.headers.get("White"), "rating": game.headers.get("WhiteElo")}, "black": {"name": game.headers.get("Black"), "rating": game.headers.get("BlackElo")}, "start_fen": game.headers.get("FEN"), "moves_data": moves_data, "initial_time_seconds": parse_initial_time_seconds(game.headers.get("TimeControl")), "stockfish_enabled": STOCKFISH_PATH is not None, "opening_names_by_ply": opening_names, "initial_tactics_enabled": tactics_enabled})


# ==================== SOCKET HANDLERS ====================

@socketio.on("connect")
def handle_connect():
    sid = request.sid
    if sid not in analysis_queues:
        analysis_queues[sid] = queue.Queue()
        worker_threads[sid] = threading.Thread(target=analysis_worker, args=(sid, analysis_queues[sid]), daemon=True)
        worker_threads[sid].start()

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if sid in analysis_queues:
        analysis_queues[sid].put(None)
        del analysis_queues[sid]
    if sid in worker_threads: del worker_threads[sid]
    if sid in commentary_engines:
        try: del commentary_engines[sid]
        except Exception: pass
    _session_blueprints.pop(sid, None)

@socketio.on("analyze_position")
def handle_analysis_request(data: Dict[str, Any]):
    sid = request.sid
    if sid in analysis_queues: analysis_queues[sid].put({"fen": data.get("fen"), "settings": data.get("settings"), "requestId": data.get("requestId")})

@socketio.on("get_move_quality")
def handle_move_quality_request(data: Dict[str, Any]):
    sid, fen_before, fen_after, ply, request_id = request.sid, data.get("fen_before"), data.get("fen_after"), data.get("ply", 0), data.get("requestId", 0)
    if not fen_before or not fen_after: return
    
    cp_before, cp_after = get_commentary_eval(sid, fen_before), get_commentary_eval(sid, fen_after)
    socketio.emit("move_quality_result", {"ply": ply, "classification": classify_move_quality(cp_before, cp_after, chess.Board(fen_before).turn == chess.WHITE), "requestId": request_id}, room=sid)

@socketio.on("get_ai_commentary")
def handle_commentary_request(data: Dict[str, Any]):
    sid = request.sid
    settings = data.get("settings") if isinstance(data, dict) else {}
    tactics_enabled = _tactics_enabled_from_settings(settings)
    fen_before, fen_after, move_san, ply, opening_name, audio_enabled = data.get("previous_fen") or data.get("fen_before"), data.get("current_fen") or data.get("fen_after"), data.get("humanMove") or data.get("move_san", ""), data.get("ply", 0), data.get("opening", "Unknown"), bool(data.get("audio_enabled"))
    prev_prev_fen = data.get("previous_previous_fen")
    previous_move_san = data.get("previous_move_san", "")

    if not fen_after: return
    prev_board, curr_board = chess.Board(fen_before) if fen_before else None, chess.Board(fen_after)
    prev_prev_board = chess.Board(prev_prev_fen) if prev_prev_fen else None

    move = None
    if prev_board and move_san:
        try: move = prev_board.parse_san(move_san)
        except ValueError: pass
    if not move and prev_board: move = _find_move_between_boards(prev_board, curr_board)

    previous_move = None
    if prev_prev_board and prev_board and previous_move_san:
        try: previous_move = prev_prev_board.parse_san(previous_move_san)
        except ValueError: previous_move = None
    if not previous_move and prev_prev_board and prev_board:
        previous_move = _find_move_between_boards(prev_prev_board, prev_board)

    cp_before, cp_after = get_commentary_eval(sid, fen_before) if fen_before else 0, get_commentary_eval(sid, fen_after)
    quality = classify_move_quality(cp_before, cp_after, prev_board.turn == chess.WHITE if prev_board else True)

    missed_tactics = {}
    if quality in ["blunder", "mistake"] and prev_board and move:
        try: missed_tactics = detect_missed_tactics(sid, prev_board, move, cp_before, cp_after, quality, tactics_enabled=tactics_enabled)
        except Exception as e: logger.debug(f"Missed tactics error: {e}")

    analysis_after = _build_analysis_after(sid, prev_board, curr_board, move, tactics_enabled)

    # Guard: if we have no previous board, we cannot meaningfully describe a
    # move (prev_board.turn would be curr_board.turn, i.e. the NEXT player,
    # which inverts the mover name in all commentary).  Fall back to a
    # minimal response instead of generating inverted commentary.
    if prev_board is None:
        socketio.emit("ai_commentary_text_result", {"commentary": "", "ply": ply, "move": move_san, "quality": quality}, room=sid)
        return

    previous_blueprint = _session_blueprints.get(sid)
    context = build_rich_context(prev_board=prev_board, curr_board=curr_board, move=move, move_san=move_san, analysis_after=analysis_after, cp_before=cp_before, cp_after=cp_after, opening_name=opening_name, missed_tactics=missed_tactics, tactics_enabled=tactics_enabled, previous_move=previous_move, previous_blueprint=previous_blueprint, sid=sid)
    _session_blueprints[sid] = context.get("blueprint")

    commentary = generate_gemini_commentary(context=context) if text_model else ""
    if not commentary: commentary = generate_fallback_commentary(context)

    socketio.emit("ai_commentary_text_result", {"commentary": commentary, "ply": ply, "move": move_san, "quality": "brilliant" if context.get("tactical", {}).get("is_brilliant") else quality}, room=sid)
    if audio_enabled and commentary: socketio.start_background_task(emit_commentary_audio_for_room, sid, ply, commentary)

@socketio.on("synthesize_commentary_audio")
def handle_synthesize_commentary_audio(data: Dict[str, Any]):
    sid = request.sid
    if not isinstance(data, dict): return
    if commentary := _normalize_tts_text(data.get("commentary", "")):
        try: ply = int(data.get("ply", 0))
        except (TypeError, ValueError): ply = 0
        socketio.start_background_task(emit_commentary_audio_for_room, sid, ply, commentary)


def _prewarm_audio_cache(commentaries: Dict[int, str]):
    """Background task to pre-generate audio for all moves using KittenTTS."""
    logger.info(f"Pre-warming audio cache for {len(commentaries)} moves...")
    for text in commentaries.values():
        generate_tts_audio_payload(text)
    logger.info("Audio cache pre-warming complete!")


@socketio.on("prefetch_ai_commentary_batch")
def handle_batch_commentary_request(data: Dict[str, Any]):
    sid = request.sid
    try:
        settings = data.get("settings") if isinstance(data, dict) else {}
        tactics_enabled = _tactics_enabled_from_settings(settings)
        fens, moves, openings, game_key = data.get("fens", []), data.get("moves", []), data.get("openings", []), data.get("gameKey", "")
        if not fens or not moves:
            socketio.emit("batch_commentary_error", {"error": "Missing game data", "gameKey": game_key}, room=sid)
            return

        contexts, total = [], min(len(moves), len(fens) - 1)
        previous_move = None
        previous_blueprint = None
        socketio.emit("batch_commentary_progress", {"progress": 5, "message": f"Preparing commentary for {total} moves...", "gameKey": game_key}, room=sid)

        for ply in range(1, total + 1):
            try:
                fen_before, fen_after, move_san, opening_name = fens[ply - 1], fens[ply], moves[ply - 1], openings[ply] if ply < len(openings) else "Unknown"
                prev_board, curr_board = chess.Board(fen_before), chess.Board(fen_after)
                move = _find_move_between_boards(prev_board, curr_board)

                cp_before, cp_after = get_commentary_eval(sid, fen_before), get_commentary_eval(sid, fen_after)
                quality = classify_move_quality(cp_before, cp_after, prev_board.turn == chess.WHITE)
                
                missed_tactics = {}
                if quality in ["blunder", "mistake"]:
                    try: missed_tactics = detect_missed_tactics(sid, prev_board, move, cp_before, cp_after, quality, tactics_enabled=tactics_enabled)
                    except Exception: pass

                analysis_after = _build_analysis_after(sid, prev_board, curr_board, move, tactics_enabled)

                context = build_rich_context(
                    prev_board=prev_board,
                    curr_board=curr_board,
                    move=move,
                    move_san=move_san,
                    analysis_after=analysis_after,
                    cp_before=cp_before,
                    cp_after=cp_after,
                    opening_name=opening_name,
                    missed_tactics=missed_tactics,
                    tactics_enabled=tactics_enabled,
                    previous_move=previous_move,
                    previous_blueprint=previous_blueprint,
                    sid=sid,
                )
                contexts.append(context)
                previous_move = move
                previous_blueprint = context.get("blueprint")

                if ply % 5 == 0 or ply == total: socketio.emit("batch_commentary_progress", {"progress": int(5 + (ply / total) * 50), "message": f"Prepared commentary data for {ply}/{total} moves...", "gameKey": game_key}, room=sid)
            except Exception as e:
                logger.error(f"Error analyzing move {ply}: {e}"); continue

        socketio.emit("batch_commentary_progress", {"progress": 60, "message": "Generating commentary...", "gameKey": game_key}, room=sid)
        commentaries = generate_batch_gemini_commentary(contexts) if text_model and contexts else {}

        move_qualities = {}
        for i, ctx in enumerate(contexts):
            ply = ctx.get("ply", i + 1)
            if ply not in commentaries: commentaries[ply] = generate_fallback_commentary(ctx)
            move_qualities[ply] = "brilliant" if ctx.get("tactical", {}).get("is_brilliant") else ctx.get("quality", "good")

        socketio.emit("batch_commentary_ready", {"commentaries": commentaries, "move_qualities": move_qualities, "gameKey": game_key}, room=sid)
        
        # PROACTIVELY PRE-GENERATE ALL AUDIO IN THE BACKGROUND
        if tts_model:
            socketio.start_background_task(_prewarm_audio_cache, commentaries)
            
    except Exception as e:
        socketio.emit("batch_commentary_error", {"error": str(e), "gameKey": data.get("gameKey", "")}, room=sid)

def _find_move_between_boards(prev_board: chess.Board, curr_board: chess.Board) -> Optional[chess.Move]:
    for move in prev_board.legal_moves:
        probe = prev_board.copy(stack=False)
        probe.push(move)
        if probe.board_fen() == curr_board.board_fen() and probe.turn == curr_board.turn: return move
    return None

load_openings()
atexit.register(close_resources)

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)