import threading
from typing import Any, Dict, Optional

import chess
from stockfish import Stockfish

from analysis_socket import commentary_engines, commentary_locks
from runtime import STOCKFISH_PATH, logger

COMMENTARY_ENGINE_THREADS = 1
COMMENTARY_ENGINE_DEFAULT_DEPTH = 12


def _ensure_commentary_lock(sid: str):
    lock = commentary_locks.get(sid)
    if lock is None:
        lock = threading.Lock()
        commentary_locks[sid] = lock
    return lock


def _restart_commentary_engine(sid: str, reason: str = "") -> Optional[Stockfish]:
    if not STOCKFISH_PATH:
        return None

    old_engine = commentary_engines.pop(sid, None)
    if old_engine is not None:
        try:
            del old_engine
        except Exception:
            pass

    try:
        new_engine = Stockfish(
            path=STOCKFISH_PATH,
            depth=COMMENTARY_ENGINE_DEFAULT_DEPTH,
            parameters={"Threads": COMMENTARY_ENGINE_THREADS},
        )
        commentary_engines[sid] = new_engine
        _ensure_commentary_lock(sid)
        if reason:
            logger.warning("Restarted commentary engine for %s after failure: %s", sid, reason)
        else:
            logger.warning("Restarted commentary engine for %s.", sid)
        return new_engine
    except Exception as exc:
        logger.error("Failed to restart commentary engine for %s: %s", sid, exc)
        return None


def _run_with_commentary_engine_retry(sid: str, op, default):
    lock = _ensure_commentary_lock(sid)
    last_exc: Optional[Exception] = None

    for attempt in range(2):
        engine = commentary_engines.get(sid)
        if engine is None:
            engine = _restart_commentary_engine(sid, reason="engine missing")
            if engine is None:
                return default

        try:
            with lock:
                return op(engine)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Commentary engine call failed for %s (attempt %s/2): %s",
                sid,
                attempt + 1,
                exc,
            )
            _restart_commentary_engine(sid, reason=str(exc))

    if last_exc is not None:
        logger.error("Commentary engine call ultimately failed for %s: %s", sid, last_exc)
    return default


def get_cp_val(eval_dict):
    if not eval_dict:
        return 0
    eval_type = eval_dict.get('type')
    eval_value = eval_dict.get('value', 0)
    if eval_type == 'cp':
        return int(eval_value)
    if eval_type == 'mate':
        return 2000 if eval_value > 0 else -2000
    return 0


def classify_move_quality(cp_before: int, cp_after: int, is_white_move: bool) -> str:
    cp_loss = (cp_before - cp_after) if is_white_move else (cp_after - cp_before)
    clamped_cp_loss = max(0, cp_loss)

    is_winning_before = cp_before > 200 if is_white_move else cp_before < -200
    is_not_winning_after = (
        -200 <= cp_after <= 200
        or (is_white_move and cp_after < -200)
        or (not is_white_move and cp_after > 200)
    )
    if is_winning_before and is_not_winning_after:
        return 'blunder'

    if clamped_cp_loss <= 15:
        return 'best'
    if clamped_cp_loss <= 40:
        return 'excellent'
    if clamped_cp_loss <= 90:
        return 'good'
    if clamped_cp_loss <= 200:
        return 'inaccuracy'
    if clamped_cp_loss <= 450:
        return 'mistake'
    return 'blunder'


def get_commentary_eval(sid: str, fen: str) -> int:
    def _op(engine: Stockfish) -> int:
        engine.set_depth(COMMENTARY_ENGINE_DEFAULT_DEPTH)
        engine.set_fen_position(fen)
        return get_cp_val(engine.get_evaluation())

    return _run_with_commentary_engine_retry(sid, _op, default=0)


def _cp_for_color(cp_white: int, color: chess.Color) -> int:
    return cp_white if color == chess.WHITE else -cp_white


def engine_eval_cp_for_board(sid: str, board: chess.Board, depth: int = 12) -> Optional[int]:
    def _op(engine: Stockfish) -> int:
        engine.set_depth(depth)
        engine.set_fen_position(board.fen())
        return get_cp_val(engine.get_evaluation())

    return _run_with_commentary_engine_retry(sid, _op, default=None)


def engine_best_line_san(sid: str, board: chess.Board, max_plies: int = 4, depth: int = 12) -> str:
    def _op(engine: Stockfish) -> str:
        temp = board.copy(stack=False)
        san_line = []
        engine.set_depth(depth)
        engine.set_fen_position(temp.fen())
        for _ in range(max_plies):
            uci = engine.get_best_move()
            if not uci:
                break
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in temp.legal_moves:
                break
            try:
                san_line.append(temp.san(move))
            except ValueError:
                san_line.append(uci)
            temp.push(move)
            engine.set_fen_position(temp.fen())
        return " ".join(san_line)

    return _run_with_commentary_engine_retry(sid, _op, default="")


def engine_eval_cp_for_color(sid: str, board: chess.Board, color: chess.Color, depth: int = 12) -> Optional[int]:
    cp_white = engine_eval_cp_for_board(sid, board, depth=depth)
    if cp_white is None:
        return None
    return _cp_for_color(cp_white, color)


def engine_eval_cp_after_move_for_color(
    sid: str,
    board: chess.Board,
    move: chess.Move,
    color: chess.Color,
    depth: int = 12,
) -> Optional[int]:
    if move not in board.legal_moves:
        return None
    board_after = board.copy(stack=False)
    board_after.push(move)
    return engine_eval_cp_for_color(sid, board_after, color, depth=depth)


def engine_best_move_and_eval_for_color(
    sid: str,
    board: chess.Board,
    color: chess.Color,
    depth: int = 12,
) -> Optional[Dict[str, Any]]:
    def _op(engine: Stockfish) -> Optional[Dict[str, Any]]:
        engine.set_depth(depth)
        engine.set_fen_position(board.fen())
        uci = engine.get_best_move()
        if not uci:
            return None
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None

        board_after = board.copy(stack=False)
        board_after.push(move)
        engine.set_fen_position(board_after.fen())
        cp_white = get_cp_val(engine.get_evaluation())

        try:
            san = board.san(move)
        except ValueError:
            san = move.uci()

        return {
            'move': move,
            'san': san,
            'cp': _cp_for_color(cp_white, color),
        }

    return _run_with_commentary_engine_retry(sid, _op, default=None)
