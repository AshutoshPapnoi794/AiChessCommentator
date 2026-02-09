import queue
import threading
from typing import Any, Dict, Optional

import chess
from flask import request
from stockfish import Stockfish

from narrative import GameNarrativeMemory, session_memories
from runtime import DEFAULT_ANALYSIS_DEPTH, ENGINE_THREADS, STOCKFISH_PATH, logger, socketio

analysis_queues: Dict[str, "queue.Queue[Optional[Dict[str, Any]]] "] = {}
worker_threads: Dict[str, threading.Thread] = {}
commentary_engines: Dict[str, Stockfish] = {}
commentary_locks: Dict[str, threading.Lock] = {}
tactics_lock = threading.Lock()


def _safe_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def analysis_worker(sid: str, q: "queue.Queue[Optional[Dict[str, Any]]] "):
    stockfish = None
    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": ENGINE_THREADS})
        while True:
            job = q.get()
            if job is None:
                break

            while True:
                try:
                    pending = q.get_nowait()
                except queue.Empty:
                    break
                if pending is None:
                    job = None
                    break
                job = pending
            if job is None:
                break

            fen = job.get('fen')
            if not fen:
                continue
            settings = job.get('settings') or {}
            request_id = job.get('requestId')

            depth = _safe_int(settings.get('depth'), DEFAULT_ANALYSIS_DEPTH, 10, 24)
            threads = _safe_int(settings.get('threads'), ENGINE_THREADS, 1, ENGINE_THREADS)

            stockfish.update_engine_parameters({"Threads": threads})
            stockfish.set_depth(depth)
            stockfish.set_fen_position(fen)
            stockfish.get_best_move()

            evaluation = stockfish.get_evaluation()
            board = chess.Board(fen)
            top_moves = []
            for move_data in stockfish.get_top_moves(3) or []:
                uci = move_data.get('Move')
                if not uci:
                    continue
                try:
                    san = board.san(chess.Move.from_uci(uci))
                except ValueError:
                    san = uci
                top_moves.append({
                    'san': san,
                    'cp': move_data.get('Centipawn'),
                    'mate': move_data.get('Mate'),
                    'uci': uci,
                })

            socketio.emit(
                'analysis_result',
                {
                    'eval': evaluation,
                    'moves': top_moves,
                    'requestId': request_id,
                    'depth': depth,
                    'fen': fen,
                },
                room=sid,
            )
    except Exception as exc:
        logger.error("Analysis worker error for %s: %s", sid, exc)
    finally:
        if stockfish is not None:
            try:
                del stockfish
            except Exception:
                pass


@socketio.on('connect')
def handle_connect():
    sid = request.sid
    session_memories[sid] = GameNarrativeMemory()

    if STOCKFISH_PATH:
        q = queue.Queue()
        analysis_queues[sid] = q
        worker = threading.Thread(target=analysis_worker, args=(sid, q), daemon=True, name=f"analysis-{sid[:8]}")
        worker.start()
        worker_threads[sid] = worker
        try:
            commentary_engines[sid] = Stockfish(path=STOCKFISH_PATH, depth=12, parameters={"Threads": 1})
            commentary_locks[sid] = threading.Lock()
        except Exception as exc:
            logger.warning("Failed to initialize commentary engine for %s: %s", sid, exc)


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    session_memories.pop(sid, None)

    q = analysis_queues.pop(sid, None)
    if q:
        q.put(None)
    worker_threads.pop(sid, None)

    commentary_locks.pop(sid, None)
    engine = commentary_engines.pop(sid, None)
    if engine is not None:
        try:
            del engine
        except Exception:
            pass


@socketio.on('analyze_position')
def handle_analysis(data):
    q = analysis_queues.get(request.sid)
    if q and isinstance(data, dict):
        q.put(data)
