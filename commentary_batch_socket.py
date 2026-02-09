import threading
from typing import Any, Dict

from flask import request

from commentary_batch import build_batch_contexts, generate_batch_commentary
from runtime import logger, socketio, text_model


BATCH_CACHE_LOCK = threading.Lock()
BATCH_COMMENTARY_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
BATCH_IN_FLIGHT: Dict[str, str] = {}


def _batch_progress_emit(sid: str, game_key: str, pct: int, message: str) -> None:
    socketio.emit(
        'batch_commentary_progress',
        {
            'gameKey': game_key,
            'progress': max(0, min(100, int(pct))),
            'message': message,
        },
        room=sid,
    )


def _run_batch_prefetch(
    sid: str,
    game_key: str,
    fens: list[str],
    moves: list[Any],
    openings: list[Any],
) -> None:
    try:
        _batch_progress_emit(sid, game_key, 1, "Starting full-game analysis...")

        contexts, move_quality_map = build_batch_contexts(
            sid=sid,
            fens=fens,
            moves=moves,
            openings=openings,
            progress_cb=lambda pct, msg: _batch_progress_emit(sid, game_key, pct, msg),
        )
        commentary_map = generate_batch_commentary(
            text_model=text_model,
            contexts=contexts,
            progress_cb=lambda pct, msg: _batch_progress_emit(sid, game_key, pct, msg),
        )

        payload = {
            'commentaries': {str(ply): line for ply, line in commentary_map.items()},
            'move_qualities': {str(ply): quality for ply, quality in move_quality_map.items()},
        }
        with BATCH_CACHE_LOCK:
            sid_cache = BATCH_COMMENTARY_CACHE.setdefault(sid, {})
            sid_cache[game_key] = payload

        _batch_progress_emit(sid, game_key, 100, "Commentary ready.")
        socketio.emit(
            'batch_commentary_ready',
            {
                'gameKey': game_key,
                **payload,
            },
            room=sid,
        )
    except Exception as exc:
        logger.error("Batch commentary prefetch failed for %s (%s): %s", sid, game_key, exc)
        socketio.emit(
            'batch_commentary_error',
            {
                'gameKey': game_key,
                'message': 'Failed to precompute game commentary.',
            },
            room=sid,
        )
    finally:
        with BATCH_CACHE_LOCK:
            if BATCH_IN_FLIGHT.get(sid) == game_key:
                BATCH_IN_FLIGHT.pop(sid, None)


@socketio.on('prefetch_ai_commentary_batch')
def handle_prefetch_ai_commentary_batch(data):
    sid = request.sid
    if not isinstance(data, dict):
        socketio.emit(
            'batch_commentary_error',
            {'gameKey': '', 'message': 'Invalid batch payload.'},
            room=sid,
        )
        return

    game_key = str(data.get('gameKey') or '').strip()
    fens = data.get('fens')
    moves = data.get('moves')
    openings = data.get('openings') or []
    force = bool(data.get('force'))

    if not game_key or not isinstance(fens, list) or not isinstance(moves, list):
        socketio.emit(
            'batch_commentary_error',
            {'gameKey': game_key, 'message': 'Incomplete batch payload.'},
            room=sid,
        )
        return
    if len(moves) > 320 or len(fens) > 321:
        socketio.emit(
            'batch_commentary_error',
            {'gameKey': game_key, 'message': 'Game too large for single-pass commentary prefetch.'},
            room=sid,
        )
        return

    with BATCH_CACHE_LOCK:
        cached = BATCH_COMMENTARY_CACHE.get(sid, {}).get(game_key)
        in_flight_key = BATCH_IN_FLIGHT.get(sid)
        if cached and not force:
            socketio.emit(
                'batch_commentary_ready',
                {
                    'gameKey': game_key,
                    **cached,
                },
                room=sid,
            )
            return
        if in_flight_key == game_key and not force:
            _batch_progress_emit(sid, game_key, 5, "Batch commentary already running...")
            return
        BATCH_IN_FLIGHT[sid] = game_key

    socketio.start_background_task(
        _run_batch_prefetch,
        sid,
        game_key,
        [str(f) for f in fens],
        moves,
        openings if isinstance(openings, list) else [],
    )


@socketio.on('disconnect')
def clear_batch_commentary_cache():
    sid = request.sid
    with BATCH_CACHE_LOCK:
        BATCH_COMMENTARY_CACHE.pop(sid, None)
        BATCH_IN_FLIGHT.pop(sid, None)
