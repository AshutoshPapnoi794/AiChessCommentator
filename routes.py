import io
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import chess
import chess.pgn
from flask import abort, render_template, request

from runtime import (
    GAME_CACHE,
    OPENING_BOOK,
    STOCKFISH_PATH,
    USER_GAMES_CACHE,
    app,
    lichess_get,
    parse_initial_time_seconds,
)


@app.after_request
def add_static_cache_headers(response):
    if request.path.startswith('/static/'):
        response.headers.setdefault('Cache-Control', 'public, max-age=604800')
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    games, username, error = [], '', None
    page = 1
    newest_timestamp, oldest_timestamp, games_count = 0, 0, 0

    username = (request.form.get('username') or request.args.get('username') or '').strip()
    page = request.args.get('page', 1, type=int)

    if username:
        if not bool(re.match(r'^[A-Za-z0-9_.-]{1,50}$', username)):
            error = 'Invalid username format.'
        else:
            since = request.args.get('since', type=int)
            until = request.args.get('until', type=int)
            fetched_data = get_games_for_user(username, since=since, until=until)

            if fetched_data is None:
                error = f"Could not fetch games for '{username}'."
            elif not fetched_data['games']:
                error = "No games found."
            else:
                games = fetched_data['games']
                newest_timestamp = fetched_data['newest_timestamp']
                oldest_timestamp = fetched_data['oldest_timestamp']
                games_count = fetched_data['count']

    return render_template(
        'index.html',
        games=games,
        username=username,
        error=error,
        page=page,
        newest_timestamp=newest_timestamp,
        oldest_timestamp=oldest_timestamp,
        games_count=games_count,
    )


def get_games_for_user(username: str, max_games: int = 10, since: Optional[int] = None, until: Optional[int] = None):
    cache_key = (username.lower(), max_games, since or 0, until or 0)
    cached_payload = USER_GAMES_CACHE.get(cache_key)
    if cached_payload:
        for gid, pgn in cached_payload.get('pgn_by_id', {}).items():
            GAME_CACHE.set(gid, pgn)
        return {
            'games': cached_payload['games'],
            'count': cached_payload['count'],
            'newest_timestamp': cached_payload['newest_timestamp'],
            'oldest_timestamp': cached_payload['oldest_timestamp'],
        }

    params = {'max': max_games, 'opening': 'true', 'clocks': 'true'}
    if since:
        params['since'] = since
    if until:
        params['until'] = until

    response = lichess_get(f"games/user/{username}", params=params)
    if response is None:
        return None

    if not response.text.strip():
        return {'games': [], 'count': 0, 'newest_timestamp': 0, 'oldest_timestamp': 0}

    games_data = []
    pgn_by_id: Dict[str, str] = {}
    pgn_io = io.StringIO(response.text)
    newest, oldest = 0, float('inf')

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        headers = game.headers
        game_id = headers.get('Site', '').split('/')[-1]
        if not game_id:
            continue

        utc_date = headers.get('UTCDate', '1970.01.01')
        utc_time = headers.get('UTCTime', '00:00:00')
        try:
            dt_obj = datetime.strptime(f"{utc_date} {utc_time}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)
            timestamp_ms = int(dt_obj.timestamp() * 1000)
            newest = max(newest, timestamp_ms)
            oldest = min(oldest, timestamp_ms)
        except ValueError:
            pass

        games_data.append({
            'id': game_id,
            'white_player': headers.get('White', '?'),
            'black_player': headers.get('Black', '?'),
            'result': headers.get('Result', '*'),
            'date': headers.get('UTCDate', 'N/A'),
        })

        pgn_text = str(game)
        pgn_by_id[game_id] = pgn_text
        GAME_CACHE.set(game_id, pgn_text)

    if oldest == float('inf'):
        oldest = 0

    payload = {
        'games': games_data,
        'count': len(games_data),
        'newest_timestamp': newest,
        'oldest_timestamp': oldest,
        'pgn_by_id': pgn_by_id,
    }
    USER_GAMES_CACHE.set(cache_key, payload)
    return {
        'games': games_data,
        'count': len(games_data),
        'newest_timestamp': newest,
        'oldest_timestamp': oldest,
    }


@app.route('/game/<game_id>')
def view_game(game_id):
    pgn_text = GAME_CACHE.get(game_id)
    if not pgn_text:
        response = lichess_get(
            f"game/export/{game_id}",
            params={'moves': 'true', 'clocks': 'true', 'opening': 'true'},
        )
        if response is None or not response.text.strip():
            abort(404)
        pgn_text = response.text
        GAME_CACHE.set(game_id, pgn_text)

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        abort(404)

    board = game.board()
    moves_data = []

    opening_names = ["Starting Position"]
    last_known_opening = "Starting Position"

    pgn_str = ""
    for node in game.mainline():
        move_san = board.san(node.move)
        if board.turn == chess.WHITE:
            pgn_str += f"{board.fullmove_number}. "
        pgn_str += f"{move_san} "

        match = OPENING_BOOK.get(pgn_str.strip())
        if match:
            last_known_opening = match

        opening_names.append(last_known_opening)
        moves_data.append({'san': move_san, 'clock': node.clock(), 'ply': node.ply()})
        board.push(node.move)

    return render_template(
        'game.html',
        game={
            'white': {'name': game.headers.get('White'), 'rating': game.headers.get('WhiteElo')},
            'black': {'name': game.headers.get('Black'), 'rating': game.headers.get('BlackElo')},
            'start_fen': game.headers.get('FEN'),
            'moves_data': moves_data,
            'initial_time_seconds': parse_initial_time_seconds(game.headers.get('TimeControl')),
            'stockfish_enabled': STOCKFISH_PATH is not None,
            'opening_names_by_ply': opening_names,
        },
    )
