# --- START OF FILE app.py ---

import os
import re
import io
import logging
import threading
import math
import queue
from typing import List, Optional, Dict
import base64
import mimetypes
import struct
from datetime import datetime
import atexit

import requests
import chess
import chess.pgn
from flask import Flask, render_template, request, abort
from flask_socketio import SocketIO
from stockfish import Stockfish
from dotenv import load_dotenv

from tactics_analyzer import TacticsAnalyzer

# --- UNIFIED GEMINI IMPORTS ---
import google.generativeai as genai
from google.generativeai import types

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key')
socketio = SocketIO(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- ENVIRONMENT AND PATH SETUP ---
def find_stockfish_executable():
    """Finds a valid Stockfish executable path."""
    stockfish_path = os.getenv('STOCKFISH_PATH')
    if stockfish_path and os.path.exists(stockfish_path) and os.access(stockfish_path, os.X_OK):
        logger.info("Using Stockfish executable from STOCKFISH_PATH: %s", stockfish_path)
        return stockfish_path

    candidate_names = ['stockfish-ubuntu-x86-64-avx2', 'stockfish', 'stockfish-linux-x86-64-avx2']
    for name in candidate_names:
        candidate_path = os.path.join('stockfish', name)
        if os.path.exists(candidate_path):
            if os.access(candidate_path, os.X_OK):
                logger.info("Found valid Stockfish executable at: %s", candidate_path)
                return candidate_path
            else:
                logger.error(
                    "Stockfish executable found at '%s' but it is NOT EXECUTABLE. "
                    "Please run 'chmod +x %s' in your terminal.",
                    candidate_path, candidate_path
                )
    return None

STOCKFISH_PATH = find_stockfish_executable()
if not STOCKFISH_PATH:
    logging.warning("Stockfish executable not found or not executable. Analysis will be disabled.")

# --- TACTICS ANALYZER SETUP ---
tactics_analyzer = None
if STOCKFISH_PATH:
    try:
        tactics_analyzer = TacticsAnalyzer(engine_path=STOCKFISH_PATH)
        atexit.register(tactics_analyzer.close)
        logging.info("TacticsAnalyzer initialized successfully.")
    except Exception as e:
        logging.error("Failed to initialize TacticsAnalyzer: %s", e, exc_info=True)
        tactics_analyzer = None


# --- UNIFIED GEMINI SETUP ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
text_model = None
tts_client = None

if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not found. All AI features will be disabled.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        # Setup for Text Generation Model
        text_model_name = os.getenv('GEMINI_TEXT_MODEL', 'gemini-1.5-flash')
        text_model = genai.GenerativeModel(text_model_name)
        logging.info("Gemini text model ('%s') configured successfully.", text_model_name)

        # Setup for Text-to-Speech (TTS) Client and Model
        # The standard client automatically handles authentication for all services.
        # We specify the model during the generation call.
        tts_model_name = os.getenv('GEMINI_TTS_MODEL', 'models/text-to-speech') # Example model
        logging.info("Gemini services (including TTS model '%s') configured.", tts_model_name)
        # Note: No separate tts_client is needed with the unified API.

    except Exception as e:
        logging.error("Failed to configure Gemini services: %s", e, exc_info=True)
        text_model = None

# --- CONSTANTS AND GLOBALS ---
LICHESS_API_URL = "https://lichess.org/api"
API_HEADERS = {'User-Agent': 'LichessGameViewer/1.0 (https://example.com)'}
LICHESS_TOKEN = os.getenv('LICHESS_TOKEN')
if LICHESS_TOKEN:
    API_HEADERS['Authorization'] = f"Bearer {LICHESS_TOKEN}"
USERNAME_RE = re.compile(r'^[A-Za-z0-9_.-]{1,50}$')
GAMEID_RE = re.compile(r'^[A-Za-z0-9_-]+$')
REQUEST_TIMEOUT = (5, 20)
GAME_CACHE: Dict[str, str] = {}
analysis_queues: Dict[str, queue.Queue] = {}
worker_threads: Dict[str, threading.Thread] = {}
OPENING_BOOK: Dict[str, str] = {}

# --- OPENING BOOK LOADER ---
def load_openings():
    """Loads opening data from TSV files into the global OPENING_BOOK."""
    global OPENING_BOOK
    count = 0
    filenames = ['a.tsv', 'b.tsv', 'c.tsv', 'd.tsv', 'e.tsv']
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == 0 and line.startswith('eco\t'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        _eco, name, pgn = parts
                        OPENING_BOOK[pgn.strip()] = name.strip()
                        count += 1
        except FileNotFoundError:
            logger.warning("Opening data file not found: %s. Skipping.", filename)
        except Exception as e:
            logger.error("Error reading opening data from %s: %s", filename, e)
    if count > 0:
        logger.info("Loaded %d opening positions from TSV files.", count)
    else:
        logger.warning("Could not load any opening data. Make sure a.tsv...e.tsv are present.")

# load_openings()

# --- HELPER FUNCTIONS and ROUTE HANDLERS ---
def validate_username(username: str) -> bool: return bool(USERNAME_RE.match(username))
def validate_game_id(game_id: str) -> bool: return bool(GAMEID_RE.match(game_id))

def get_games_for_user(username: str, max_games: int = 10, since: int = None, until: int = None) -> Optional[Dict]:
    if not validate_username(username): return None
    url = f"{LICHESS_API_URL}/games/user/{username}"
    headers = API_HEADERS.copy()
    headers['Accept'] = 'application/x-chess-pgn'
    params = {'max': max_games, 'opening': 'true', 'clocks': 'true'}
    if since: params['since'] = since
    if until: params['until'] = until

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        multi_pgn_text = resp.text
        if not multi_pgn_text: return {'games': [], 'count': 0}

        games_data: List[Dict] = []
        pgn_io = io.StringIO(multi_pgn_text)
        newest_timestamp, oldest_timestamp = 0, float('inf')

        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None: break
            headers_map = game.headers
            
            utc_date = headers_map.get('UTCDate', '1970.01.01')
            utc_time = headers_map.get('UTCTime', '00:00:00')
            try:
                dt_obj = datetime.strptime(f"{utc_date} {utc_time}", "%Y.%m.%d %H:%M:%S")
                timestamp_ms = int(dt_obj.timestamp() * 1000)
                newest_timestamp = max(newest_timestamp, timestamp_ms)
                oldest_timestamp = min(oldest_timestamp, timestamp_ms)
            except ValueError:
                timestamp_ms = None 

            site_url = headers_map.get('Site', '')
            gid = site_url.split('/')[-1] if site_url.startswith('https://lichess.org/') else None
            if not gid: continue
            
            exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
            single_pgn_text = game.accept(exporter)
            GAME_CACHE[gid] = single_pgn_text
            
            result = headers_map.get('Result', '')
            result_text = {'1-0': '1-0 (White win)', '0-1': '0-1 (Black win)', '1/2-1/2': '1/2-1/2 (Draw)'}.get(result, headers_map.get('Termination', 'Unknown'))
            
            games_data.append({
                'id': gid,
                'white_player': f"{headers_map.get('White', '?')} ({headers_map.get('WhiteElo', '?')})",
                'black_player': f"{headers_map.get('Black', '?')} ({headers_map.get('BlackElo', '?')})",
                'result': result_text,
                'date': headers_map.get('UTCDate', 'N/A')
            })

        return {
            'games': games_data, 'count': len(games_data),
            'newest_timestamp': newest_timestamp, 'oldest_timestamp': oldest_timestamp
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching games for user %s: %s", username, e)
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    games, username, error = [], '', None
    page = 1
    newest_timestamp, oldest_timestamp, games_count = 0, 0, 0

    username = (request.form.get('username') or request.args.get('username') or '').strip()
    page = request.args.get('page', 1, type=int)

    if username:
        if not validate_username(username): 
            error = 'Invalid username format.'
        else:
            GAME_CACHE.clear()
            since = request.args.get('since', type=int)
            until = request.args.get('until', type=int)
            fetched_data = get_games_for_user(username, since=since, until=until)
            
            if fetched_data is None: 
                error = f"Could not fetch games for '{username}'. The user may not exist or the API is unavailable."
            elif not fetched_data['games']: 
                error = f"No more games found for '{username}'." if page > 1 else f"No recent games found for '{username}'."
            else: 
                games = fetched_data['games']
                newest_timestamp = fetched_data['newest_timestamp']
                oldest_timestamp = fetched_data['oldest_timestamp']
                games_count = fetched_data['count']

    return render_template('index.html', games=games, username=username, error=error, page=page,
                           newest_timestamp=newest_timestamp, oldest_timestamp=oldest_timestamp, games_count=games_count)

@app.route('/game/<game_id>')
def view_game(game_id: str):
    game_id = (game_id or '').strip()
    if not validate_game_id(game_id): abort(400)
    pgn_text = GAME_CACHE.get(game_id)
    if not pgn_text:
        try:
            resp = requests.get(f"{LICHESS_API_URL}/game/export/{game_id}", headers=API_HEADERS, 
                                params={'moves': 'true', 'clocks': 'true', 'opening': 'true'}, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            pgn_text = resp.text
            if pgn_text: GAME_CACHE[game_id] = pgn_text
        except requests.exceptions.RequestException: abort(503)
    if not pgn_text: abort(404)
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game: abort(500)
    headers_map = game.headers
    
    moves_with_clocks, opening_names_by_ply = [], ["Starting Position"]
    last_found_opening, pgn_string = "Starting Position", ""
    board = game.board()
    for node in game.mainline():
        move = node.move
        if board.turn == chess.WHITE: pgn_string += f"{board.fullmove_number}. "
        pgn_string += board.san(move) + " "
        last_found_opening = OPENING_BOOK.get(pgn_string.strip(), last_found_opening)
        opening_names_by_ply.append(last_found_opening)
        moves_with_clocks.append({ 'san': board.san(move), 'ply': node.ply(), 'clock': node.clock() })
        board.push(move)

    game_data = {
        'white': {'name': headers_map.get('White', 'N/A'), 'rating': headers_map.get('WhiteElo', '?')},
        'black': {'name': headers_map.get('Black', 'N/A'), 'rating': headers_map.get('BlackElo', '?')},
        'start_fen': headers_map.get('FEN'), 'moves_data': moves_with_clocks,
        'initial_time_seconds': int(headers_map.get('TimeControl', '600+0').split('+')[0]),
        'stockfish_enabled': STOCKFISH_PATH is not None, 'opening_names_by_ply': opening_names_by_ply,
    }
    return render_template('game.html', game=game_data)

# --- WORKER THREAD & SOCKETIO HANDLERS ---
def analysis_worker(sid: str, q: queue.Queue):
    logger.info("Starting analysis worker for sid: %s", sid)
    stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": 4, "Hash": 128})
    while True:
        try:
            job = q.get()
            if job is None: break
            while not q.empty():
                try: job = q.get_nowait()
                except queue.Empty: break
            if job is None: break
            fen, settings, request_id = job['fen'], job['settings'], job['requestId']
            stockfish.update_engine_parameters({"Threads": settings.get('threads', 4)})
            
            for depth in range(8, settings.get('depth', 18) + 1, 2):
                stockfish.set_depth(depth)
                stockfish.set_fen_position(fen)
                stockfish.get_best_move()
                if not q.empty(): break
                evaluation = stockfish.get_evaluation()
                board = chess.Board(fen)
                top_moves = [{'san': board.san(chess.Move.from_uci(m['Move'])), 'cp': m.get('Centipawn'), 'mate': m.get('Mate'), 'uci': m['Move']}
                             for m in stockfish.get_top_moves(3) if m and 'Move' in m]
                socketio.emit('analysis_result', {'eval': evaluation, 'moves': top_moves, 'requestId': request_id, 'depth': depth}, room=sid)
        except Exception as e:
            logger.error("Error in analysis worker for sid %s: %s", sid, e, exc_info=True)
            socketio.emit('analysis_error', {'message': 'Worker thread encountered an error.'}, room=sid)
    del stockfish
    logger.info("Stopped analysis worker for sid: %s", sid)

@socketio.on('connect')
def handle_connect():
    if STOCKFISH_PATH:
        sid = request.sid
        logger.info('Client connected: %s', sid)
        q = queue.Queue()
        analysis_queues[sid] = q
        thread = threading.Thread(target=analysis_worker, args=(sid, q))
        worker_threads[sid] = thread
        thread.daemon = True
        thread.start()

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    logger.info('Client disconnected: %s', sid)
    if sid in analysis_queues:
        analysis_queues[sid].put(None)
        del analysis_queues[sid]
    if sid in worker_threads:
        del worker_threads[sid]

@socketio.on('analyze_position')
def handle_analysis_request(data):
    if request.sid in analysis_queues:
        analysis_queues[request.sid].put(data)

# --- ACCURACY AND MOVE QUALITY ---
def get_cp_value(evaluation: dict) -> int:
    if evaluation['type'] == 'cp': return evaluation['value']
    if evaluation['type'] == 'mate':
        return 30000 - evaluation['value'] if evaluation['value'] > 0 else -30000 + abs(evaluation['value'])
    return 0

def classify_move(clamped_cp_loss: int, cp_before: int, cp_after: int, is_white_move: bool) -> str:
    is_winning_before = cp_before > 200 if is_white_move else cp_before < -200
    is_losing_after = (is_white_move and cp_after < -200) or (not is_white_move and cp_after > 200)
    is_drawn_after = -200 <= cp_after <= 200
    if is_winning_before and (is_losing_after or is_drawn_after): return 'blunder'

    if clamped_cp_loss <= 15: return 'best'
    if clamped_cp_loss <= 40: return 'excellent'
    if clamped_cp_loss <= 90: return 'good'
    if clamped_cp_loss <= 200: return 'inaccuracy'
    if clamped_cp_loss <= 450: return 'mistake'
    return 'blunder'

@socketio.on('get_move_quality')
def handle_move_quality_request(data):
    fen_before, fen_after, sid = data.get('fen_before'), data.get('fen_after'), request.sid
    if not STOCKFISH_PATH or not fen_before or not fen_after: return
    
    stockfish = None
    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, depth=12, parameters={"Threads": 1, "Hash": 64})
        stockfish.set_fen_position(fen_before)
        cp_before = get_cp_value(stockfish.get_evaluation())
        stockfish.set_fen_position(fen_after)
        cp_after = get_cp_value(stockfish.get_evaluation())
        is_white_move = chess.Board(fen_before).turn == chess.WHITE
        cp_loss = (cp_before - cp_after) if is_white_move else (cp_after - cp_before)
        classification = classify_move(max(0, cp_loss), cp_before, cp_after, is_white_move)
        socketio.emit('move_quality_result', {'classification': classification}, room=sid)
    except Exception as e:
        logger.error("Error during move quality check for sid %s: %s", sid, e)
    finally:
        if stockfish:
            del stockfish

def full_game_analysis_threaded(sid: str, fens: List[str]):
    stockfish = None
    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, depth=14, parameters={"Threads": 1, "Hash": 64})
        white_accuracies, black_accuracies = [], []
        for i in range(len(fens) - 1):
            stockfish.set_fen_position(fens[i])
            cp_before = get_cp_value(stockfish.get_evaluation())
            stockfish.set_fen_position(fens[i+1])
            cp_after = get_cp_value(stockfish.get_evaluation())
            is_white_move = chess.Board(fens[i]).turn == chess.WHITE
            cp_loss = max(0, (cp_before - cp_after) if is_white_move else (cp_after - cp_before))
            move_accuracy = max(0, min(100, 103 * math.exp(-0.004 * cp_loss) - 3))
            (white_accuracies if is_white_move else black_accuracies).append(move_accuracy)
        
        acc_w = round(sum(white_accuracies) / len(white_accuracies)) if white_accuracies else 100
        acc_b = round(sum(black_accuracies) / len(black_accuracies)) if black_accuracies else 100
        socketio.emit('full_analysis_complete', {'white': acc_w, 'black': acc_b}, room=sid)
    except Exception as e:
        logger.error("Full game analysis failed for sid %s: %s", sid, e, exc_info=True)
        socketio.emit('analysis_error', {'message': 'Could not calculate accuracy.'}, room=sid)
    finally:
        if stockfish:
            del stockfish


@socketio.on('request_full_analysis')
def handle_full_analysis_request(data):
    fens = data.get('fens')
    if fens and STOCKFISH_PATH:
        threading.Thread(target=full_game_analysis_threaded, args=(request.sid, fens), daemon=True).start()

# --- AI COMMENTARY HELPERS AND HANDLER ---
def format_evaluation(evaluation: dict) -> str:
    if evaluation.get('type') == 'cp': return f"{'+' if evaluation['value'] > 0 else ''}{evaluation['value']/100.0:.2f}"
    if evaluation.get('type') == 'mate': return f"Mate in {abs(evaluation['value'])}"
    return "N/A"

def format_tactical_analysis(analysis: Dict) -> str:
    """Formats the analysis dictionary into a markdown string for the LLM prompt."""
    if not analysis: return ""
    summary_points = []
    
    # Static Analysis
    static_analysis = analysis.get('static_analysis', {})
    if static_analysis.get('piece_interactions'):
        interactions_str = "\n".join(f"*   {rel}" for rel in static_analysis['piece_interactions'])
        summary_points.append(f"**Key Piece Interactions (SEE Analysis):**\n{interactions_str}")
    
    move_analysis = analysis.get('analysis_of_move', {})
    if move_analysis.get('discovered_attacks') or static_analysis.get('pins') or static_analysis.get('forks') or static_analysis.get('skewers') or static_analysis.get('back_rank_weakness'):
        tactical_overview = []
        if move_analysis.get('discovered_attacks'):
            for da in move_analysis['discovered_attacks']:
                tactical_overview.append(f"Discovered Attack: Moving the {da['moving_piece']} reveals an attack from the {da['revealed_attacker']} on {da['revealed_attacker_square']} to the {da['target']} on {da['target_square']}.")
        if move_analysis.get('clearance_sacrifices'):
            for cs in move_analysis['clearance_sacrifices']:
                tactical_overview.append(f"Clearance Sacrifice: The move is a sacrifice to clear a line/square. Type: {cs['type']}.")
        if static_analysis.get('pins'):
            for pin in static_analysis['pins']:
                tactical_overview.append(f"Pin: A {pin['pinner_piece']} on {pin['pinner_square']} creates an **{pin['type']} Pin** on the opponent's {pin['pinned_piece']} at {pin['pinned_square']}.")
        if static_analysis.get('forks'):
            for fork in static_analysis['forks']:
                targets = ", ".join([f"{t['piece']} on {t['square']}" for t in fork['targets']])
                tactical_overview.append(f"Fork Threat: A {fork['forking_piece']} on {fork['forking_piece_square']} is forking: {targets}.")
        if static_analysis.get('skewers'):
            for skewer in static_analysis['skewers']:
                summary_points.append(f"Skewer Threat: {skewer['attacker']} is skewering {skewer['skewed']} and {skewer['behind']}.")
        if static_analysis.get('back_rank_weakness'):
            for brw in static_analysis['back_rank_weakness']:
                if brw['severity'] in ['Critical', 'Severe']:
                    tactical_overview.append(f"Back-Rank Weakness: {brw['color']}'s king is vulnerable on the back rank ({brw['severity']} severity).")
        
        if tactical_overview:
            overview_str = "\n".join(f"*   {point}" for point in tactical_overview)
            summary_points.append(f"**Other Tactical & Positional Notes:**\n{overview_str}")

    return "\n".join(summary_points)


def get_engine_recommendation(fen: str) -> str:
    """Gets the best move from Stockfish for a given FEN."""
    if not STOCKFISH_PATH:
        return "N/A"
    stockfish_temp = None
    try:
        stockfish_temp = Stockfish(path=STOCKFISH_PATH, depth=14, parameters={"Threads": 1, "Hash": 64})
        stockfish_temp.set_fen_position(fen)
        best_move_uci = stockfish_temp.get_best_move()
        if best_move_uci:
            return chess.Board(fen).san(chess.Move.from_uci(best_move_uci))
    except Exception as e:
        logger.error("Error getting engine recommendation: %s", e)
    finally:
        if stockfish_temp:
            del stockfish_temp
    return "N/A"

def prepare_commentary_prompt(data: Dict) -> str:
    """Prepares the prompt for the AI commentator."""
    previous_fen = data.get('previous_fen')
    engine_best_move_before = get_engine_recommendation(previous_fen) if previous_fen else "N/A"

    top_lines_str = "\n".join([
        f"- {line['san']} (Eval: {format_evaluation({'type': 'cp', 'value': line.get('cp', 0)}) if line.get('mate') is None else format_evaluation({'type': 'mate', 'value': line.get('mate')})})"
        for line in data.get('topLines', [])[:2]
    ])

    tactical_summary_str = ""
    if tactics_analyzer and data.get('current_fen'):
        try:
            tactical_analysis = tactics_analyzer.analyze(data['current_fen'], previous_fen)
            tactical_summary_str = format_tactical_analysis(tactical_analysis)
        except Exception as e:
            logger.error("Error during tactical analysis: %s", e)

    return f"""
You are an expert chess commentator. Analyze the move just played.

**Guiding Principles:**
1.  **Game Phase:** Use the move number to determine the game phase (Opening: 1-12, Middlegame: 13-40, Endgame: 41+).
2.  **Proportionality:** Match commentary intensity to the move's impact. A minor inaccuracy is not a catastrophe.
3.  **Narrative Framing:** Focus on the *purpose* behind moves. What was the player trying to achieve?

{tactical_summary_str}

**GAME CONTEXT**
- PGN so far: {data.get('pgn', '')}
- Move Number: {data.get('ply', 0) // 2 + 1}

**ANALYSIS OF THE MOVE**
- Position BEFORE: {previous_fen}
- Player to move was: {"White" if chess.Board(previous_fen).turn == chess.WHITE else "Black"}
- Engine recommendation was: **{engine_best_move_before}**

- The move ACTUALLY played: **{data.get('humanMove', 'N/A')}**

- Position AFTER: {data.get('current_fen')}
- New evaluation is: **{format_evaluation(data.get('evaluation', {}))}**
- Engine's top lines are now:
{top_lines_str}

**Your Task:**
Generate a brief, expert commentary on the move **{data.get('humanMove', 'N/A')}**. Explain its purpose and consequences, comparing it to the engine's recommendation. Use Markdown for emphasis.
"""

def generate_text_commentary(sid: str, prompt: str):
    """Generates text commentary using the Gemini model."""
    try:
        logger.info("="*20 + " PROMPT SENT TO GEMINI " + "="*20)
        logger.info(prompt)
        logger.info("="*50)

        text_response = text_model.generate_content(prompt)
        text_commentary = text_response.text
        socketio.emit('ai_commentary_text_result', {'commentary': text_commentary}, room=sid)
        return text_commentary
    except Exception as e:
        logger.error("Error generating text commentary for sid %s: %s", sid, e, exc_info=True)
        socketio.emit('ai_commentary_error', {'message': 'Failed to generate text commentary.'}, room=sid)
        return None

def generate_audio_commentary(sid: str, text: str):
    """Generates audio commentary using the Gemini TTS model."""
    try:
        tts_model_name = os.getenv('GEMINI_TTS_MODEL', 'models/text-to-speech')

        # The API expects a list of content parts
        response = genai.generate_content(
            model=tts_model_name,
            prompt=text,
            response_mime_type="audio/wav" # Request WAV directly
        )

        # The audio data is in response.audio_content
        if response.audio_content:
            socketio.emit('ai_commentary_audio_result', {
                'audio_data': base64.b64encode(response.audio_content).decode('utf-8')
            }, room=sid)
        else:
            logger.warning("TTS generation succeeded but returned no audio content.")

    except Exception as e:
        logger.error("Error generating audio commentary for sid %s: %s", sid, e, exc_info=True)
        socketio.emit('ai_commentary_error', {'message': 'Failed to generate audio commentary.'}, room=sid)


@socketio.on('get_ai_commentary')
def handle_ai_commentary_request(data):
    """Handles the request for AI commentary, orchestrating prompt creation, text, and audio generation."""
    sid = request.sid
    if not text_model:
        return socketio.emit('ai_commentary_text_result', {'commentary': 'AI text commentator is not available.'}, room=sid)

    prompt = prepare_commentary_prompt(data)
    text_commentary = generate_text_commentary(sid, prompt)

    if text_commentary and data.get('audio_enabled'):
        generate_audio_commentary(sid, text_commentary)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '5000')))