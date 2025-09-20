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
from PIL import Image # <-- NEW IMPORT

import requests
import chess
import chess.pgn
from flask import Flask, render_template, request, abort
from flask_socketio import SocketIO
from stockfish import Stockfish
from dotenv import load_dotenv

# --- CORRECTED GEMINI IMPORTS ---
import google.generativeai as genai_text_model  # For the text model
from google import genai as genai_tts_client     # For the TTS Client
from google.genai import types                   # Shared types

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key')
socketio = SocketIO(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- STOCKFISH SETUP ---
STOCKFISH_PATH = os.getenv('STOCKFISH_PATH')
if not STOCKFISH_PATH:
    candidate_names = [
        'stockfish-ubuntu-x86-64-avx2', 'stockfish', 'stockfish-linux-x86-64-avx2'
    ]
    for name in candidate_names:
        candidate_path = os.path.join('stockfish', name)
        if os.path.exists(candidate_path):
            STOCKFISH_PATH = candidate_path
            logging.info("Found potential Stockfish executable at: %s", STOCKFISH_PATH)
            break
if not STOCKFISH_PATH or not os.path.exists(STOCKFISH_PATH):
    logging.warning("Stockfish executable not found. Analysis will be disabled.")
    STOCKFISH_PATH = None
elif not os.access(STOCKFISH_PATH, os.X_OK):
    logging.error(
        "Stockfish executable found at '%s' but it is NOT EXECUTABLE. "
        "Please run 'chmod +x %s' in your terminal. Analysis is disabled.",
        STOCKFISH_PATH, STOCKFISH_PATH
    )
    STOCKFISH_PATH = None

# --- GEMINI SETUP WITH TWO KEYS ---
GEMINI_TEXT_API_KEY = os.getenv('GEMINI_TEXT_API_KEY')
GEMINI_TTS_API_KEY = os.getenv('GEMINI_TTS_API_KEY')

text_model = None
tts_client = None

if not GEMINI_TEXT_API_KEY:
    logging.warning("GEMINI_TEXT_API_KEY not found. Text AI features will be disabled.")
else:
    try:
        genai_text_model.configure(api_key=GEMINI_TEXT_API_KEY)
        text_model = genai_text_model.GenerativeModel('gemini-1.5-flash')
        logging.info("Gemini text model configured successfully.")
    except Exception as e:
        logging.error("Failed to configure Gemini text model: %s", e, exc_info=True)
        text_model = None

if not GEMINI_TTS_API_KEY:
    logging.warning("GEMINI_TTS_API_KEY not found. Audio AI features will be disabled.")
else:
    try:
        tts_client = genai_tts_client.Client(api_key=GEMINI_TTS_API_KEY)
        logging.info("Gemini TTS client configured successfully.")
    except Exception as e:
        logging.error("Failed to configure Gemini TTS client: %s", e, exc_info=True)
        tts_client = None

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
OPENING_BOOK: Dict[str, str] = {} # <-- NEW: For opening lookup

# --- NEW: OPENING BOOK LOADER ---
def load_openings():
    """Loads opening data from TSV files into the global OPENING_BOOK."""
    global OPENING_BOOK
    count = 0
    filenames = ['a.tsv', 'b.tsv', 'c.tsv', 'd.tsv', 'e.tsv']
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == 0 and line.startswith('eco\t'):  # Skip header
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

load_openings() # <-- NEW: Load data on startup

# --- TTS HELPER FUNCTIONS ---
def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b"data", data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    bits_per_sample, rate = 16, 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try: rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError): pass
        elif param.startswith("audio/L"):
            try: bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError): pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

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
            
            # Extract timestamp for pagination
            utc_date = headers_map.get('UTCDate', '1970.01.01')
            utc_time = headers_map.get('UTCTime', '00:00:00')
            try:
                dt_obj = datetime.strptime(f"{utc_date} {utc_time}", "%Y.%m.%d %H:%M:%S")
                timestamp_ms = int(dt_obj.timestamp() * 1000)
                newest_timestamp = max(newest_timestamp, timestamp_ms)
                oldest_timestamp = min(oldest_timestamp, timestamp_ms)
            except ValueError:
                timestamp_ms = None # Skip if format is unexpected

            site_url = headers_map.get('Site', '')
            gid = site_url.split('/')[-1] if site_url.startswith('https://lichess.org/') else None
            if not gid: continue
            
            exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
            single_pgn_text = game.accept(exporter)
            GAME_CACHE[gid] = single_pgn_text
            
            result = headers_map.get('Result', '')
            if result == '1-0': result_text = '1-0 (White win)'
            elif result == '0-1': result_text = '0-1 (Black win)'
            elif result == '1/2-1/2': result_text = '1/2-1/2 (Draw)'
            else: result_text = headers_map.get('Termination', 'Unknown')
            
            games_data.append({
                'id': gid,
                'white_player': f"{headers_map.get('White', '?')} ({headers_map.get('WhiteElo', '?')})",
                'black_player': f"{headers_map.get('Black', '?')} ({headers_map.get('BlackElo', '?')})",
                'result': result_text,
                'date': headers_map.get('UTCDate', 'N/A')
            })

        return {
            'games': games_data,
            'count': len(games_data),
            'newest_timestamp': newest_timestamp,
            'oldest_timestamp': oldest_timestamp
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching games for user %s: %s", username, e)
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    games, username, error = [], '', None
    page = 1
    newest_timestamp, oldest_timestamp, games_count = 0, 0, 0

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
    else: # GET request
        username = (request.args.get('username') or '').strip()
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

    return render_template('index.html', 
        games=games, 
        username=username, 
        error=error,
        page=page,
        newest_timestamp=newest_timestamp,
        oldest_timestamp=oldest_timestamp,
        games_count=games_count
    )

@app.route('/game/<game_id>')
def view_game(game_id: str):
    game_id = (game_id or '').strip()
    if not validate_game_id(game_id): abort(400)
    pgn_text = GAME_CACHE.get(game_id)
    if not pgn_text:
        url = f"{LICHESS_API_URL}/game/export/{game_id}"
        try:
            params = {'moves': 'true', 'clocks': 'true', 'opening': 'true'}
            resp = requests.get(url, headers=API_HEADERS, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            pgn_text = resp.text
            if pgn_text: GAME_CACHE[game_id] = pgn_text
        except requests.exceptions.RequestException: abort(503)
    if not pgn_text: abort(404)
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game: abort(500)
    headers_map = game.headers
    start_fen = headers_map.get('FEN')
    time_control = headers_map.get('TimeControl', '600+0')
    initial_time_seconds = int(time_control.split('+')[0])
    
    # --- MODIFIED: Process moves, clocks, and opening names simultaneously ---
    moves_with_clocks = []
    opening_names_by_ply = ["Starting Position"]
    last_found_opening = "Starting Position"
    pgn_string = ""
    board = game.board()

    for node in game.mainline():
        move = node.move
        
        # Opening lookup logic
        if board.turn == chess.WHITE:
            pgn_string += f"{board.fullmove_number}. "
        pgn_string += board.san(move) + " "
        lookup_key = pgn_string.strip()
        if lookup_key in OPENING_BOOK:
            last_found_opening = OPENING_BOOK[lookup_key]
        opening_names_by_ply.append(last_found_opening)
        
        # Existing move data extraction
        moves_with_clocks.append({ 'san': board.san(move), 'ply': node.ply(), 'clock': node.clock() })
        board.push(move)

    game_data = {
        'white': {'name': headers_map.get('White', 'N/A'), 'rating': headers_map.get('WhiteElo', '?')},
        'black': {'name': headers_map.get('Black', 'N/A'), 'rating': headers_map.get('BlackElo', '?')},
        'start_fen': start_fen,
        'moves_data': moves_with_clocks,
        'initial_time_seconds': initial_time_seconds,
        'stockfish_enabled': STOCKFISH_PATH is not None,
        'opening_names_by_ply': opening_names_by_ply, # <-- NEW: Pass dynamic opening names
    }
    return render_template('game.html', game=game_data)

# --- WORKER THREAD FUNCTION ---
def analysis_worker(sid: str, q: queue.Queue):
    logger.info("Starting analysis worker for sid: %s", sid)
    stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": 4, "Hash": 128})

    while True:
        try:
            job = q.get()
            if job is None: break
            while not q.empty():
                try:
                    job = q.get_nowait()
                    if job is None: break
                except queue.Empty: break
            if job is None: break
            fen, settings, request_id = job['fen'], job['settings'], job['requestId']
            max_depth = settings.get('depth', 18)
            threads = settings.get('threads', 4)
            stockfish.update_engine_parameters({"Threads": threads})
            
            for current_depth in range(8, max_depth + 1, 2):
                stockfish.set_depth(current_depth)
                stockfish.set_fen_position(fen)
                stockfish.get_best_move()
                if not q.empty(): break
                evaluation = stockfish.get_evaluation()
                top_moves_uci = stockfish.get_top_moves(3)
                board = chess.Board(fen)
                top_moves_san = []
                if top_moves_uci:
                    for move_info in top_moves_uci:
                        try:
                            move = chess.Move.from_uci(move_info['Move'])
                            top_moves_san.append({
                                'san': board.san(move),
                                'cp': move_info.get('Centipawn'),
                                'mate': move_info.get('Mate'),
                                'uci': move_info['Move']
                            })
                        except (KeyError, AttributeError, ValueError): continue
                socketio.emit('analysis_result', {
                    'eval': evaluation, 'moves': top_moves_san, 'requestId': request_id, 'depth': current_depth
                }, room=sid)
        except Exception as e:
            logger.error("Error in analysis worker for sid %s: %s", sid, e, exc_info=True)
            socketio.emit('analysis_error', {'message': 'Worker thread encountered an error.'}, room=sid)
    del stockfish 
    logger.info("Stopped analysis worker for sid: %s", sid)

# --- SOCKETIO HANDLERS ---
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    logger.info('Client connected: %s', sid)
    if not STOCKFISH_PATH: return
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
    sid = request.sid
    if sid in analysis_queues:
        analysis_queues[sid].put(data)

# --- ACCURACY AND MOVE QUALITY ANALYSIS ---
def get_cp_value(evaluation: dict) -> int:
    if evaluation['type'] == 'cp': return evaluation['value']
    if evaluation['type'] == 'mate':
        return 30000 - evaluation['value'] if evaluation['value'] > 0 else -30000 + abs(evaluation['value'])
    return 0

def classify_move(clamped_cp_loss: int, cp_before: int, cp_after: int, is_white_move: bool) -> str:
    is_winning_before = cp_before > 200 if is_white_move else cp_before < -200
    is_not_winning_after = -200 <= cp_after <= 200 or \
                           (is_white_move and cp_after < -200) or \
                           (not is_white_move and cp_after > 200)

    if is_winning_before and is_not_winning_after:
        return 'blunder'

    if clamped_cp_loss <= 15: return 'best'
    if clamped_cp_loss <= 40: return 'excellent'
    if clamped_cp_loss <= 90: return 'good'
    if clamped_cp_loss <= 200: return 'inaccuracy'
    if clamped_cp_loss <= 450: return 'mistake'
    return 'blunder'

@socketio.on('get_move_quality')
def handle_move_quality_request(data):
    sid = request.sid
    fen_before = data.get('fen_before')
    fen_after = data.get('fen_after')

    if not STOCKFISH_PATH or not fen_before or not fen_after:
        return

    stockfish = None
    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, depth=12, parameters={"Threads": 1, "Hash": 64})
        
        stockfish.set_fen_position(fen_before)
        eval_before = stockfish.get_evaluation()
        cp_before = get_cp_value(eval_before)
        
        stockfish.set_fen_position(fen_after)
        eval_after = stockfish.get_evaluation()
        cp_after = get_cp_value(eval_after)
        
        is_white_move = chess.Board(fen_before).turn == chess.WHITE
        cp_loss = (cp_before - cp_after) if is_white_move else (cp_after - cp_before)
        clamped_loss = max(0, cp_loss)
        
        classification = classify_move(clamped_loss, cp_before, cp_after, is_white_move)
        
        socketio.emit('move_quality_result', {'classification': classification}, room=sid)
    except Exception as e:
        logger.error("Error during real-time move quality check for sid %s: %s", sid, e)
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
            eval_before = stockfish.get_evaluation()
            cp_before = get_cp_value(eval_before)
            
            stockfish.set_fen_position(fens[i+1])
            eval_after = stockfish.get_evaluation()
            cp_after = get_cp_value(eval_after)
            
            is_white_move = chess.Board(fens[i]).turn == chess.WHITE
            cp_loss = (cp_before - cp_after) if is_white_move else (cp_after - cp_before)
            clamped_loss = max(0, cp_loss)
            
            move_accuracy = 103 * math.exp(-0.004 * clamped_loss) - 3
            move_accuracy = max(0, min(100, move_accuracy))

            if is_white_move: white_accuracies.append(move_accuracy)
            else: black_accuracies.append(move_accuracy)

        accuracy_white = round(sum(white_accuracies) / len(white_accuracies)) if white_accuracies else 100
        accuracy_black = round(sum(black_accuracies) / len(black_accuracies)) if black_accuracies else 100
        
        socketio.emit('full_analysis_complete', {'white': accuracy_white, 'black': accuracy_black}, room=sid)
    except Exception as e:
        logger.error("Full game analysis failed for sid %s: %s", sid, e, exc_info=True)
        socketio.emit('analysis_error', {'message': 'Could not calculate accuracy.'}, room=sid)
    finally:
        if stockfish:
            del stockfish

@socketio.on('request_full_analysis')
def handle_full_analysis_request(data):
    sid = request.sid
    fens = data.get('fens')
    if not fens or not STOCKFISH_PATH: return
    thread = threading.Thread(target=full_game_analysis_threaded, args=(sid, fens))
    thread.daemon = True
    thread.start()

# --- AI COMMENTARY HANDLER ---
def format_evaluation(evaluation: dict) -> str:
    if evaluation['type'] == 'cp': return f"{'+' if evaluation['value'] > 0 else ''}{evaluation['value']/100.0:.2f}"
    if evaluation['type'] == 'mate': return f"Mate in {abs(evaluation['value'])}"
    return "N/A"

@socketio.on('get_ai_commentary')
def handle_ai_commentary_request(data):
    sid = request.sid
    
    if not text_model:
        socketio.emit('ai_commentary_text_result', {
            'commentary': 'AI text commentator is not available (server-side configuration error).'
        }, room=sid)
        return

    try:
        # --- NEW: Process image and prepare multi-modal request ---
        image_b64 = data.get('board_image_base64')
        board_image = None
        if image_b64:
            try:
                image_bytes = base64.b64decode(image_b64)
                board_image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                logger.warning("Failed to process board image from base64: %s", e)

        # --- Existing data extraction ---
        ply = data.get('ply', 0)
        pgn = data.get('pgn', '')
        human_move = data.get('humanMove', 'N/A')
        engine_best_move = data.get('engineBestMove', 'N/A')
        evaluation = format_evaluation(data.get('evaluation', {}))
        top_lines = data.get('topLines', [])
        audio_enabled = data.get('audio_enabled', False)
        
        top_lines_str = "\n".join([
            f"- {line['san']} (Eval: {format_evaluation({'type': 'cp', 'value': line.get('cp', 0)}) if line.get('mate') is None else format_evaluation({'type': 'mate', 'value': line.get('mate')})})"
            for line in top_lines[:2]
        ])
        
        prompt = f"""
You are an expert chess commentator. Your goal is to provide insightful, narrative-driven commentary for an audience of club-level players.
Analyze the following board position (image provided) and the move just played.
**Guiding Principles:**
1.  **Game Phase Awareness:** Use the move number to determine the game phase.
    *   **Opening (Moves 1-15):** Be brief. Focus on opening principles. A slightly inaccurate move is just "quiet" or "unusual," not a major event.
    *   **Middlegame (Moves 16-40):** Focus on strategic plans, tactical opportunities, and blunders.
    *   **Endgame (Moves 41+):** Focus on technical aspects like king activity and pawn structures.
2.  **Proportionality:** Match your commentary's intensity to the move's impact. A simple developing move gets one sentence. A game-changing blunder deserves more detail.
3.  **Positive and Narrative Framing:** Tell a story. Focus on the *purpose* behind moves.
**Game State:**
- Move Number: {ply // 2 + 1}
- PGN so far: {pgn}
- The move just played was: **{human_move}**
- The engine's evaluation is: **{evaluation}**
- The engine's preferred move was: **{engine_best_move}**
- Other top engine lines:
{top_lines_str}
**Your Task:**
Based on your principles, the text data, and the provided board image, generate a brief, expert commentary on the move **{human_move}**. Use Markdown for emphasis.
"""
        
        # 1. GENERATE AND EMIT TEXT
        request_contents = [prompt]
        if board_image:
            request_contents.append(board_image)
        
        text_response = text_model.generate_content(request_contents)
        text_commentary = text_response.text
        
        socketio.emit('ai_commentary_text_result', {'commentary': text_commentary}, room=sid)

        # 2. GENERATE AND EMIT AUDIO (Only if TTS client is available AND user enabled it)
        if not (audio_enabled and tts_client):
            return

        tts_model_name = "gemini-2.5-flash-preview-tts"
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=text_commentary)])]
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon"))
            ),
        )

        audio_buffer = io.BytesIO()
        first_chunk = True
        mime_type = "audio/L16;rate=24000"

        for chunk in tts_client.models.generate_content_stream(
            model=tts_model_name, contents=contents, config=generate_content_config
        ):
            if not (chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts): continue
            
            part = chunk.candidates[0].content.parts[0]
            if part.inline_data and part.inline_data.data:
                if first_chunk:
                    mime_type = part.inline_data.mime_type
                    first_chunk = False
                audio_buffer.write(part.inline_data.data)

        if audio_buffer.getbuffer().nbytes > 0:
            wav_data = convert_to_wav(audio_buffer.getvalue(), mime_type)
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            socketio.emit('ai_commentary_audio_result', {'audio_data': audio_base64}, room=sid)

    except Exception as e:
        logger.error("Error during AI commentary generation for sid %s: %s", sid, e, exc_info=True)
        socketio.emit('ai_commentary_error', {'message': 'Failed to generate AI commentary.'}, room=sid)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '5000')))