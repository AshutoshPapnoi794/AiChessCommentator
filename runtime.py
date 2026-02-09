import atexit
import logging
import os
import stat
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask
from flask_socketio import SocketIO

from tactics_analyzer import TacticsAnalyzer

import google.generativeai as genai_text_model
from google import genai as genai_tts_client
from google.genai import types

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key')
socketio = SocketIO(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- STOCKFISH SETUP ---
def make_stockfish_executable(path: str) -> bool:
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)
        logger.info("Made %s executable.", path)
        return True
    except Exception as exc:
        logger.error("Failed to make %s executable: %s", path, exc)
        return False


STOCKFISH_PATH = os.getenv('STOCKFISH_PATH')
if not STOCKFISH_PATH:
    candidate_names = ['stockfish-ubuntu-x86-64-avx2', 'stockfish', 'stockfish-linux-x86-64-avx2']
    for name in candidate_names:
        candidate_path = os.path.join('stockfish', name)
        if os.path.exists(candidate_path):
            STOCKFISH_PATH = candidate_path
            break

if STOCKFISH_PATH:
    if not os.access(STOCKFISH_PATH, os.X_OK):
        logger.warning("Stockfish found at %s but not executable. Attempting to fix...", STOCKFISH_PATH)
        make_stockfish_executable(STOCKFISH_PATH)

    if not os.access(STOCKFISH_PATH, os.X_OK):
        logger.error("Stockfish is still not executable. Analysis disabled.")
        STOCKFISH_PATH = None
    else:
        logger.info("Stockfish initialized at: %s", STOCKFISH_PATH)


# --- TACTICS ANALYZER ---
tactics_analyzer = None
if STOCKFISH_PATH:
    try:
        tactics_analyzer = TacticsAnalyzer(engine_path=STOCKFISH_PATH)
        atexit.register(tactics_analyzer.close)
    except Exception as exc:
        logger.error("Tactics Analyzer init failed: %s", exc)
        tactics_analyzer = None


# --- GEMINI ---
GEMINI_TEXT_API_KEY = os.getenv('GEMINI_TEXT_API_KEY')
GEMINI_TTS_API_KEY = os.getenv('GEMINI_TTS_API_KEY')
text_model = None
tts_client = None

if GEMINI_TEXT_API_KEY:
    try:
        genai_text_model.configure(api_key=GEMINI_TEXT_API_KEY)
        text_model = genai_text_model.GenerativeModel('gemini-2.5-flash')
    except Exception as exc:
        logger.error("Text Model Error: %s", exc)

if GEMINI_TTS_API_KEY:
    try:
        tts_client = genai_tts_client.Client(api_key=GEMINI_TTS_API_KEY)
    except Exception as exc:
        logger.error("TTS Client Error: %s", exc)


# --- NETWORK / CACHE ---
LICHESS_API_URL = "https://lichess.org/api"
API_HEADERS = {'User-Agent': 'LichessGameViewer/1.0'}
if os.getenv('LICHESS_TOKEN'):
    API_HEADERS['Authorization'] = f"Bearer {os.getenv('LICHESS_TOKEN')}"

HTTP_TIMEOUT = (5, 20)
HTTP_SESSION = requests.Session()
ENGINE_THREADS = max(1, min(2, os.cpu_count() or 2))
DEFAULT_ANALYSIS_DEPTH = 16


class TimedLRUCache:
    """Simple thread-safe TTL + LRU cache for API/game payloads."""

    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[Any, Tuple[float, Any]]" = OrderedDict()
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


def lichess_get(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
    url = f"{LICHESS_API_URL}/{path.lstrip('/')}"
    headers = API_HEADERS.copy()
    headers['Accept'] = 'application/x-chess-pgn'
    try:
        response = HTTP_SESSION.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        logger.warning("Lichess request failed for %s: %s", url, exc)
        return None


def parse_initial_time_seconds(time_control: Optional[str]) -> int:
    if not time_control or time_control in {"-", "?"}:
        return 600
    base = time_control.split("+", 1)[0]
    if "/" in base:
        base = base.split("/", 1)[-1]
    try:
        parsed = int(base)
        return parsed if parsed > 0 else 600
    except ValueError:
        return 600


GAME_CACHE = TimedLRUCache(max_size=500, ttl_seconds=30 * 60)
USER_GAMES_CACHE = TimedLRUCache(max_size=128, ttl_seconds=3 * 60)
OPENING_BOOK: Dict[str, str] = {}


def load_openings() -> None:
    filenames = ['a.tsv', 'b.tsv', 'c.tsv', 'd.tsv', 'e.tsv']
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as handle:
                for idx, line in enumerate(handle):
                    if idx == 0:
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        OPENING_BOOK[parts[2].strip()] = parts[1].strip()
        except FileNotFoundError:
            logger.warning("Opening file not found: %s", filename)


load_openings()


def close_http_session() -> None:
    HTTP_SESSION.close()
