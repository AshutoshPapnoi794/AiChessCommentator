<div align="center">

# ♟️ AI Chess Commentator

**Real-Time Automated Grandmaster Commentary & Tactical Engine**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-Flask%20%7C%20Socket.IO-green.svg)](https://flask.palletsprojects.org/)
[![Engine](https://img.shields.io/badge/engine-Stockfish%20NNUE-orange.svg)](https://stockfishchess.org/)
[![AI Model](https://img.shields.io/badge/AI-Google%20Gemini%202.5%20Flash-purple.svg)](https://deepmind.google/technologies/gemini/)
[![Audio Engine](https://img.shields.io/badge/Audio-KittenTTS-ff69b4.svg)](https://github.com/KittenML)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

*Delivering dynamic, human-like grandmaster commentary, deep tactical analysis, and instant text-to-speech voiceover for every chess move.*

---

</div>

## 🌟 Overview

**AI Chess Commentator** is an interactive, real-time chess platform and analysis engine. By combining **Stockfish NNUE** evaluation, custom **tactical motif detection**, and **Google Gemini 2.5** natural language generation with **KittenTTS** audio synthesis, the system offers an engaging broadcast-style commentary experience similar to grandmaster stream commentary.

Whether you're playing interactively, reviewing custom PGNs, or importing live games from **Lichess**, the engine evaluates position dynamics, flags tactical turns (forks, pins, discovered attacks, sacrifices), tags move quality with Chess.com-style accuracy badges, and speaks natural commentary in real time.

---

## ✨ Key Features

- **🎙️ Real-Time Audio & Text Commentary:** Generates human-sounding commentary on every move powered by Gemini 2.5 Flash and narrated instantly using KittenTTS local neural speech synthesis.
- **⚡ Stockfish NNUE Evaluation & Classification:** Computes move quality in centipawns, identifying *Brilliant*, *Best*, *Good*, *Inaccuracy*, *Mistake*, *Miss*, and *Blunder* moves.
- **🔍 Advanced Tactical Motif Detection:** Detects complex tactical themes including forks, pins, skewers, discovered attacks, double checks, removal of defenders, trapped pieces, and sacrificial continuations.
- **🌐 Interactive Web Interface:** Responsive web GUI built with `chessboard.js` and `chess.js` supporting legal move validation, visual accuracy badge overlays, sound effects, and game clocks.
- **🔄 Lichess Integration & PGN Ingestion:** Fetch public games directly from any Lichess username or paste PGN notation to analyze entire games in batch mode.
- **⚡ Smart Batch Prefetching & Prewarming:** Analyzes full games in parallel and pre-warms audio caches for instant, lag-free move navigation.
- **📊 Evaluation & Benchmark Suite:** Built-in benchmarking scripts (`evaluate_commentary.py`, `evaluate_model.py`) to measure commentary coverage, precision, hallucination rate, and engine alignment against reference grandmaster datasets.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client (Browser)                              │
│         Interactive Board (chessboard.js)  |  Socket.IO Client         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ WebSocket Events
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Flask Backend Application (app.py)                 │
├───────────────────────────────┬─────────────────────────────────────────┤
│  Stockfish Engine             │  Tactics Analyzer                       │
│  (Position & CP Eval)         │  (Forks, Pins, Sacrifices, Threats)     │
├───────────────────────────────┼─────────────────────────────────────────┤
│  Commentary Logic Blueprint   │  Structured Natural Language Generator  │
│  (Lead, Support, Verdict)     │  (Rule-based fallback & verification)   │
└───────────────┬───────────────┴────────────────────────┬────────────────┘
                │ API Request                            │ Audio Waveform
                ▼                                        ▼
┌───────────────────────────────┐        ┌────────────────────────────────┐
│ Google Gemini 2.5 Flash API   │        │ KittenTTS Neural Audio Engine  │
│ (Polished Prose Generation)   │        │ (Lightning-Fast Speech Synthes)│
└───────────────────────────────┘        └────────────────────────────────┘
```

### Data Flow
1. **Board Input:** User plays a move on the web board or selects a game from Lichess / PGN.
2. **Engine Analysis:** Stockfish evaluates position before & after the move, returning centipawn evaluation, mate scores, and principal variations.
3. **Tactical Detection:** `tactics_analyzer.py` checks piece geometry, attack matrices, defender removal, and sacrifice patterns.
4. **Blueprint Construction:** `commentary_logic.py` constructs a structured narrative blueprint containing verified facts (lead, support, verdict, locked keywords).
5. **NLG & Synthesis:** `app.py` passes the blueprint to Gemini for natural rewriting, then KittenTTS synthesizes PCM audio returned over WebSockets.

---

## 📁 Repository Structure

```text
AiChessCommentator/
├── app.py                      # Primary Flask application server & Socket.IO handlers
├── commentary_logic.py         # Narrative blueprint builder & evaluation parser
├── tactics_analyzer.py         # Advanced tactical motif analyzer (forks, pins, sacrifices)
├── structured_commentary.py    # Rule-based structured natural language generation
├── semantic_commentary.py      # Semantic text formatting & quality validation
│
├── evaluate_commentary.py      # Commentary benchmarking script vs. ground truth
├── evaluate_model.py           # Metric evaluator (loss, bleu/rouge proxies, accuracy)
│
├── static/                     # Frontend static assets
│   ├── css/app.css             # Application stylesheet & badges layout
│   ├── js/game.js              # Client board controller & Socket.IO event handler
│   ├── img/                    # Piece sets (SVG/PNG) & classification icons
│   └── lib/                    # Client libraries (chess.js, chessboard.js)
│
├── templates/                  # HTML templates
│   ├── index.html              # Home page & Lichess game browser
│   └── game.html               # Game analysis & interactive board view
│
├── chess/                      # Board visual assets & audio effects
│   ├── sounds/                 # Sound files (move, capture, check, castle)
│   └── 3dboard.jpg             # Board background texture
│
├── stockfish/                  # Stockfish engine directory
│   └── stockfish-ubuntu-x86-64-avx2  # Precompiled engine binary
│
├── tests/                      # Unit & integration test suite
│   ├── test_commentary_quality.py   # Commentary validation tests
│   └── test_evaluate_commentary.py  # Evaluator unit tests
│
├── Grandmaster_Commentry.pgn   # Grandmaster reference commentary PGN dataset
├── commented_test_game.pgn     # Test game PGN with annotated commentary
├── [a-e].tsv                   # ECO opening book tabular datasets
├── requirements.txt            # Python package dependencies
├── package.json                # Frontend NPM configuration
└── .env                        # Environment variable configuration
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python:** `3.10` or higher
- **Stockfish:** Pre-compiled Linux binary included (`stockfish/stockfish-ubuntu-x86-64-avx2`). For Windows/macOS, download the corresponding Stockfish binary and update `STOCKFISH_PATH`.
- **API Keys (Optional):** Google Gemini API key for AI text enhancement.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AiChessCommentator.git
cd AiChessCommentator
```

### 2. Environment Setup
Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
npm install
```

### 4. Engine Permissions (Linux/macOS)
Ensure the Stockfish engine binary is executable:
```bash
chmod +x stockfish/stockfish-ubuntu-x86-64-avx2
```

### 5. Environment Configuration
Create a `.env` file in the root directory:
```ini
# Server Configuration
PORT=5000
DEBUG=True
SECRET_KEY=your_secret_key_here

# Stockfish Binary Path
STOCKFISH_PATH=stockfish/stockfish-ubuntu-x86-64-avx2

# AI Services
GEMINI_TEXT_API_KEY=your_gemini_api_key_here
DISABLE_KITTENTTS=False

# Lichess API (Optional - raises rate limits)
LICHESS_TOKEN=your_lichess_token_here
```

---

## 🚀 Usage

### 1. Running the Web Application
Start the server using Python:
```bash
python app.py
```
Open your browser and navigate to:
```text
http://localhost:5000
```

### 2. Live Interaction & Features
- **Play vs Engine / Self-Play:** Make moves on the board to receive live move commentary, sound effects, and accuracy classification badges.
- **Import Lichess User Games:** Enter a Lichess username on the home page to list recent games, click any game to open the full commentary board.
- **Batch Commentary Prefetch:** Click the prefetch button in game view to generate commentary and audio across the full game in the background.

### 3. Running Evaluation Benchmarks
Run the evaluation suite to benchmark commentary quality against reference games:
```bash
# Evaluate commentary alignment & quality
python evaluate_commentary.py

# Evaluate model metrics
python evaluate_model.py
```

---

## 📡 WebSocket API Reference

The server uses **Flask-SocketIO** to communicate real-time state with the frontend:

| Event Name | Direction | Payload Description |
| :--- | :--- | :--- |
| `analyze_position` | Client ➔ Server | Sends FEN string and engine settings (`depth`, `threads`) for evaluation. |
| `analysis_result` | Server ➔ Client | Returns centipawn eval, mate scores, and top candidate moves. |
| `get_move_quality` | Client ➔ Server | Compares positions before/after a move to classify accuracy. |
| `move_quality_result` | Server ➔ Client | Returns move quality tag (e.g. `brilliant`, `blunder`). |
| `get_ai_commentary` | Client ➔ Server | Requests live text commentary & optional audio for a ply. |
| `ai_commentary_text_result` | Server ➔ Client | Returns generated commentary string and move classification. |
| `synthesize_commentary_audio`| Client ➔ Server | Requests KittenTTS text-to-speech audio synthesis for text. |
| `ai_commentary_audio_result` | Server ➔ Client | Returns Base64-encoded WAV audio data for browser playback. |
| `prefetch_ai_commentary_batch`| Client ➔ Server | Batched analysis request for full game PGN array. |

---

## 🏷️ Move Quality & Badge Reference

| Badge | Symbol | Classification | Criteria / Evaluation Shift |
| :---: | :---: | :--- | :--- |
| **Brilliant** | 💎 | `brilliant` | Sacrificial move that maintains or improves winning advantage. |
| **Best / Great** | 🎯 | `best` | Top engine move or critical strategic find (loss ≤ 10 centipawns). |
| **Excellent** | 🟢 | `excellent` | Strong continuation maintaining advantage (loss ≤ 30 centipawns). |
| **Good** | 🟢 | `good` | Solid positionally sound move (loss ≤ 70 centipawns). |
| **Book** | 📖 | `book` | Standard opening theory move from opening database. |
| **Inaccuracy** | 🟡 | `inaccuracy` | Suboptimal move ceding minor advantage (loss ≤ 150 centipawns). |
| **Mistake** | 🟠 | `mistake` | Noticeable strategic error (loss ≤ 350 centipawns). |
| **Miss** | ❌ | `miss` | Overlooked tactic, checkmate, or winning continuation. |
| **Blunder** | 🔴 | `blunder` | Critical error severely impacting win probability (> 350 centipawns). |

---

## 🧪 Testing & Verification

Run the test suite using `pytest`:

```bash
pytest
```

To run a specific test module:
```bash
pytest tests/test_commentary_quality.py
```

---

## 📄 License

This project is open-source software and utilizes the **Stockfish Chess Engine** licensed under the **GNU General Public License v3.0 (GPLv3)**. See `stockfish/Copying.txt` for detailed licensing terms.

---

<div align="center">
  <sub>Built with ❤️ using Python, Flask, Stockfish, Gemini, and KittenTTS.</sub>
</div>
