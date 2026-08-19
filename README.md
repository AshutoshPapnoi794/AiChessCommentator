# AI Chess Commentator

An interactive chess platform and analysis engine that delivers real-time, automated grandmaster-level chess commentary. Combining Stockfish NNUE position evaluation with semantic natural language generation, the system assesses moves, assigns classification badges (e.g., *Brilliant*, *Blunder*, *Book*), and generates dynamic broadcast-style commentary as games unfold.

---

## Key Features

* **Real-Time Semantic Commentary:** Translates tactical motifs, positional shifts, and centipawn evaluations into human-like grandmaster commentary.
* **Stockfish-Powered Move Classification:** Evaluates moves against engine benchmarks and tags them with Chess.com-style classifications:
* 💎 **Brilliant**
* 🎯 **Best / Great / Excellent / Good**
* 📖 **Book**
* ⚠️ **Inaccuracy / Mistake / Miss / Blunder**


* **Interactive Web GUI:** Play directly in browser using an integrated chessboard interface (`chessboard.js` and `chess.js`) with legal move validation and piece movement sounds.
* **PGN Ingestion & Evaluation Suite:** Ingests PGN games (including curated grandmaster datasets) and evaluates commentary quality against ground-truth benchmarks.
* **Audio & Visual Effects:** Includes move sounds (captures, checks, castling, promotions) and 2D/3D board coordinate visualization.

---

## Repository Structure

```text
AiChessCommentator/
├── app.py                      # Core web server & routing logic
├── commentary_logic.py         # Move assessment & evaluation parsing logic
├── semantic_commentary.py      # Natural language generation for commentary
├── evaluate_commentary.py      # Commentary benchmarking & scoring scripts
├── evaluate_model.py           # Model evaluation and loss/metric tracking
│
├── stockfish/                  # Stockfish chess engine source & binaries
│   ├── stockfish-ubuntu-x86-64-avx2  # Precompiled Linux binary
│   └── src/                    # Engine source code & Makefile
│
├── chess/                      # Board assets & audio sound effects
│   ├── sounds/                 # MP3 effects (capture, castle, check, etc.)
│   └── 3dboard.jpg             # Board visualization texture
│
├── static/                     # Web application frontend assets
│   ├── css/app.css             # Main stylesheet
│   ├── js/game.js              # Client-side board state & API handler
│   ├── img/
│   │   ├── pieces/             # SVG/PNG chess piece sets
│   │   └── symbols/            # Accuracy badges (blunder, brilliant, etc.)
│   └── lib/                    # Client libraries (chess.js, chessboard.js)
│
├── Grandmaster_Commentry.pgn   # Training/reference grandmaster games
├── commented_test_game.pgn     # Test game dataset with commentary
├── [a-e].tsv                   # Tabular datasets for training and eval
├── requirements.txt            # Python dependencies
├── package.json                # Frontend package dependencies
└── .env                        # Environment configurations

```

---

## System Architecture

```
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  Browser GUI    │ <───> │   app.py (Server)    │ <───> │  Stockfish Engine      │
│ (chessboard.js) │       │                      │       │  (NNUE Evaluation)     │
└─────────────────┘       └──────────┬───────────┘       └────────────────────────┘
                                     │
                          ┌──────────┴───────────┐
                          │ Commentary Logic &   │
                          │ Semantic Engine      │
                          └──────────────────────┘

```

1. **Board Input:** The user makes a move on the web board or uploads a PGN.
2. **Engine Evaluation:** Stockfish evaluates position score, optimal lines, and depth changes.
3. **Move Classification:** `commentary_logic.py` compares the played move against engine suggestions to categorize the move type.
4. **Semantic Generation:** `semantic_commentary.py` transforms board state variables (threats, piece mobility, king safety, tactics) into contextual commentary.
5. **Client Stream:** The frontend updates the board, plays move audio, places accuracy icons, and renders commentary in real time.

---

## Installation & Setup

### Prerequisites

* **Python:** Version 3.10 or higher
* **Node.js & npm:** (Optional, for managing frontend dependencies)
* **Stockfish:** Linux x86-64 binary provided (`stockfish/stockfish-ubuntu-x86-64-avx2`). For macOS or Windows, install the appropriate binary or compile from `stockfish/src`.

### 1. Clone & Environment Setup

```bash
git clone <repository-url>
cd AiChessCommentator

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
npm install

```

### 3. Configure Engine Permissions (Linux/macOS)

Ensure the bundled Stockfish engine has execute permissions:

```bash
chmod +x stockfish/stockfish-ubuntu-x86-64-avx2

```

*(Optional)* If compiling Stockfish from source:

```bash
cd stockfish/src
make -j profile-build ARCH=x86-64-avx2
cd ../..

```

### 4. Configure Environment Variables

Create or adjust `.env` in the root directory:

```ini
PORT=5000
DEBUG=True
STOCKFISH_PATH=stockfish/stockfish-ubuntu-x86-64-avx2

```

---

## Usage

### Running the Web Application

Start the local server:

```bash
python app.py

```

Open `http://localhost:5000` (or the configured port) in your web browser.

### Running Evaluation Scripts

To benchmark commentary outputs against test games and ground-truth TSV datasets:

```bash
# Evaluate model commentary quality
python evaluate_commentary.py

# Run model evaluation metrics
python evaluate_model.py

```

---

## Move Classification Reference

| Symbol | Badge Name | Criteria |
| --- | --- | --- |
| 💎 | **Brilliant** | Sacrificial move that is the best or only winning continuation. |
| 🎯 | **Great / Best** | Highest engine-ranked move in a critical position. |
| 🟢 | **Excellent / Good** | Solid strategic continuation with minimal score deviation. |
| 📖 | **Book** | Recognized theoretical opening move. |
| 🟡 | **Inaccuracy** | Suboptimal move conceding a minor positional advantage. |
| 🟠 | **Mistake** | Noticeable error worsening the evaluation significantly. |
| ❌ | **Miss** | Overlooked strong tactic or winning continuation. |
| 🔴 | **Blunder** | Critical error drastically shifting the game outcome. |

---

## License

This project utilizes open-source software, including the Stockfish chess engine licensed under the **GNU General Public License v3.0 (GPLv3)**. See `stockfish/Copying.txt` for detailed licensing terms.
