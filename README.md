# Lichess Game Viewer with AI Commentary

This web application allows you to fetch, view, and analyze chess games from Lichess.org. It provides a user-friendly interface to review games and leverages the Stockfish chess engine for in-depth analysis and the Google Gemini API for generating AI-powered commentary.

## Features

-   **Fetch Lichess Games**: Enter a Lichess username to fetch their recent games.
-   **Interactive Game Viewer**: A clean and interactive chessboard to play through games move by move.
-   **Stockfish Engine Analysis**: Get real-time engine evaluation and top move recommendations.
-   **Move Quality Analysis**: Classify each move (e.g., Best, Excellent, Inaccuracy, Blunder).
-   **AI-Generated Commentary**: Generate expert-level commentary for any move using the Google Gemini API.
-   **Text-to-Speech (TTS)**: Listen to the AI commentary for a more immersive experience.

## Setup and Installation

### Prerequisites

-   Python 3.10+
-   [Stockfish Chess Engine](https://stockfishchess.org/download/)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root of the project directory and add the following variables:

```
# A secret key for Flask sessions
SECRET_KEY='a_very_secret_key'

# Your Google Gemini API Key
GEMINI_API_KEY='your_gemini_api_key'

# Path to the Stockfish executable
STOCKFISH_PATH='/path/to/your/stockfish/executable'
```

-   `GEMINI_API_KEY`: Required for AI commentary and TTS features.
-   `STOCKFISH_PATH`: If not set, the application will attempt to find a Stockfish executable in a `stockfish/` directory in the project root.

## Usage

1.  **Start the Flask Application**:

    ```bash
    python3 app.py
    ```

2.  **Open in Browser**:
    Navigate to `http://127.0.0.1:5000` in your web browser.

3.  **Fetch Games**:
    Enter a valid Lichess username and click "Fetch Games".

4.  **View and Analyze**:
    Click on a game from the list to view it. Use the controls to navigate through the game, request engine analysis, and generate AI commentary.
