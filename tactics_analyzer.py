import os
from typing import Any, Dict, Optional

import chess
import chess.engine

from tactics_positional_mixin import PositionalFeaturesMixin
from tactics_tactical_mixin import TacticalFeaturesMixin


class TacticsAnalyzer(TacticalFeaturesMixin, PositionalFeaturesMixin):
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100,
    }
    PIECE_VALUES_CENTIPAWNS = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    def __init__(self, engine_path: str):
        if not engine_path or not os.path.exists(engine_path):
            raise FileNotFoundError(f"Stockfish engine not found at path: {engine_path}")
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        except Exception as exc:
            print(f"Failed to initialize Stockfish engine: {exc}")
            self.engine = None
        self.HIGH_VALUE_THRESHOLD = self.PIECE_VALUES[chess.KNIGHT]

    def close(self):
        if self.engine:
            self.engine.quit()
            self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def analyze(self, current_fen: str, previous_fen: Optional[str] = None) -> Dict[str, Any]:
        if not self.engine:
            return {"error": "Stockfish engine not initialized."}

        board = chess.Board(current_fen)

        analysis = {
            "fen": current_fen,
            "static_analysis": {
                "pins": self._detect_positional_pins(board),
                "passed_pawns": self._detect_passed_pawns(board),
                "back_rank_weakness": self._detect_back_rank_weakness(board),
                "isolated_pawns": self._analyze_isolated_pawns(board),
                "skewers": self._find_and_validate_skewers(board),
                "forks": self._find_and_validate_forks(board),
                "piece_interactions": self._analyze_tactical_relationships(board),
            },
        }

        if previous_fen:
            prev_board = chess.Board(previous_fen)
            move = self._find_move_between_boards(prev_board, board)
            if move:
                analysis["analysis_of_move"] = {
                    "move_uci": move.uci(),
                    "discovered_attacks": self._find_discovered_attack(prev_board, move),
                    "clearance_sacrifices": self._find_clearance_sacrifice(prev_board, move),
                }

        return analysis

    def _get_piece_value(self, piece: chess.Piece) -> int:
        return self.PIECE_VALUES.get(piece.piece_type, 0)

    def _get_piece_value_cp(self, piece: chess.Piece) -> int:
        return self.PIECE_VALUES_CENTIPAWNS.get(piece.piece_type, 0)

    def _score_to_float(self, score: chess.engine.Score, pov_color: chess.Color) -> float:
        pov_score = score.pov(pov_color)
        if pov_score.is_mate():
            return 100.0 * (1 if pov_score.mate() > 0 else -1)
        return (pov_score.cp or 0) / 100.0

    def _find_move_between_boards(self, prev_board: chess.Board, curr_board: chess.Board) -> Optional[chess.Move]:
        for move in prev_board.legal_moves:
            board_after_move = prev_board.copy(stack=False)
            board_after_move.push(move)
            if board_after_move.board_fen() == curr_board.board_fen():
                return move
        return None
