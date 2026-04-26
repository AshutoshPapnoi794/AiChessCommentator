import unittest
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SITE_PACKAGES = next((path for path in sys.path if "site-packages" in path), "")
if SITE_PACKAGES:
    sys.path = [SITE_PACKAGES] + [path for path in sys.path if path != SITE_PACKAGES and path not in {"", ROOT}] + [ROOT]
elif ROOT not in sys.path:
    sys.path.append(ROOT)

import chess

import app
from structured_commentary import build_structured_commentary


class CommentaryQualityTests(unittest.TestCase):
    def build_context(self, prefix_moves, san, sid, opening_name="Test Opening"):
        board = chess.Board()
        previous_move = None
        for prefix_san in prefix_moves:
            prev = board.copy(stack=False)
            move = prev.parse_san(prefix_san)
            board.push(move)
            previous_move = move

        prev = board.copy(stack=False)
        move = prev.parse_san(san)
        curr = prev.copy(stack=False)
        curr.push(move)

        cp_before = app.get_commentary_eval(sid, prev.fen())
        cp_after = app.get_commentary_eval(sid, curr.fen())
        quality = app.classify_move_quality(cp_before, cp_after, prev.turn == chess.WHITE)
        analysis_after = app._build_analysis_after(sid, prev, curr, move, True)
        missed = {}
        if quality in {"mistake", "blunder"}:
            missed = app.detect_missed_tactics(sid, prev, move, cp_before, cp_after, quality, tactics_enabled=True)

        return app.build_rich_context(
            prev_board=prev,
            curr_board=curr,
            move=move,
            move_san=san,
            analysis_after=analysis_after,
            cp_before=cp_before,
            cp_after=cp_after,
            opening_name=opening_name,
            missed_tactics=missed,
            tactics_enabled=True,
            previous_move=previous_move,
            sid=sid,
        )

    def assert_is_live_commentary(self, commentary):
        self.assertNotIn("Position Summary", commentary)
        self.assertNotIn("Tactical Observations", commentary)
        self.assertNotIn("\n\n", commentary)

    def test_caro_kann_move_mentions_opening_plan(self):
        context = self.build_context(["e4"], "c6", sid="caro-c6", opening_name="Caro-Kann Defense")
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("Caro-Kann", commentary)
        self.assertIn("...d5", commentary)
        self.assertIn("bishop on c8", commentary)

    def test_black_d_pawn_opens_c8_bishop_not_f8(self):
        context = self.build_context(["e4", "c6", "d4"], "d5", sid="caro-d5", opening_name="Caro-Kann Defense")
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("bishop on c8", commentary)
        self.assertNotIn("bishop on f8", commentary)

    def test_central_capture_is_described_as_trade_and_recapture(self):
        context = self.build_context(["e4", "c6", "d4", "d5"], "exd5", sid="trade", opening_name="Caro-Kann Defense")
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("trading for the pawn on d5", commentary)
        self.assertIn("cxd5", commentary)
        self.assertNotIn("taking space in the center", commentary)

    def test_blocking_check_mentions_block_and_pin(self):
        context = self.build_context(["e4", "d5", "Bb5+"], "Nc6", sid="block-pin", opening_name="Scandinavian Defense")
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("blocks the check", commentary)
        self.assertIn("knight on c6", commentary)
        self.assertIn("pinned", commentary)
        self.assertIn("king on e8", commentary)

    def test_development_supports_own_center_pawn(self):
        context = self.build_context(["d4", "d5"], "Nf3", sid="support-d4", opening_name="Queen Pawn Game")
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("supports the pawn on d4", commentary)
        self.assertNotIn("pressure to d4", commentary)

    def test_h3_questions_bishop_instead_of_talking_about_pawn_trade(self):
        context = self.build_context(["d4", "d5", "Nf3", "Nc6", "e3", "Bg4"], "h3", sid="h3", opening_name="Queen Pawn Game")
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("bishop on g4", commentary)
        self.assertIn("retreat", commentary)
        self.assertIn("knight on f3", commentary)
        self.assertNotIn("trading the pawn", commentary)

    def test_bishop_capture_is_treated_as_trade_not_pin(self):
        context = self.build_context(["d4", "d5", "Nf3", "Nc6", "e3", "Bg4", "h3"], "Bxf3", sid="bxf3", opening_name="Queen Pawn Game")
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("captures the knight on f3", commentary)
        self.assertIn("Qxf3", commentary)
        self.assertNotIn("useful pin", commentary)

    def test_qxf3_is_described_as_recapture_not_queen_sacrifice(self):
        context = self.build_context(["d4", "d5", "Nf3", "Nc6", "e3", "Bg4", "h3", "Bxf3"], "Qxf3", sid="qxf3", opening_name="Queen Pawn Game")
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("natural recapture", commentary)
        self.assertIn("restoring material balance", commentary)
        self.assertNotIn("gives itself up", commentary)
        self.assertIn("recaptures", context["draft_commentary"])

    def test_b5_mentions_queenside_space_and_cramped_bishop(self):
        context = self.build_context(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4"], "b5", sid="b5", opening_name="Ruy Lopez")
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("queenside", commentary)
        self.assertIn("bishop", commentary)
        self.assertNotIn("retreat or exchange itself", commentary)

    def test_bxc3_mentions_doubled_pawns_after_recapture(self):
        context = self.build_context(
            ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "O-O", "Bd3", "d5", "Nf3", "c5", "O-O"],
            "Bxc3",
            sid="bxc3-structure",
            opening_name="Nimzo-Indian Defense",
        )
        commentary = app.generate_fallback_commentary(context)

        self.assert_is_live_commentary(commentary)
        self.assertIn("bxc3", commentary)
        self.assertIn("doubled pawns on the c-file", commentary)
        self.assertIn("doubled pawns", context["draft_commentary"])

    def test_inaccurate_retreat_is_not_called_sensible(self):
        board = chess.Board()
        for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]:
            move = board.parse_san(san)
            board.push(move)

        prev = board.copy(stack=False)
        move = prev.parse_san("Ba4")
        curr = prev.copy(stack=False)
        curr.push(move)
        context = {
            "move": {"san": "Ba4", "mover": "White", "piece": "bishop"},
            "quality": "inaccuracy",
            "tactical": {},
            "analysis": {},
            "verified_facts": app.build_verified_facts(prev, curr, move),
        }
        commentary = build_structured_commentary(
            prev_board=prev,
            curr_board=curr,
            move=move,
            context=context,
            engine_info={"best_line_before": "Bxc6+ bxc6 O-O Bd6"},
        )

        self.assertIn("retreats", commentary)
        self.assertIn("damaging Black's pawn structure on the c-file", commentary)
        self.assertNotIn("sensible retreat", commentary)

    def test_qg3_blunder_is_called_hanging_queen(self):
        prev = chess.Board("4k3/8/8/8/4n3/8/6Q1/4K3 w - - 0 1")
        move = prev.parse_san("Qg3")
        curr = prev.copy(stack=False)
        curr.push(move)
        context = {
            "move": {"san": "Qg3", "mover": "White", "piece": "queen"},
            "quality": "blunder",
            "tactical": {"hung_piece": {"piece": "queen", "square": "g3"}},
            "analysis": {},
            "verified_facts": app.build_verified_facts(prev, curr, move),
        }
        commentary = build_structured_commentary(
            prev_board=prev,
            curr_board=curr,
            move=move,
            context=context,
            engine_info={"best_line_after": "Nxg3+", "best_move_after": "Nxg3+"},
        )

        self.assertIn("hanging the queen on g3", commentary)
        self.assertIn("Nxg3+", commentary)
        self.assertNotIn("practical move", commentary)

    def test_missed_tactic_blunder_mentions_critical_move_and_idea(self):
        prev = chess.Board("4k3/8/8/8/3q4/8/3Q4/4K3 w - - 0 1")
        move = prev.parse_san("Qc2")
        curr = prev.copy(stack=False)
        curr.push(move)
        context = {
            "move": {"san": "Qc2", "mover": "White", "piece": "queen"},
            "quality": "blunder",
            "tactical": {},
            "analysis": {},
            "verified_facts": app.build_verified_facts(prev, curr, move),
            "missed_tactics": {
                "missed_best_move": "Qxd4",
                "summary": "A disastrous mistake — Qxd4 wins the queen.",
                "tactical_details": {
                    "is_capture": True,
                    "captured_piece": "queen",
                    "gives_check": False,
                    "gives_checkmate": False,
                    "attack_info": {},
                    "tactic_types": [],
                },
            },
        }
        commentary = build_structured_commentary(
            prev_board=prev,
            curr_board=curr,
            move=move,
            context=context,
            engine_info={},
        )

        self.assertIn("Qxd4", commentary)
        self.assertIn("winning the queen immediately", commentary)
        self.assertIn("misses the tactical point", commentary)
        self.assertNotIn("cleaner line", commentary)

    def test_knight_taking_queen_is_not_called_exchange(self):
        prev = chess.Board("4k3/8/8/8/4n3/6Q1/8/7K b - - 0 1")
        move = prev.parse_san("Nxg3+")
        curr = prev.copy(stack=False)
        curr.push(move)
        context = {
            "move": {"san": "Nxg3+", "mover": "Black", "piece": "knight"},
            "quality": "best",
            "tactical": {
                "capture_analysis": app.tactics_analyzer.analyze_capture(prev, curr, move),
            },
            "analysis": {},
            "verified_facts": app.build_verified_facts(prev, curr, move),
        }
        commentary = build_structured_commentary(
            prev_board=prev,
            curr_board=curr,
            move=move,
            context=context,
            engine_info={},
        )

        self.assertIn("wins the queen", commentary)
        self.assertNotIn("exchange", commentary)


if __name__ == "__main__":
    unittest.main()
