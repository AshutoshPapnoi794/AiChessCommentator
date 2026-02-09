from typing import Dict, List

import chess
import chess.engine


class PositionalFeaturesMixin:
    def _get_piece_identifier(self, board: chess.Board, square: int) -> str:
        piece = board.piece_at(square)
        if not piece:
            return ""
        color = "white" if piece.color == chess.WHITE else "black"
        piece_name = chess.piece_name(piece.piece_type).title()
        square_name = chess.square_name(square)
        return f"{color} {piece_name} at {square_name}"

    def _get_see_score(self, board: chess.Board, square: int, attacker_sq: int) -> int:
        attacker_piece = board.piece_at(attacker_sq)
        victim_piece = board.piece_at(square)
        if not attacker_piece or not victim_piece:
            return 0

        gain = [self._get_piece_value_cp(victim_piece)]
        sim_board = board.copy()
        sim_board.remove_piece_at(attacker_sq)
        side_to_move = not attacker_piece.color

        attackers = sim_board.attackers(chess.WHITE, square) | sim_board.attackers(chess.BLACK, square)
        all_attackers_sqs = sorted(list(attackers), key=lambda s: self._get_piece_value_cp(sim_board.piece_at(s)))

        current_capture_value = self._get_piece_value_cp(attacker_piece)

        while all_attackers_sqs:
            lva_sq = next((sq for sq in all_attackers_sqs if sim_board.piece_at(sq).color == side_to_move), None)
            if lva_sq is None:
                break

            all_attackers_sqs.remove(lva_sq)
            gain.append(current_capture_value)
            current_capture_value = self._get_piece_value_cp(sim_board.piece_at(lva_sq))
            side_to_move = not side_to_move

        score = 0
        for i, value in enumerate(gain):
            score += value if i % 2 == 0 else -value
        return score

    def _analyze_tactical_relationships(self, board: chess.Board) -> List[str]:
        relations = []
        processed_pairs = set()

        for victim_sq in chess.SQUARES:
            victim_piece = board.piece_at(victim_sq)
            if not victim_piece:
                continue

            for attacker_sq in board.attackers(not victim_piece.color, victim_sq):
                attacker_piece = board.piece_at(attacker_sq)
                if not attacker_piece:
                    continue

                attacker_id = self._get_piece_identifier(board, attacker_sq)
                victim_id = self._get_piece_identifier(board, victim_sq)

                canonical_pair = tuple(sorted((attacker_id, victim_id)))
                if canonical_pair in processed_pairs:
                    continue

                if victim_piece.piece_type == chess.KING:
                    relations.append(f"{attacker_id} CHECKS the {victim_id}")
                    processed_pairs.add(canonical_pair)
                    continue

                see_attacker_wins = self._get_see_score(board, victim_sq, attacker_sq) > 0
                see_victim_wins = False
                if attacker_sq in board.attacks(victim_sq):
                    see_victim_wins = self._get_see_score(board, attacker_sq, victim_sq) > 0

                if see_attacker_wins:
                    relations.append(f"{attacker_id} THREATENS the {victim_id}")
                elif see_victim_wins:
                    relations.append(f"{victim_id} THREATENS the {attacker_id}")
                else:
                    relations.append(f"{attacker_id} pressures the {victim_id}")

                processed_pairs.add(canonical_pair)
        return sorted(relations)

    def _detect_passed_pawns(self, board: chess.Board) -> Dict[str, List[Dict]]:
        results = {"white": [], "black": []}
        all_pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
        for square in all_pawns:
            piece = board.piece_at(square)
            color, file, rank = piece.color, chess.square_file(square), chess.square_rank(square)

            files_to_check = {file} | {f for f in [file - 1, file + 1] if 0 <= f <= 7}
            ranks_to_check = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)

            has_opp_pawn = any(
                board.piece_type_at(chess.square(f, r)) == chess.PAWN for f in files_to_check for r in ranks_to_check
            )
            if not has_opp_pawn:
                color_key = "white" if color == chess.WHITE else "black"
                results[color_key].append({
                    "square": chess.square_name(square),
                    "is_protected": board.is_attacked_by(color, square),
                })
        return results

    def _detect_back_rank_weakness(self, board: chess.Board) -> List[Dict]:
        weaknesses = []
        for color in [chess.WHITE, chess.BLACK]:
            back_rank = 0 if color == chess.WHITE else 7
            king_square = board.king(color)
            if king_square is None or chess.square_rank(king_square) != back_rank:
                continue

            escape_squares = self._get_king_escape_squares(king_square, color)
            if any(not board.piece_at(sq) for sq in escape_squares):
                continue

            defenders = {
                s
                for f in range(8)
                for s in board.attackers(color, chess.square(f, back_rank))
                if board.piece_type_at(s) != chess.KING
            }
            num_defenders = len(defenders)

            severity = "Potential"
            if num_defenders == 0:
                severity = "Critical"
            elif num_defenders == 1:
                severity = "Severe"
            elif num_defenders == 2:
                severity = "Moderate"
            weaknesses.append(
                {
                    "color": "White" if color == chess.WHITE else "Black",
                    "severity": severity,
                    "defenders": sorted([chess.square_name(sq) for sq in defenders]),
                }
            )
        return weaknesses

    def _get_king_escape_squares(self, king_square: int, color: chess.Color) -> List[int]:
        file = chess.square_file(king_square)
        forward_rank = chess.square_rank(king_square) + (1 if color == chess.WHITE else -1)
        if not (0 <= forward_rank <= 7):
            return []
        return [chess.square(f, forward_rank) for f in [file - 1, file, file + 1] if 0 <= f <= 7]

    def _analyze_isolated_pawns(self, board: chess.Board) -> Dict[str, List[Dict]]:
        analysis = {"White": [], "Black": []}
        for color, color_name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            friendly_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
            for pawn_sq in board.pieces(chess.PAWN, color):
                pawn_file = chess.square_file(pawn_sq)
                if (pawn_file - 1) not in friendly_pawn_files and (pawn_file + 1) not in friendly_pawn_files:
                    pawn_info = {"square": chess.square_name(pawn_sq)}
                    if board.turn != color:
                        pawn_info["engine_tactic"] = self._get_opponent_tactic(board, pawn_sq)
                    else:
                        pawn_info["engine_tactic"] = self._get_owner_tactic(board, pawn_sq)
                    analysis[color_name].append(pawn_info)
        return analysis

    def _get_opponent_tactic(self, board: chess.Board, pawn_sq: int) -> str:
        pawn_color = board.piece_at(pawn_sq).color
        blockade_sq = pawn_sq + (8 if pawn_color == chess.WHITE else -8)
        temp_board = board.copy()
        temp_board.turn = not pawn_color
        try:
            info = self.engine.analyse(temp_board, chess.engine.Limit(time=0.2))
            best_move = info.get("pv", [None])[0]
            if not best_move:
                return "No legal moves."
            if best_move.to_square == blockade_sq:
                return f"Blockade with {temp_board.san(best_move)}"
            temp_board.push(best_move)
            if temp_board.is_attacked_by(not pawn_color, pawn_sq):
                return f"Attack with {board.san(best_move)}"
            return f"Other priorities (best move: {board.san(best_move)})"
        except Exception:
            return "Engine analysis failed."

    def _get_owner_tactic(self, board: chess.Board, pawn_sq: int) -> str:
        try:
            info = self.engine.analyse(board, chess.engine.Limit(time=0.2))
            best_move = info.get("pv", [None])[0]
            if not best_move:
                return "No legal moves."
            if best_move.from_square == pawn_sq:
                return f"Push the pawn with {board.san(best_move)}"
            return f"Other priorities (best move: {board.san(best_move)})"
        except Exception:
            return "Engine analysis failed."
