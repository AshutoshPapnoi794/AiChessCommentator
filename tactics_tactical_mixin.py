from typing import Dict, List

import chess
import chess.engine


class TacticalFeaturesMixin:
    def _detect_positional_pins(self, board: chess.Board) -> List[Dict]:
        pins = []
        for pinner_sq in range(64):
            pinner_piece = board.piece_at(pinner_sq)
            if not pinner_piece or pinner_piece.piece_type not in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
                continue

            directions = []
            if pinner_piece.piece_type in [chess.ROOK, chess.QUEEN]:
                directions.extend([(0, 1), (0, -1), (1, 0), (-1, 0)])
            if pinner_piece.piece_type in [chess.BISHOP, chess.QUEEN]:
                directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

            pinner_rank, pinner_file = chess.square_rank(pinner_sq), chess.square_file(pinner_sq)

            for dr, df in directions:
                ray_pieces = []
                for i in range(1, 8):
                    rank, file = pinner_rank + i * dr, pinner_file + i * df
                    if not (0 <= rank < 8 and 0 <= file < 8):
                        break
                    sq = chess.square(file, rank)
                    piece_on_ray = board.piece_at(sq)
                    if piece_on_ray:
                        ray_pieces.append((sq, piece_on_ray))

                if len(ray_pieces) < 2:
                    continue

                pinned_sq, pinned_piece = ray_pieces[0]
                valuable_sq, valuable_piece = ray_pieces[1]

                if not (pinned_piece.color != pinner_piece.color and valuable_piece.color == pinned_piece.color):
                    continue

                is_diagonal_pin = dr != 0 and df != 0
                if pinned_piece.piece_type == chess.PAWN and not is_diagonal_pin:
                    continue

                pin_type = None
                if valuable_piece.piece_type == chess.KING:
                    pin_type = "Absolute"
                elif self._get_piece_value_cp(valuable_piece) > self._get_piece_value_cp(pinned_piece):
                    pin_type = "Relative"

                if pin_type:
                    pins.append(
                        {
                            "type": pin_type,
                            "pinner_square": chess.square_name(pinner_sq),
                            "pinner_piece": pinner_piece.symbol(),
                            "pinned_square": chess.square_name(pinned_sq),
                            "pinned_piece": pinned_piece.symbol(),
                            "valuable_piece_square": chess.square_name(valuable_sq),
                            "valuable_piece": valuable_piece.symbol(),
                        }
                    )

        return [dict(t) for t in {tuple(d.items()) for d in pins}]

    def _find_and_validate_forks(self, board: chess.Board) -> List[Dict]:
        candidates = self._detect_fork_candidates(board)
        return self._verify_forks_with_engine(board, candidates)

    def _detect_fork_candidates(self, board: chess.Board) -> List[Dict]:
        candidates = []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if not piece:
                continue

            attacked_squares = board.attacks(sq)
            targets = []
            for target_sq in attacked_squares:
                target_piece = board.piece_at(target_sq)
                if target_piece and target_piece.color != piece.color and self._get_piece_value(target_piece) >= self.HIGH_VALUE_THRESHOLD:
                    targets.append(
                        {
                            "piece": target_piece.symbol(),
                            "square": chess.square_name(target_sq),
                            "value_cp": self._get_piece_value_cp(target_piece),
                        }
                    )

            if len(targets) >= 2:
                candidates.append(
                    {
                        "forking_piece": piece.symbol(),
                        "forking_piece_square": chess.square_name(sq),
                        "forking_piece_value_cp": self._get_piece_value_cp(piece),
                        "targets": targets,
                    }
                )
        return candidates

    def _capture_moves(self, board: chess.Board, from_sq: int, to_sq: int, piece_type: int) -> List[chess.Move]:
        if piece_type == chess.PAWN and chess.square_rank(to_sq) in {0, 7}:
            return [
                chess.Move(from_sq, to_sq, promotion=chess.QUEEN),
                chess.Move(from_sq, to_sq, promotion=chess.ROOK),
                chess.Move(from_sq, to_sq, promotion=chess.BISHOP),
                chess.Move(from_sq, to_sq, promotion=chess.KNIGHT),
            ]
        return [chess.Move(from_sq, to_sq)]

    def _capture_trade_delta_cp(self, board: chess.Board, move: chess.Move, forking_piece_value_cp: int) -> int:
        target_piece = board.piece_at(move.to_square)
        if not target_piece:
            return -99999

        target_value_cp = self._get_piece_value_cp(target_piece)
        after_capture = board.copy(stack=False)
        after_capture.push(move)

        recapture_exists = any(
            legal_move.to_square == move.to_square and after_capture.is_capture(legal_move)
            for legal_move in after_capture.legal_moves
        )
        if recapture_exists:
            return target_value_cp - forking_piece_value_cp
        return target_value_cp

    def _is_profitable_fork_capture(
        self,
        forking_piece_value_cp: int,
        target_piece_value_cp: int,
        trade_delta_cp: int,
    ) -> bool:
        # Fork motif requires that the forking unit can actually win material.
        # We reject "forks" where the forking piece is more valuable than the target.
        if forking_piece_value_cp > target_piece_value_cp:
            return False
        # Same-value forks only count when the defender cannot neutralize and
        # the eventual capture is genuinely winning.
        if forking_piece_value_cp == target_piece_value_cp:
            return trade_delta_cp > 0
        return trade_delta_cp >= 0

    def _profitable_fork_captures_after_defense(
        self,
        board_after_defense: chess.Board,
        forking_piece_sq: int,
        forking_color: chess.Color,
        target_squares: List[int],
        forking_piece_value_cp: int,
    ) -> List[Dict]:
        if board_after_defense.turn != forking_color:
            return []

        forking_piece = board_after_defense.piece_at(forking_piece_sq)
        if not forking_piece or forking_piece.color != forking_color:
            return []

        profitable: List[Dict] = []
        for target_sq in target_squares:
            target_piece = board_after_defense.piece_at(target_sq)
            if not target_piece or target_piece.color == forking_color:
                continue

            for move in self._capture_moves(board_after_defense, forking_piece_sq, target_sq, forking_piece.piece_type):
                if move not in board_after_defense.legal_moves:
                    continue
                trade_delta_cp = self._capture_trade_delta_cp(
                    board_after_defense,
                    move,
                    forking_piece_value_cp=forking_piece_value_cp,
                )
                target_value_cp = self._get_piece_value_cp(target_piece)
                if self._is_profitable_fork_capture(forking_piece_value_cp, target_value_cp, trade_delta_cp):
                    profitable.append(
                        {
                            "square": chess.square_name(target_sq),
                            "piece": target_piece.symbol(),
                            "trade_delta_cp": trade_delta_cp,
                        }
                    )
                    break

        return profitable

    def _is_relevant_fork_defense_move(
        self,
        board_before_defense: chess.Board,
        board_after_defense: chess.Board,
        defense_move: chess.Move,
        forking_piece_sq: int,
        target_squares: List[int],
        defender_color: chess.Color,
    ) -> bool:
        # Capturing the forking piece is always a direct fork defense.
        if defense_move.to_square == forking_piece_sq and board_before_defense.is_capture(defense_move):
            return True

        # Moving one of the forked targets is a direct fork defense.
        if defense_move.from_square in target_squares:
            return True

        # Any move that increases defender coverage on a forked target is also a direct defense.
        for target_sq in target_squares:
            defenders_before = len(board_before_defense.attackers(defender_color, target_sq))
            defenders_after = len(board_after_defense.attackers(defender_color, target_sq))
            if defenders_after > defenders_before:
                return True

        return False

    def _verify_forks_with_engine(self, board: chess.Board, fork_candidates: List[Dict]) -> List[Dict]:
        if not fork_candidates:
            return []

        verified_forks = []
        for candidate in fork_candidates:
            forking_piece_sq = chess.parse_square(candidate['forking_piece_square'])
            forking_piece = board.piece_at(forking_piece_sq)
            if not forking_piece:
                continue

            forking_color = forking_piece.color
            forking_piece_value_cp = candidate.get('forking_piece_value_cp', self._get_piece_value_cp(forking_piece))
            target_squares = []
            for target in candidate.get('targets', []):
                try:
                    target_squares.append(chess.parse_square(target['square']))
                except (KeyError, ValueError):
                    continue
            if len(target_squares) < 2:
                continue

            winning_targets = set()
            defender_color = not forking_color

            # If defender is to move, a true fork must survive every reasonable one-move defense:
            # if defender can neutralize in one move so no profitable target remains, it's not a fork.
            if board.turn == defender_color:
                defender_can_neutralize = False
                saw_relevant_defense = False
                for defense_move in board.legal_moves:
                    defended_board = board.copy(stack=False)
                    defended_board.push(defense_move)
                    if not self._is_relevant_fork_defense_move(
                        board_before_defense=board,
                        board_after_defense=defended_board,
                        defense_move=defense_move,
                        forking_piece_sq=forking_piece_sq,
                        target_squares=target_squares,
                        defender_color=defender_color,
                    ):
                        continue
                    saw_relevant_defense = True
                    captures = self._profitable_fork_captures_after_defense(
                        board_after_defense=defended_board,
                        forking_piece_sq=forking_piece_sq,
                        forking_color=forking_color,
                        target_squares=target_squares,
                        forking_piece_value_cp=forking_piece_value_cp,
                    )
                    if not captures:
                        defender_can_neutralize = True
                        break
                    winning_targets.update(c['square'] for c in captures)

                if defender_can_neutralize:
                    continue
                if not saw_relevant_defense:
                    probe_board = board.copy(stack=False)
                    probe_board.turn = forking_color
                    captures = self._profitable_fork_captures_after_defense(
                        board_after_defense=probe_board,
                        forking_piece_sq=forking_piece_sq,
                        forking_color=forking_color,
                        target_squares=target_squares,
                        forking_piece_value_cp=forking_piece_value_cp,
                    )
                    if not captures:
                        continue
                    winning_targets.update(c['square'] for c in captures)
            else:
                captures = self._profitable_fork_captures_after_defense(
                    board_after_defense=board,
                    forking_piece_sq=forking_piece_sq,
                    forking_color=forking_color,
                    target_squares=target_squares,
                    forking_piece_value_cp=forking_piece_value_cp,
                )
                if not captures:
                    continue
                winning_targets.update(c['square'] for c in captures)

            if winning_targets:
                enriched_candidate = dict(candidate)
                enriched_candidate['winning_targets'] = sorted(winning_targets)
                verified_forks.append(enriched_candidate)

        return verified_forks

    def _find_and_validate_skewers(self, board: chess.Board) -> List[Dict]:
        candidates = self._find_skewer_candidates_from_notebook(board)
        validated_skewers = []
        for cand in candidates:
            if self._validate_skewer_with_engine(board, cand):
                validated_skewers.append(cand)
        return validated_skewers

    def _find_skewer_candidates_from_notebook(self, board: chess.Board) -> List[Dict]:
        candidates = []
        for a_sq, a_piece in board.piece_map().items():
            if a_piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                continue
            for s_sq in board.attacks(a_sq):
                s_piece = board.piece_at(s_sq)
                if not s_piece or s_piece.color == a_piece.color:
                    continue
                ray = chess.SquareSet.ray(a_sq, s_sq)
                if not ray:
                    continue

                behind = [r for r in ray if r != s_sq and board.piece_at(r)]
                if not behind:
                    continue
                b_sq = behind[0]
                b_piece = board.piece_at(b_sq)
                if not b_piece or b_piece.color != s_piece.color:
                    continue

                if self._get_piece_value(s_piece) > self._get_piece_value(b_piece):
                    candidates.append(
                        {
                            "attacker_sq": a_sq,
                            "attacker": a_piece,
                            "skewed_sq": s_sq,
                            "skewed": s_piece,
                            "behind_sq": b_sq,
                            "behind": b_piece,
                        }
                    )
        return candidates

    def _validate_skewer_with_engine(self, board: chess.Board, candidate: Dict) -> bool:
        attacker_color = candidate["attacker"].color
        skewed_color = candidate["skewed"].color
        a_sq, s_sq, b_sq = candidate["attacker_sq"], candidate["skewed_sq"], candidate["behind_sq"]

        try:
            eval_before_info = self.engine.analyse(board, chess.engine.Limit(depth=14))
            eval_before = eval_before_info["score"].pov(attacker_color).score(mate_score=10000)

            move_capture = chess.Move(a_sq, s_sq)
            if move_capture not in board.legal_moves:
                return False

            temp_board_capture = board.copy()
            temp_board_capture.push(move_capture)
            eval_after_capture = self.engine.analyse(temp_board_capture, chess.engine.Limit(depth=14))["score"].pov(attacker_color).score(mate_score=10000)

            if (eval_after_capture or 0) - (eval_before or 0) < 50:
                return False

            board_for_skewed = board.copy()
            board_for_skewed.turn = skewed_color

            best_move = self.engine.play(board_for_skewed, chess.engine.Limit(depth=14)).move
            board_for_skewed.push(best_move)
            eval_after_skewed_best = self.engine.analyse(board_for_skewed, chess.engine.Limit(depth=14))["score"].pov(attacker_color).score(mate_score=10000)

            still_attacked = board_for_skewed.is_attacked_by(attacker_color, b_sq)
            eval_gain = (eval_after_capture or 0) - (eval_after_skewed_best or 0)

            return eval_gain > 100 and still_attacked
        except (chess.engine.EngineError, IndexError, AttributeError):
            return False

    def _find_discovered_attack(self, prev_board: chess.Board, move: chess.Move) -> List[Dict]:
        curr_board = prev_board.copy()
        curr_board.push(move)
        moving_piece, moving_color = prev_board.piece_at(move.from_square), prev_board.color_at(move.from_square)
        discoveries = []

        for sq in (
            prev_board.pieces(chess.ROOK, moving_color)
            | prev_board.pieces(chess.BISHOP, moving_color)
            | prev_board.pieces(chess.QUEEN, moving_color)
        ):
            if sq == move.from_square:
                continue
            newly_attacked = curr_board.attacks(sq) - prev_board.attacks(sq)
            for target_sq in newly_attacked:
                target = curr_board.piece_at(target_sq)
                if target and target.color != moving_color:
                    discoveries.append(
                        {
                            "moving_piece": moving_piece.symbol(),
                            "revealed_attacker": prev_board.piece_at(sq).symbol(),
                            "revealed_attacker_square": chess.square_name(sq),
                            "target": target.symbol(),
                            "target_square": chess.square_name(target_sq),
                            "target_value": self._get_piece_value(target),
                        }
                    )
        return discoveries

    def _is_true_sacrifice(self, board: chess.Board, move: chess.Move, moving_color: chess.Color) -> bool:
        to_sq = move.to_square
        opponent_color = not moving_color

        if not board.is_attacked_by(opponent_color, to_sq):
            return False

        opponent_attackers = board.attackers(opponent_color, to_sq)
        if not opponent_attackers:
            return False

        lowest_attacker_sq = min(opponent_attackers, key=lambda sq: self._get_piece_value(board.piece_at(sq)))

        temp_board = board.copy()
        try:
            recapture_move = chess.Move(lowest_attacker_sq, to_sq)
            if board.piece_type_at(lowest_attacker_sq) == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
                recapture_move.promotion = chess.QUEEN

            temp_board.push(recapture_move)
        except AssertionError:
            return True

        if temp_board.is_attacked_by(moving_color, to_sq):
            return False

        return True

    def _find_clearance_sacrifice(self, prev_board: chess.Board, move: chess.Move) -> List[Dict]:
        curr_board = prev_board.copy()
        curr_board.push(move)
        sac_piece = prev_board.piece_at(move.from_square)
        moving_color = sac_piece.color

        if not self._is_true_sacrifice(curr_board, move, moving_color):
            return []

        clearance_details = None
        for piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
            for sq in prev_board.pieces(piece_type, moving_color):
                if sq != move.from_square and (curr_board.attacks(sq) - prev_board.attacks(sq)):
                    clearance_details = {"type": "Line-Clearing", "cleared_for_piece_on": chess.square_name(sq)}
                    break
            if clearance_details:
                break

        if not clearance_details:
            vacated_sq = move.from_square
            temp_board = curr_board.copy()
            temp_board.turn = moving_color
            for pt in [chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP]:
                for mover_sq in curr_board.pieces(pt, moving_color):
                    potential_move = chess.Move(mover_sq, vacated_sq)
                    if potential_move in temp_board.legal_moves:
                        b_after = temp_board.copy()
                        b_after.push(potential_move)
                        if b_after.is_check():
                            clearance_details = {
                                "type": "Square-Clearing (Vacating)",
                                "vacated_square": chess.square_name(vacated_sq),
                                "for_piece_on": chess.square_name(mover_sq),
                            }
                            break
                if clearance_details:
                    break

        if not clearance_details:
            return []

        try:
            info_before = self.engine.analyse(prev_board, chess.engine.Limit(time=0.3))
            info_after = self.engine.analyse(curr_board, chess.engine.Limit(time=0.3))
            eval_change = self._score_to_float(info_after['score'], moving_color) - self._score_to_float(info_before['score'], moving_color)

            if eval_change < -1.5 and not info_after['score'].is_mate():
                return []

            return [{"evaluation_change": round(eval_change, 2), **clearance_details}]
        except (chess.engine.EngineError, IndexError):
            return []
