# tactics_analyzer.py

import chess
import chess.engine
import os
from typing import List, Dict, Optional, Any

class TacticsAnalyzer:
    """
    Analyzes chess positions for tactical and positional features using a Stockfish engine.

    This class provides detailed, structured output suitable for guiding Large Language Models (LLMs)
    in generating chess commentary. It uses a persistent engine instance for efficiency.
    """

    PIECE_VALUES = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
    }
    PIECE_VALUES_CENTIPAWNS = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
    }
    HIGH_VALUE_THRESHOLD_FOR_FORK = PIECE_VALUES[chess.KNIGHT]

    def __init__(self, engine_path: str, analysis_depth: int = 14):
        """
        Initializes the analyzer with a chess engine.

        Args:
            engine_path (str): Path to the Stockfish executable.
            analysis_depth (int): The default depth for engine analysis.

        Raises:
            FileNotFoundError: If the engine executable is not found at the given path.
            chess.engine.EngineError: If the engine process fails to start.
        """
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Chess engine not found at: {engine_path}")

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            self.analysis_limit = chess.engine.Limit(depth=analysis_depth)
        except chess.engine.EngineError as e:
            print(f"Failed to initialize Stockfish engine: {e}")
            self.engine = None
            raise


    def close(self):
        """Safely closes the Stockfish engine process."""
        if self.engine:
            self.engine.quit()
            self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def analyze(self, current_fen: str, previous_fen: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs a comprehensive tactical and positional analysis of a board state.

        Args:
            current_fen (str): The FEN string of the current position to analyze.
            previous_fen (Optional[str]): The FEN string of the position before the last move.
                                          If provided, move-specific tactics will be analyzed.

        Returns:
            Dict[str, Any]: A dictionary containing all detected tactical and positional features.
        """
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
                "piece_interactions": self._analyze_tactical_relationships(board), # <-- NEW
            }
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

    # --- HELPER METHODS --------------------------------------------------------

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

    # --- TACTIC 1: PINS (Restored from notebook cell 6f7f6558) -------------------

    def _detect_positional_pins(self, board: chess.Board) -> List[Dict]:
        """Detects significant POSITIONAL PINS using the original ray-casting logic."""
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
                    if not (0 <= rank < 8 and 0 <= file < 8): break
                    sq = chess.square(file, rank)
                    piece_on_ray = board.piece_at(sq)
                    if piece_on_ray:
                        ray_pieces.append((sq, piece_on_ray))

                if len(ray_pieces) >= 2:
                    pinned_sq, pinned_piece = ray_pieces[0]
                    valuable_sq, valuable_piece = ray_pieces[1]

                    if not (pinned_piece.color != pinner_piece.color and valuable_piece.color == pinned_piece.color):
                        continue
                    
                    is_diagonal_pin = (dr != 0 and df != 0)
                    if pinned_piece.piece_type == chess.PAWN and not is_diagonal_pin:
                        continue
                    
                    pin_type = None
                    if valuable_piece.piece_type == chess.KING:
                        pin_type = "Absolute"
                    elif self._get_piece_value_cp(valuable_piece) > self._get_piece_value_cp(pinned_piece):
                        pin_type = "Relative"

                    if pin_type:
                        pins.append({
                            "type": pin_type, "pinner_square": chess.square_name(pinner_sq),
                            "pinner_piece": pinner_piece.symbol(), "pinned_square": chess.square_name(pinned_sq),
                            "pinned_piece": pinned_piece.symbol(), "valuable_piece_square": chess.square_name(valuable_sq),
                            "valuable_piece": valuable_piece.symbol()
                        })
        
        return [dict(t) for t in {tuple(d.items()) for d in pins}]

    # --- TACTIC 2: FORKS (Logic from notebook cell 06ca9308) ---------------------

    def _find_and_validate_forks(self, board: chess.Board) -> List[Dict]:
        candidates = self._detect_fork_candidates(board)
        return self._verify_forks_with_engine(board, candidates)

    def _detect_fork_candidates(self, board: chess.Board) -> List[Dict]:
        """Detects potential fork situations for all pieces."""
        candidates = []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if not piece:
                continue

            attacked_squares = board.attacks(sq)
            targets = [
                {"piece": board.piece_at(tsq).symbol(), "square": chess.square_name(tsq)}
                for tsq in attacked_squares
                if board.piece_at(tsq) and board.piece_at(tsq).color != piece.color and
                   self._get_piece_value(board.piece_at(tsq)) >= self.HIGH_VALUE_THRESHOLD_FOR_FORK
            ]

            if len(targets) >= 2:
                candidates.append({
                    "forking_piece": piece.symbol(),
                    "forking_piece_square": chess.square_name(sq),
                    "targets": targets
                })
        return candidates

    def _verify_forks_with_engine(self, board: chess.Board, fork_candidates: List[Dict]) -> List[Dict]:
        """Uses the engine to verify if a fork is tactically sound."""
        if not fork_candidates:
            return []

        verified_forks = []
        for candidate in fork_candidates:
            forking_piece_sq = chess.parse_square(candidate['forking_piece_square'])
            forking_piece = board.piece_at(forking_piece_sq)
            if not forking_piece:
                continue

            is_fork_sound = self._is_fork_sound(board.copy(), forking_piece, candidate['targets'])
            if is_fork_sound:
                verified_forks.append(candidate)

        return verified_forks

    def _is_fork_sound(self, board: chess.Board, forking_piece: chess.Piece, targets: List[Dict]) -> bool:
        """Helper to determine if a single fork is sound."""
        try:
            initial_info = self.engine.analyse(board, self.analysis_limit)
            eval_before = self._get_pov_score(initial_info, forking_piece.color)

            if board.turn == forking_piece.color:
                # If it's our turn, check if the best move is one of the fork captures
                if 'pv' in initial_info and initial_info['pv']:
                    best_move = initial_info['pv'][0]
                    target_squares = {chess.parse_square(t['square']) for t in targets}
                    if best_move.from_square == forking_piece.from_square and best_move.to_square in target_squares:
                        return True
            else:
                # If it's opponent's turn, see if their best move leads to a bad position for them
                if 'pv' in initial_info and initial_info['pv']:
                    board.push(initial_info['pv'][0])
                    info_after = self.engine.analyse(board, self.analysis_limit)
                    eval_after = self._get_pov_score(info_after, forking_piece.color)
                    if (eval_after - eval_before) > 90:  # Threshold for significant gain
                        return True
        except (chess.engine.EngineError, IndexError):
            return False
        return False

    def _get_pov_score(self, analysis_info: Dict, color: chess.Color) -> int:
        """Extracts the score from analysis info from a specific color's perspective."""
        score = analysis_info["score"].pov(color)
        return score.score(mate_score=30000) or 0

    # --- TACTIC 3: SKEWERS ---

    def _find_and_validate_skewers(self, board: chess.Board) -> List[Dict]:
        """Finds and validates skewer tactical opportunities."""
        candidates = self._find_skewer_candidates(board)
        return [cand for cand in candidates if self._validate_skewer_with_engine(board, cand)]

    def _find_skewer_candidates(self, board: chess.Board) -> List[Dict]:
        """Identifies potential skewer candidates based on piece alignments."""
        candidates = []
        for attacker_sq, attacker_piece in board.piece_map().items():
            if attacker_piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                continue
            for skewed_sq in board.attacks(attacker_sq):
                skewed_piece = board.piece_at(skewed_sq)
                if not skewed_piece or skewed_piece.color == attacker_piece.color:
                    continue

                ray = chess.SquareSet.ray(attacker_sq, skewed_sq)
                behind_pieces = [sq for sq in ray if sq != skewed_sq and board.piece_at(sq)]
                if not behind_pieces:
                    continue

                behind_sq = behind_pieces[0]
                behind_piece = board.piece_at(behind_sq)
                if behind_piece and behind_piece.color == skewed_piece.color and \
                   self._get_piece_value(skewed_piece) > self._get_piece_value(behind_piece):
                    candidates.append({
                        "attacker_sq": attacker_sq, "attacker": attacker_piece,
                        "skewed_sq": skewed_sq, "skewed": skewed_piece,
                        "behind_sq": behind_sq, "behind": behind_piece,
                    })
        return candidates

    def _validate_skewer_with_engine(self, board: chess.Board, candidate: Dict) -> bool:
        """Uses the engine to validate if a skewer is a real tactical threat."""
        attacker_color = candidate["attacker"].color
        a_sq, s_sq, b_sq = candidate["attacker_sq"], candidate["skewed_sq"], candidate["behind_sq"]
        
        try:
            eval_before = self._get_pov_score(self.engine.analyse(board, self.analysis_limit), attacker_color)

            move_capture = chess.Move(a_sq, s_sq)
            if move_capture not in board.legal_moves:
                return False
            
            board_after_capture = board.copy()
            board_after_capture.push(move_capture)
            eval_after_capture = self._get_pov_score(self.engine.analyse(board_after_capture, self.analysis_limit), attacker_color)

            if (eval_after_capture - eval_before) < 50:  # Not a significant gain from capture
                return False

            board_for_skewed_move = board.copy()
            board_for_skewed_move.turn = candidate["skewed"].color
            
            result = self.engine.play(board_for_skewed_move, self.analysis_limit)
            board_for_skewed_move.push(result.move)
            eval_after_skewed_best = self._get_pov_score(self.engine.analyse(board_for_skewed_move, self.analysis_limit), attacker_color)

            still_attacked = board_for_skewed_move.is_attacked_by(attacker_color, b_sq)
            eval_gain = eval_after_capture - eval_after_skewed_best

            return eval_gain > 100 and still_attacked
        except (chess.engine.EngineError, IndexError, AttributeError):
            return False

    # --- TACTIC 4: DISCOVERED ATTACKS (from cell c73f32a3) -----------------------

    def _find_discovered_attack(self, prev_board: chess.Board, move: chess.Move) -> List[Dict]:
        curr_board = prev_board.copy()
        curr_board.push(move)
        moving_piece, moving_color = prev_board.piece_at(move.from_square), prev_board.color_at(move.from_square)
        discoveries = []

        for sq in prev_board.pieces(chess.ROOK, moving_color) | prev_board.pieces(chess.BISHOP, moving_color) | prev_board.pieces(chess.QUEEN, moving_color):
            if sq == move.from_square: continue
            newly_attacked = curr_board.attacks(sq) - prev_board.attacks(sq)
            for target_sq in newly_attacked:
                target = curr_board.piece_at(target_sq)
                if target and target.color != moving_color:
                    discoveries.append({
                        "moving_piece": moving_piece.symbol(), "revealed_attacker": prev_board.piece_at(sq).symbol(),
                        "revealed_attacker_square": chess.square_name(sq), "target": target.symbol(),
                        "target_square": chess.square_name(target_sq), "target_value": self._get_piece_value(target)
                    })
        return discoveries

    # --- TACTIC 5: CLEARANCE SACRIFICE (IMPROVED LOGIC) --------------------------
    
    def _is_true_sacrifice(self, board: chess.Board, move: chess.Move, moving_color: chess.Color) -> bool:
        """Checks if a move is a genuine material sacrifice, not just a trade."""
        to_sq = move.to_square
        opponent_color = not moving_color

        if not board.is_attacked_by(opponent_color, to_sq):
            return False

        opponent_attackers = board.attackers(opponent_color, to_sq)
        if not opponent_attackers: return False
        
        lowest_attacker_sq = min(opponent_attackers, key=lambda sq: self._get_piece_value(board.piece_at(sq)))
        
        temp_board = board.copy()
        try:
            # Check for promotion, otherwise it's a normal capture
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
        """
        Identifies and validates if a move is a clearance sacrifice.
        A clearance sacrifice opens a critical line or square for another piece.
        """
        curr_board = prev_board.copy()
        curr_board.push(move)
        moving_color = prev_board.color_at(move.from_square)

        if not self._is_true_sacrifice(curr_board, move, moving_color):
            return []

        clearance_details = self._get_clearance_details(prev_board, curr_board, move)
        if not clearance_details:
            return []

        try:
            # Use a slightly lower depth for this heuristic to keep it fast
            limit = chess.engine.Limit(time=0.3)
            info_before = self.engine.analyse(prev_board, limit)
            info_after = self.engine.analyse(curr_board, limit)
            
            eval_change = self._score_to_float(info_after['score'], moving_color) - \
                          self._score_to_float(info_before['score'], moving_color)

            # A sacrifice shouldn't result in a significantly worse position
            if eval_change < -1.5 and not info_after['score'].is_mate():
                return []

            return [{"evaluation_change": round(eval_change, 2), **clearance_details}]
        except (chess.engine.EngineError, IndexError):
            return []

    def _get_clearance_details(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> Optional[Dict]:
        """Determines if a sacrifice cleared a line or a square."""
        moving_color = prev_board.color_at(move.from_square)

        # Check for line-clearing
        for piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
            for sq in prev_board.pieces(piece_type, moving_color):
                if sq != move.from_square and (curr_board.attacks(sq) - prev_board.attacks(sq)):
                    return {"type": "Line-Clearing", "cleared_for_piece_on": chess.square_name(sq)}

        # Check for square-clearing (vacating)
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
                        return {
                            "type": "Square-Clearing (Vacating)",
                            "vacated_square": chess.square_name(vacated_sq),
                            "for_piece_on": chess.square_name(mover_sq)
                        }
        return None

    # --- STATIC EXCHANGE EVALUATION (SEE) ANALYSIS ---
    
    def _get_piece_identifier(self, board: chess.Board, square: int) -> str:
        """Creates a unique and readable string for a piece, e.g., 'white Queen at d1'."""
        piece = board.piece_at(square)
        if not piece: return ""
        color = "white" if piece.color == chess.WHITE else "black"
        piece_name = chess.piece_name(piece.piece_type).title()
        square_name = chess.square_name(square)
        return f"{color} {piece_name} at {square_name}"
    
    def _get_see_score(self, board: chess.Board, square: int, attacker_sq: int) -> int:
        """Performs Static Exchange Evaluation (SEE) for a capture on a given square."""
        attacker_piece = board.piece_at(attacker_sq)
        victim_piece = board.piece_at(square)
        if not attacker_piece or not victim_piece: return 0

        gain = [self._get_piece_value_cp(victim_piece)]
        sim_board = board.copy()
        sim_board.remove_piece_at(attacker_sq)
        side_to_move = not attacker_piece.color
        
        attackers = sim_board.attackers(chess.WHITE, square) | sim_board.attackers(chess.BLACK, square)
        all_attackers_sqs = sorted(list(attackers), key=lambda s: self._get_piece_value_cp(sim_board.piece_at(s)))

        current_capture_value = self._get_piece_value_cp(attacker_piece)

        while all_attackers_sqs:
            lva_sq = next((sq for sq in all_attackers_sqs if sim_board.piece_at(sq).color == side_to_move), None)
            if lva_sq is None: break
            
            all_attackers_sqs.remove(lva_sq)
            gain.append(current_capture_value)
            current_capture_value = self._get_piece_value_cp(sim_board.piece_at(lva_sq))
            side_to_move = not side_to_move

        score = 0
        for i, value in enumerate(gain):
            score += value if i % 2 == 0 else -value
        return score

    def _analyze_tactical_relationships(self, board: chess.Board) -> List[str]:
        """
        Generates a list of tactical relationships (checks, threats, pressure) based on SEE.
        This helps understand the tension and tactical possibilities in the position.
        """
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
                
                # Ensure we only process each pair of interacting pieces once
                canonical_pair = tuple(sorted((attacker_id, victim_id)))
                if canonical_pair in processed_pairs:
                    continue

                if victim_piece.piece_type == chess.KING:
                    relations.append(f"{attacker_id} CHECKS the {victim_id}")
                else:
                    relation = self._determine_see_relationship(board, attacker_sq, victim_sq)
                    relations.append(relation)
                
                processed_pairs.add(canonical_pair)

        return sorted(relations)

    def _determine_see_relationship(self, board: chess.Board, attacker_sq: int, victim_sq: int) -> str:
        """Determines the nature of a tactical relationship using SEE."""
        attacker_id = self._get_piece_identifier(board, attacker_sq)
        victim_id = self._get_piece_identifier(board, victim_sq)

        # Check if the attacker has a favorable capture
        see_attacker_wins = self._get_see_score(board, victim_sq, attacker_sq) > 0
        if see_attacker_wins:
            return f"{attacker_id} THREATENS the {victim_id}"

        # Check if the victim can profitably capture the attacker
        if board.is_attacked_by(board.color_at(victim_sq), attacker_sq):
            see_victim_wins = self._get_see_score(board, attacker_sq, victim_sq) > 0
            if see_victim_wins:
                return f"{victim_id} THREATENS the {attacker_id}"

        return f"{attacker_id} pressures the {victim_id}"

    # --- POSITIONAL FEATURE: PASSED PAWNS --------------------------------------

    def _detect_passed_pawns(self, board: chess.Board) -> Dict[str, List[Dict]]:
        results = {"white": [], "black": []}
        all_pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
        for square in all_pawns:
            piece = board.piece_at(square)
            color, file, rank = piece.color, chess.square_file(square), chess.square_rank(square)
            
            files_to_check = {file} | {f for f in [file - 1, file + 1] if 0 <= f <= 7}
            ranks_to_check = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
            
            has_opp_pawn = any(board.piece_type_at(chess.square(f, r)) == chess.PAWN for f in files_to_check for r in ranks_to_check)
            if not has_opp_pawn:
                color_key = "white" if color == chess.WHITE else "black"
                results[color_key].append({"square": chess.square_name(square), "is_protected": board.is_attacked_by(color, square)})
        return results
        
    # --- POSITIONAL FEATURE: BACK-RANK WEAKNESS --------------------------------

    def _detect_back_rank_weakness(self, board: chess.Board) -> List[Dict]:
        weaknesses = []
        for color in [chess.WHITE, chess.BLACK]:
            back_rank = 0 if color == chess.WHITE else 7
            king_square = board.king(color)
            if king_square is None or chess.square_rank(king_square) != back_rank: continue

            escape_squares = self._get_king_escape_squares(king_square, color)
            if any(not board.piece_at(sq) for sq in escape_squares): continue

            defenders = {s for f in range(8) for s in board.attackers(color, chess.square(f, back_rank)) if board.piece_type_at(s) != chess.KING}
            num_defenders = len(defenders)
            
            severity = "Potential"
            if num_defenders == 0: severity = "Critical"
            elif num_defenders == 1: severity = "Severe"
            elif num_defenders == 2: severity = "Moderate"
            weaknesses.append({
                "color": "White" if color == chess.WHITE else "Black", "severity": severity,
                "defenders": sorted([chess.square_name(sq) for sq in defenders]),
            })
        return weaknesses

    def _get_king_escape_squares(self, king_square: int, color: chess.Color) -> List[int]:
        file = chess.square_file(king_square)
        forward_rank = chess.square_rank(king_square) + (1 if color == chess.WHITE else -1)
        if not (0 <= forward_rank <= 7): return []
        return [chess.square(f, forward_rank) for f in [file-1, file, file+1] if 0 <= f <= 7]
        
    # --- POSITIONAL FEATURE: ISOLATED PAWNS ------------------------------------
        
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
            if not best_move: return "No legal moves."
            if best_move.to_square == blockade_sq: return f"Blockade with {temp_board.san(best_move)}"
            temp_board.push(best_move)
            if temp_board.is_attacked_by(not pawn_color, pawn_sq): return f"Attack with {board.san(best_move)}"
            return f"Other priorities (best move: {board.san(best_move)})"
        except Exception: return "Engine analysis failed."

    def _get_owner_tactic(self, board: chess.Board, pawn_sq: int) -> str:
        try:
            info = self.engine.analyse(board, chess.engine.Limit(time=0.2))
            best_move = info.get("pv", [None])[0]
            if not best_move: return "No legal moves."
            if best_move.from_square == pawn_sq: return f"Push the pawn with {board.san(best_move)}"
            return f"Other priorities (best move: {board.san(best_move)})"
        except Exception: return "Engine analysis failed."