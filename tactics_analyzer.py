"""
Enhanced Chess Tactics Analyzer for AI Commentary
Provides rich, contextual tactical and strategic analysis for chess positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import chess


class TacticalImpact(Enum):
    CRITICAL = "critical"      
    SIGNIFICANT = "significant" 
    MODERATE = "moderate"       
    MINOR = "minor"             


@dataclass
class TacticalPattern:
    pattern_type: str
    description: str
    impact: TacticalImpact
    pieces_involved: List[str]
    squares: List[str]
    material_gain: float = 0.0
    forcing_moves: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    narrative: str = ""


@dataclass
class KingSafetyAssessment:
    color: str
    safety_level: str
    pawn_shield_health: int
    open_lines: int
    attackers_nearby: int
    escape_squares: int
    description: str
    threats: List[str] = field(default_factory=list)


@dataclass
class PositionAssessment:
    material_balance: float
    king_safety: Dict[str, KingSafetyAssessment]
    piece_activity: Dict[str, str]
    pawn_structure: Dict[str, Any]
    weak_squares: Dict[str, List[str]]
    tactical_patterns: List[TacticalPattern]
    strategic_themes: List[str]
    threats: List[str]
    game_phase: str


class EnhancedTacticsAnalyzer:
    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.25,
        chess.BISHOP: 3.25,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.75,
        chess.KING: 0.0,
    }

    TACTICAL_NARRATIVES = {
        "fork": {
            "knight": ["A well-timed knight fork! {piece} attacks {targets} simultaneously, and {opponent} cannot save both."],
            "queen": ["The queen orchestrates a double attack on {targets}, exploiting the coordination weakness."],
            "default": ["A fork! {piece} attacks {targets} at the same time, forcing material loss."]
        },
        "pin": {
            "absolute": ["An absolute pin! The {piece} is pinned to the king on {square}, rendering it immobile."],
            "relative": ["A relative pin on {square}! The {piece} cannot move without losing the more valuable {target} behind it."],
        },
        "skewer": {
            "default": ["A skewer! The valuable {front_piece} on {front_square} must move, exposing {back_piece} on {back_square} to capture."]
        },
        "discovered_attack": {
            "default": ["A discovered attack! The {moving_piece} steps aside, unmasking a devastating attack from the {unmasked_piece}."]
        },
        "double_check": {
            "default": ["Double check! Both the {piece1} and {piece2} deliver check simultaneously. The king MUST move."]
        },
        "back_rank_mate": {
            "default": ["Back rank mate! The king is trapped by its own pawns, with no escape from the rook's wrath."]
        },
        "smothered_mate": {
            "default": ["A spectacular smothered mate! The knight delivers checkmate, and the king is suffocated by its own pieces."]
        },
        "deflection": {
            "default": ["A deflection tactic! The {piece} is forced away from its critical defensive duty on {square}."]
        },
        "decoy": {
            "default": ["A decoy sacrifice! The piece is offered to lure the enemy into a devastating tactical trap."]
        },
        "clearance": {
            "default": ["Line clearance! The {piece} brilliantly vacates the space to open a decisive attacking lane."]
        },
        "interference": {
            "default": ["Interference! The {piece} drops right into the defensive line, cutting off the opponent's coordination."]
        },
        "overloading": {
            "default": ["The {piece} on {square} is overloaded! It cannot fulfill multiple defensive duties at once."]
        },
        "underpromotion": {
            "default": ["A brilliant underpromotion! The pawn becomes a {piece} instead of a queen, perfectly suited for the position."]
        },
        "zugzwang": {
            "default": ["Zugzwang! Any move {side} makes will worsen their position. The luxury of having the move has become a curse."]
        },
    }

    def __init__(self, engine_path: Optional[str] = None):
        self.engine_path = engine_path
        self._shared_engine = None

    def _get_engine(self):
        if not self.engine_path: return None
        if not self._shared_engine:
            try:
                from stockfish import Stockfish
                self._shared_engine = Stockfish(path=self.engine_path, depth=12, parameters={"Threads": 1})
            except Exception:
                self._shared_engine = None
        return self._shared_engine

    def close(self) -> None:
        if self._shared_engine:
            try: del self._shared_engine
            except Exception: pass

    def __enter__(self) -> "EnhancedTacticsAnalyzer": return self
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: self.close()

    def _get_square_color(self, square: int) -> bool:
        return (chess.square_rank(square) + chess.square_file(square)) % 2 == 1

    # ==================== MAIN ANALYSIS METHODS ====================

    def analyze(self, current_fen: str, previous_fen: Optional[str] = None) -> Dict[str, Any]:
        board = chess.Board(current_fen)
        prev_board = chess.Board(previous_fen) if previous_fen else None
        move = self._find_move_between_boards(prev_board, board) if prev_board else None

        assessment = PositionAssessment(
            material_balance=self._calculate_material_balance(board),
            king_safety={
                "white": self._assess_king_safety(board, chess.WHITE),
                "black": self._assess_king_safety(board, chess.BLACK)
            },
            piece_activity=self._assess_piece_activity(board),
            pawn_structure=self._assess_pawn_structure(board),
            weak_squares=self._find_weak_squares(board),
            tactical_patterns=self._detect_all_tactics(board, prev_board, move),
            strategic_themes=self._identify_strategic_themes(board),
            threats=self._identify_threats(board),
            game_phase=self._determine_game_phase(board)
        )
        return self._build_result_dict(assessment, board, prev_board, move)

    # ==================== KING SAFETY ANALYSIS ====================

    def _assess_king_safety(self, board: chess.Board, color: chess.Color) -> KingSafetyAssessment:
        king_sq = board.king(color)
        if king_sq is None:
            return KingSafetyAssessment(color="white" if color else "black", safety_level="none", pawn_shield_health=0, open_lines=0, attackers_nearby=0, escape_squares=0, description="King not on board")

        enemy_color = not color
        pawn_shield = self._count_pawn_shield(board, king_sq, color)
        open_lines = self._count_open_lines_to_king(board, king_sq, enemy_color)
        attackers = self._count_king_zone_attackers(board, king_sq, enemy_color)
        escape_squares = self._count_king_escape_squares(board, king_sq, color)
        
        if attackers >= 3 and escape_squares <= 1:
            safety_level = "exposed"
            description = f"The king is dangerously exposed with {attackers} attackers swarming it."
        elif attackers >= 2 or open_lines >= 2 or (pawn_shield <= 1 and attackers >= 1):
            safety_level = "vulnerable"
            description = f"The king's position is shaky — {attackers} enemy pieces lurk nearby."
        elif pawn_shield >= 2 and attackers == 0 and open_lines == 0:
            safety_level = "secure"
            description = "The king sits securely behind a healthy pawn shield."
        else:
            safety_level = "adequate"
            description = "The king is reasonably safe for now."

        threats = []
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate(): threats.append(f"Checkmate threat: {board.san(move)}")
            board.pop()

        return KingSafetyAssessment(
            color="white" if color else "black",
            safety_level=safety_level,
            pawn_shield_health=pawn_shield,
            open_lines=open_lines,
            attackers_nearby=attackers,
            escape_squares=escape_squares,
            description=description,
            threats=threats[:3]
        )

    def _count_pawn_shield(self, board: chess.Board, king_sq: int, color: chess.Color) -> int:
        shield = 0
        king_file, king_rank = chess.square_file(king_sq), chess.square_rank(king_sq)
        shield_rank = king_rank + 1 if color == chess.WHITE else king_rank - 1
        if 0 <= shield_rank < 8:
            for file_offset in [-1, 0, 1]:
                file = king_file + file_offset
                if 0 <= file < 8:
                    p = board.piece_at(chess.square(file, shield_rank))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        shield += 1
        return shield

    def _count_open_lines_to_king(self, board: chess.Board, king_sq: int, attacker_color: chess.Color) -> int:
        lines = 0
        for sq in board.pieces(chess.ROOK, attacker_color) | board.pieces(chess.QUEEN, attacker_color):
            if self._has_clear_path(board, sq, king_sq) or self._has_clear_path_on_file(board, sq, king_sq): lines += 1
        for sq in board.pieces(chess.BISHOP, attacker_color) | board.pieces(chess.QUEEN, attacker_color):
            if self._is_on_diagonal(sq, king_sq) and self._has_clear_path(board, sq, king_sq): lines += 1
        return lines

    def _count_king_zone_attackers(self, board: chess.Board, king_sq: int, attacker_color: chess.Color) -> int:
        king_file, king_rank = chess.square_file(king_sq), chess.square_rank(king_sq)
        attackers = set()
        for df in range(-2, 3):
            for dr in range(-2, 3):
                f, r = king_file + df, king_rank + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    for a in board.attackers(attacker_color, chess.square(f, r)):
                        attackers.add(a)
        return len(attackers)

    def _count_king_escape_squares(self, board: chess.Board, king_sq: int, color: chess.Color) -> int:
        escapes = 0
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if df == 0 and dr == 0: continue
                f, r = kf + df, kr + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if (not p or p.color != color) and not board.is_attacked_by(not color, sq):
                        escapes += 1
        return escapes

    # ==================== TACTICAL DETECTION ====================

    def _detect_all_tactics(self, board: chess.Board, prev_board: Optional[chess.Board],
                           move: Optional[chess.Move]) -> List[TacticalPattern]:
        patterns = []

        if board.is_checkmate():
            smothered = self._detect_smothered_mate(board)
            if smothered: patterns.append(smothered)
            else: patterns.append(self._create_checkmate_pattern(board))

        if board.is_check() and not board.is_checkmate():
            patterns.extend(self._detect_check_patterns(board))
            patterns.extend(self._detect_double_checks(board))

        patterns.extend(self._detect_forks_enhanced(board))
        patterns.extend(self._detect_pins_enhanced(board))
        patterns.extend(self._detect_skewers_enhanced(board))
        patterns.extend(self._detect_back_rank_threats(board))
        patterns.extend(self._detect_overloading(board))

        if prev_board and move:
            patterns.extend(self._detect_discovered_attacks_enhanced(board, prev_board, move))
            patterns.extend(self._detect_underpromotion(move))
            patterns.extend(self._detect_interference(board, prev_board, move))
            patterns.extend(self._detect_clearance(board, prev_board, move))
            patterns.extend(self._detect_deflection(board, prev_board, move))
            patterns.extend(self._detect_decoy(board, prev_board, move))

        if self._determine_game_phase(board) == "endgame":
            patterns.extend(self._detect_zugzwang(board))

        unique_patterns = self._deduplicate_patterns(patterns)
        unique_patterns.sort(key=lambda p: {
            TacticalImpact.CRITICAL: 0, TacticalImpact.SIGNIFICANT: 1,
            TacticalImpact.MODERATE: 2, TacticalImpact.MINOR: 3
        }.get(p.impact, 4))

        return unique_patterns[:5]

    def _detect_smothered_mate(self, board: chess.Board) -> Optional[TacticalPattern]:
        checkers = list(board.checkers())
        if len(checkers) != 1: return None
        checker_sq = checkers[0]
        if board.piece_at(checker_sq).piece_type != chess.KNIGHT: return None

        king_sq = board.king(board.turn)
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if df == 0 and dr == 0: continue
                f, r = kf + df, kr + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    p = board.piece_at(chess.square(f, r))
                    if not p or p.color != board.turn:
                        return None
        
        return TacticalPattern(
            pattern_type="smothered_mate",
            description="Smothered Mate!",
            impact=TacticalImpact.CRITICAL,
            pieces_involved=["knight", "king"],
            squares=[chess.square_name(checker_sq), chess.square_name(king_sq)],
            narrative=self.TACTICAL_NARRATIVES["smothered_mate"]["default"][0],
            consequences=["Game over"]
        )

    def _detect_underpromotion(self, move: chess.Move) -> List[TacticalPattern]:
        if move and move.promotion and move.promotion != chess.QUEEN:
            piece_name = chess.piece_name(move.promotion)
            return [TacticalPattern("underpromotion", f"Underpromotion to {piece_name}", TacticalImpact.SIGNIFICANT, [piece_name], [chess.square_name(move.to_square)], narrative=self.TACTICAL_NARRATIVES["underpromotion"]["default"][0].format(piece=piece_name))]
        return []

    def _detect_interference(self, board: chess.Board, prev_board: chess.Board, move: chess.Move) -> List[TacticalPattern]:
        patterns = []
        enemy_color = prev_board.turn
        to_sq = move.to_square
        
        for enemy_sq in list(prev_board.pieces(chess.ROOK, enemy_color)) + list(prev_board.pieces(chess.BISHOP, enemy_color)) + list(prev_board.pieces(chess.QUEEN, enemy_color)):
            if enemy_sq == to_sq: continue
            lost_squares = prev_board.attacks(enemy_sq) - board.attacks(enemy_sq)
            for sq in lost_squares:
                target_piece = prev_board.piece_at(sq)
                if target_piece and target_piece.color == enemy_color:
                    if self._is_on_same_line(enemy_sq, sq, to_sq):
                        piece_name = chess.piece_name(board.piece_at(to_sq).piece_type)
                        patterns.append(TacticalPattern("interference", "Interference", TacticalImpact.SIGNIFICANT, [piece_name], [chess.square_name(to_sq)], narrative=self.TACTICAL_NARRATIVES["interference"]["default"][0].format(piece=piece_name), consequences=["Cuts off defensive coordination"]))
                        return patterns
        return patterns

    def _detect_overloading(self, board: chess.Board) -> List[TacticalPattern]:
        patterns = []
        enemy_color = board.turn
        defender_tasks = {}

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == enemy_color:
                if list(board.attackers(not enemy_color, sq)):
                    for d_sq in list(board.attackers(enemy_color, sq)):
                        d_piece = board.piece_at(d_sq)
                        if d_piece and d_piece.piece_type != chess.KING:
                            if d_sq not in defender_tasks: defender_tasks[d_sq] = []
                            defender_tasks[d_sq].append(sq)

        for d_sq, tasks in defender_tasks.items():
            if len(tasks) >= 2:
                piece_name = chess.piece_name(board.piece_at(d_sq).piece_type)
                patterns.append(TacticalPattern("overloading", "Overloaded piece", TacticalImpact.MODERATE, [piece_name], [chess.square_name(d_sq)], narrative=self.TACTICAL_NARRATIVES["overloading"]["default"][0].format(piece=piece_name, square=chess.square_name(d_sq))))
        return patterns

    def _detect_deflection(self, board: chess.Board, prev_board: chess.Board, move: chess.Move) -> List[TacticalPattern]:
        patterns = []
        if prev_board.is_capture(move):
            captured_piece = prev_board.piece_at(move.to_square)
            if captured_piece:
                for def_sq in prev_board.attacks(move.to_square):
                    def_piece = prev_board.piece_at(def_sq)
                    if def_piece and def_piece.color == captured_piece.color:
                        if len(board.attackers(not def_piece.color, def_sq)) > len(board.attackers(def_piece.color, def_sq)):
                            patterns.append(TacticalPattern("deflection", "Deflection via capture", TacticalImpact.SIGNIFICANT, [chess.piece_name(captured_piece.piece_type)], [chess.square_name(move.to_square)], narrative=self.TACTICAL_NARRATIVES["deflection"]["default"][0].format(piece=chess.piece_name(captured_piece.piece_type), square=chess.square_name(move.to_square))))
                            return patterns
        return patterns

    def _detect_decoy(self, board: chess.Board, prev_board: chess.Board, move: chess.Move) -> List[TacticalPattern]:
        patterns = []
        mover_color = prev_board.turn
        if not board.is_attacked_by(not mover_color, move.to_square): return patterns
        
        for capture_move in board.legal_moves:
            if capture_move.to_square == move.to_square and board.is_capture(capture_move):
                sim = board.copy(stack=False)
                sim.push(capture_move)
                for f in self._detect_forks_enhanced(sim):
                    if chess.square_name(move.to_square) in f.squares:
                        patterns.append(TacticalPattern("decoy", "Decoy Sacrifice", TacticalImpact.CRITICAL, ["piece"], [chess.square_name(move.to_square)], narrative=self.TACTICAL_NARRATIVES["decoy"]["default"][0]))
                        return patterns
        return patterns

    def _detect_clearance(self, board: chess.Board, prev_board: chess.Board, move: chess.Move) -> List[TacticalPattern]:
        patterns = []
        mover_color = prev_board.turn
        for sq in list(prev_board.pieces(chess.ROOK, mover_color)) + list(prev_board.pieces(chess.BISHOP, mover_color)) + list(prev_board.pieces(chess.QUEEN, mover_color)):
            if sq == move.from_square: continue
            new_attacks = board.attacks(sq) - prev_board.attacks(sq)
            for target_sq in new_attacks:
                target_piece = board.piece_at(target_sq)
                if target_piece and target_piece.color != mover_color and target_piece.piece_type != chess.KING:
                    piece_name = chess.piece_name(prev_board.piece_at(move.from_square).piece_type)
                    patterns.append(TacticalPattern("clearance", "Line Clearance", TacticalImpact.SIGNIFICANT, [piece_name], [chess.square_name(move.from_square)], narrative=self.TACTICAL_NARRATIVES["clearance"]["default"][0].format(piece=piece_name)))
                    return patterns
        return patterns

    def _detect_zugzwang(self, board: chess.Board) -> List[TacticalPattern]:
        engine = self._get_engine()
        if not engine: return []
        try:
            engine.set_fen_position(board.fen())
            eval_real = self._cp_from_eval(engine.get_evaluation())
            null_board = board.copy()
            null_board.push(chess.Move.null())
            engine.set_fen_position(null_board.fen())
            eval_null = self._cp_from_eval(engine.get_evaluation())
            
            threshold = 150
            if (board.turn == chess.WHITE and eval_null - eval_real > threshold) or (board.turn == chess.BLACK and eval_real - eval_null > threshold):
                return [TacticalPattern("zugzwang", "Zugzwang", TacticalImpact.CRITICAL, [], [], narrative=self.TACTICAL_NARRATIVES["zugzwang"]["default"][0].format(side="White" if board.turn else "Black"))]
        except Exception: pass
        return []

    def _create_checkmate_pattern(self, board: chess.Board) -> TacticalPattern:
        mated_color = board.turn
        king_sq = board.king(mated_color)
        checkers = list(board.checkers())

        if len(checkers) == 1:
            checker = board.piece_at(checkers[0])
            narrative = f"Checkmate! The {chess.piece_name(checker.piece_type)} on {chess.square_name(checkers[0])} delivers the final blow."
        else:
            narrative = "A stunning checkmate ends the game!"

        return TacticalPattern("checkmate", "Checkmate", TacticalImpact.CRITICAL, [chess.square_name(king_sq)] if king_sq else [], [chess.square_name(king_sq)] if king_sq else [], narrative=narrative, consequences=["Game over"])

    def _detect_check_patterns(self, board: chess.Board) -> List[TacticalPattern]:
        patterns = []
        king_sq = board.king(board.turn)
        if king_sq is None: return patterns

        for checker_sq in list(board.checkers()):
            checker = board.piece_at(checker_sq)
            if not checker: continue

            escapes = self._count_king_escape_squares(board, king_sq, board.turn)
            if escapes == 0: impact = TacticalImpact.CRITICAL; narrative = f"A crushing check from the {chess.piece_name(checker.piece_type)}! The king is out of squares."
            elif escapes == 1: impact = TacticalImpact.SIGNIFICANT; narrative = f"Powerful check! The king has only one escape."
            else: impact = TacticalImpact.MODERATE; narrative = f"Check from the {chess.piece_name(checker.piece_type)}!"

            patterns.append(TacticalPattern("check", f"Check by {chess.piece_name(checker.piece_type)}", impact, [chess.piece_name(checker.piece_type), "king"], [chess.square_name(checker_sq), chess.square_name(king_sq)], narrative=narrative, consequences=["King must move"]))
        return patterns

    def _detect_forks_enhanced(self, board: chess.Board) -> List[TacticalPattern]:
        patterns = []
        for sq, piece in board.piece_map().items():
            if piece.piece_type == chess.KING: continue
            targets = []
            for t_sq in board.attacks(sq):
                t_piece = board.piece_at(t_sq)
                if t_piece and t_piece.color != piece.color:
                    targets.append({"square": t_sq, "piece": t_piece, "value": self.PIECE_VALUES.get(t_piece.piece_type, 0)})

            if len(targets) >= 2:
                targets.sort(key=lambda t: t["value"], reverse=True)
                top = targets[:2]
                min_val = min(t["value"] for t in top)
                impact = TacticalImpact.CRITICAL if min_val >= 5 else TacticalImpact.SIGNIFICANT if min_val >= 3 else TacticalImpact.MODERATE
                
                piece_name = chess.piece_name(piece.piece_type)
                t_names = [f"{chess.piece_name(t['piece'].piece_type)} on {chess.square_name(t['square'])}" for t in top]
                tmpl = self.TACTICAL_NARRATIVES["fork"].get(piece_name, self.TACTICAL_NARRATIVES["fork"]["default"])
                
                patterns.append(TacticalPattern("fork", f"{piece_name.title()} fork", impact, [piece_name] + [chess.piece_name(t["piece"].piece_type) for t in top], [chess.square_name(sq)] + [chess.square_name(t["square"]) for t in top], narrative=tmpl[0].format(piece=piece_name.title(), targets=" and ".join(t_names), opponent="Black" if piece.color == chess.WHITE else "White"), consequences=["Wins material"]))
        return patterns

    def _detect_pins_enhanced(self, board: chess.Board) -> List[TacticalPattern]:
        patterns = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for sq, piece in board.piece_map().items():
            if piece.piece_type not in [chess.ROOK, chess.BISHOP, chess.QUEEN]: continue
            for dr, df in directions:
                if piece.piece_type == chess.ROOK and dr != 0 and df != 0: continue
                if piece.piece_type == chess.BISHOP and (dr == 0 or df == 0): continue
                
                pinned, target, current_sq = None, None, sq
                for _ in range(1, 8):
                    nf, nr = chess.square_file(current_sq) + df, chess.square_rank(current_sq) + dr
                    if not (0 <= nf < 8 and 0 <= nr < 8): break
                    current_sq = chess.square(nf, nr)
                    p = board.piece_at(current_sq)
                    if p:
                        if p.color == piece.color: break
                        elif not pinned: pinned = (current_sq, p)
                        elif not target: target = (current_sq, p); break
                        else: break
                
                if pinned and target:
                    is_abs = target[1].piece_type == chess.KING
                    narrative = self.TACTICAL_NARRATIVES["pin"]["absolute" if is_abs else "relative"][0].format(piece=chess.piece_name(pinned[1].piece_type), square=chess.square_name(pinned[0]), target=chess.piece_name(target[1].piece_type))
                    patterns.append(TacticalPattern("pin", "Pin", TacticalImpact.CRITICAL if is_abs else TacticalImpact.SIGNIFICANT, [chess.piece_name(piece.piece_type), chess.piece_name(pinned[1].piece_type)], [chess.square_name(sq), chess.square_name(pinned[0])], narrative=narrative, consequences=["Piece is immobilized"]))
        return patterns

    def _detect_skewers_enhanced(self, board: chess.Board) -> List[TacticalPattern]:
        patterns = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for sq, piece in board.piece_map().items():
            if piece.piece_type not in [chess.ROOK, chess.BISHOP, chess.QUEEN]: continue
            for dr, df in directions:
                if piece.piece_type == chess.ROOK and dr != 0 and df != 0: continue
                if piece.piece_type == chess.BISHOP and (dr == 0 or df == 0): continue
                
                front, back, current_sq = None, None, sq
                for _ in range(1, 8):
                    nf, nr = chess.square_file(current_sq) + df, chess.square_rank(current_sq) + dr
                    if not (0 <= nf < 8 and 0 <= nr < 8): break
                    current_sq = chess.square(nf, nr)
                    p = board.piece_at(current_sq)
                    if p:
                        if p.color == piece.color: break
                        elif not front: front = (current_sq, p)
                        elif not back: back = (current_sq, p); break
                        else: break
                        
                if front and back and front[1].color == back[1].color:
                    fv, bv = self.PIECE_VALUES.get(front[1].piece_type, 0), self.PIECE_VALUES.get(back[1].piece_type, 0)
                    if fv > bv:
                        narrative = self.TACTICAL_NARRATIVES["skewer"]["default"][0].format(front_piece=chess.piece_name(front[1].piece_type), front_square=chess.square_name(front[0]), back_piece=chess.piece_name(back[1].piece_type), back_square=chess.square_name(back[0]))
                        patterns.append(TacticalPattern("skewer", "Skewer", TacticalImpact.SIGNIFICANT if fv >= 5 else TacticalImpact.MODERATE, [chess.piece_name(piece.piece_type)], [chess.square_name(sq), chess.square_name(front[0]), chess.square_name(back[0])], narrative=narrative, consequences=["Wins back piece"]))
        return patterns

    def _detect_discovered_attacks_enhanced(self, board: chess.Board, prev_board: chess.Board, move: chess.Move) -> List[TacticalPattern]:
        patterns = []
        if not board.is_check(): return patterns
        king_sq = board.king(board.turn)
        if king_sq is None: return patterns
        
        for checker_sq in board.attackers(prev_board.turn, king_sq):
            checker_piece = board.piece_at(checker_sq)
            if not checker_piece or checker_piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]: continue
            if checker_sq == move.to_square: continue
            
            between = self._squares_between(checker_sq, king_sq)
            if move.from_square in between:
                moving_piece = prev_board.piece_at(move.from_square)
                narrative = self.TACTICAL_NARRATIVES["discovered_attack"]["default"][0].format(moving_piece=chess.piece_name(moving_piece.piece_type) if moving_piece else "piece", unmasked_piece=chess.piece_name(checker_piece.piece_type))
                patterns.append(TacticalPattern("discovered_attack", "Discovered Attack", TacticalImpact.SIGNIFICANT, [chess.piece_name(checker_piece.piece_type)], [chess.square_name(checker_sq), chess.square_name(move.to_square)], narrative=narrative, consequences=["King must move"]))
        return patterns

    def _detect_double_checks(self, board: chess.Board) -> List[TacticalPattern]:
        patterns = []
        checkers = list(board.checkers())
        if len(checkers) >= 2:
            pieces = [chess.piece_name(board.piece_at(c).piece_type) for c in checkers]
            narrative = self.TACTICAL_NARRATIVES["double_check"]["default"][0].format(piece1=pieces[0], piece2=pieces[1])
            patterns.append(TacticalPattern("double_check", "Double Check", TacticalImpact.CRITICAL, pieces, [chess.square_name(c) for c in checkers], narrative=narrative, consequences=["King MUST move"]))
        return patterns

    def _detect_back_rank_threats(self, board: chess.Board) -> List[TacticalPattern]:
        patterns = []
        if board.is_checkmate(): return patterns
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if not king_sq or chess.square_rank(king_sq) != (0 if color == chess.WHITE else 7): continue
            
            f_rank = 1 if color == chess.WHITE else 6
            kf = chess.square_file(king_sq)
            blocked = True
            for f in [kf - 1, kf, kf + 1]:
                if 0 <= f < 8:
                    p = board.piece_at(chess.square(f, f_rank))
                    if not p or p.color != color: blocked = False; break
            
            if blocked:
                for enemy_sq in list(board.pieces(chess.ROOK, not color)) + list(board.pieces(chess.QUEEN, not color)):
                    if chess.square_file(enemy_sq) == kf and self._has_clear_path_on_file(board, enemy_sq, king_sq):
                        patterns.append(TacticalPattern("back_rank_threat", "Back Rank Threat", TacticalImpact.SIGNIFICANT, ["rook/queen", "king"], [chess.square_name(enemy_sq), chess.square_name(king_sq)], narrative=self.TACTICAL_NARRATIVES["back_rank_mate"]["default"][0], consequences=["Potential back rank mate"]))
        return patterns

    # ==================== STRATEGIC ANALYSIS ====================

    def _identify_strategic_themes(self, board: chess.Board) -> List[str]:
        themes = []
        game_phase = self._determine_game_phase(board)

        # Space Advantage
        white_space = sum(1 for sq in range(32, 64) if board.is_attacked_by(chess.WHITE, sq))
        black_space = sum(1 for sq in range(0, 32) if board.is_attacked_by(chess.BLACK, sq))
        if white_space > black_space + 12: themes.append("White enjoys a crushing space advantage, cramping Black's pieces")
        elif black_space > white_space + 12: themes.append("Black enjoys a crushing space advantage, cramping White's pieces")

        # Opposite Castling
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk and bk:
            wk_file, bk_file = chess.square_file(wk), chess.square_file(bk)
            if (wk_file <= 2 and bk_file >= 5) or (wk_file >= 5 and bk_file <= 2):
                themes.append("Opposite-side castling creates a dangerous race of pawn storms")

        # Endgame King Activity
        if game_phase == "endgame":
            if wk and bk:
                wk_score = self._centralization_score(wk)
                bk_score = self._centralization_score(bk)
                if wk_score > bk_score + 1.5: themes.append("White's king is highly active in the center")
                elif bk_score > wk_score + 1.5: themes.append("Black's king is highly active in the center")

        # Pawn structure & Bad Bishop themes
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color else "Black"
            passed = self._find_passed_pawns(board, color)
            if passed: themes.append(f"{color_name} has a dangerous passed pawn on {chess.square_name(passed[0])}")
            
            isolated = self._find_isolated_pawns(board, color)
            if isolated: themes.append(f"{color_name} is burdened with an isolated pawn on {chess.square_name(isolated[0])}")
            
            # Corrected Bad Bishop Logic: Only applies out of the opening and if genuinely blocked
            if game_phase != "opening":
                for sq in board.pieces(chess.BISHOP, color):
                    b_color = self._get_square_color(sq)
                    start_rank = 1 if color == chess.WHITE else 6
                    blocked_pawns = 0
                    for pawn_sq in board.pieces(chess.PAWN, color):
                        if self._get_square_color(pawn_sq) == b_color and chess.square_rank(pawn_sq) != start_rank:
                            blocked_pawns += 1
                    
                    if blocked_pawns >= 3 and not self._bishop_has_open_diagonals(board, sq, color):
                        themes.append(f"{color_name} suffers from a structurally 'Bad Bishop' blocked by its own pawns")
                        break # Only report once per side

        return themes[:5]

    def _assess_piece_activity(self, board: chess.Board) -> Dict[str, str]:
        activity = {}
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "white" if color else "black"
            active_pieces = sum(1 for sq in board.pieces(chess.KNIGHT, color) if self._is_good_knight_outpost(board, sq, color)) + \
                            sum(1 for sq in board.pieces(chess.BISHOP, color) if self._bishop_has_open_diagonals(board, sq, color)) + \
                            sum(1 for sq in board.pieces(chess.ROOK, color) if self._rook_on_open_file(board, sq, color))
            
            if active_pieces >= 3: activity[color_name] = "Pieces are highly active and well-coordinated."
            elif active_pieces >= 1: activity[color_name] = "Moderate piece activity. Some pieces need better squares."
            else: activity[color_name] = "Pieces are passive and need activation."
        return activity

    def _find_weak_squares(self, board: chess.Board) -> Dict[str, List[str]]:
        weak = {"white": [], "black": []}
        for color in [chess.WHITE, chess.BLACK]:
            enemy_color = not color
            c_name = "white" if color else "black"
            for sq in chess.SQUARES:
                file, rank = chess.square_file(sq), chess.square_rank(sq)
                can_defend = False
                for pf in [file - 1, file + 1]:
                    if 0 <= pf < 8:
                        pr = rank - 1 if color == chess.WHITE else rank + 1
                        if 0 <= pr < 8:
                            p = board.piece_at(chess.square(pf, pr))
                            if p and p.piece_type == chess.PAWN and p.color == color: can_defend = True; break
                if not can_defend and board.is_attacked_by(enemy_color, sq) and not board.piece_at(sq):
                    weak[c_name].append(chess.square_name(sq))
            weak[c_name] = weak[c_name][:3]
        return weak

    def _identify_threats(self, board: chess.Board) -> List[str]:
        threats = []
        for sq, piece in board.piece_map().items():
            if piece.piece_type in [chess.KING, chess.PAWN]: continue
            attackers = list(board.attackers(not piece.color, sq))
            defenders = list(board.attackers(piece.color, sq))
            if attackers and len(attackers) > len(defenders):
                threats.append(f"{chess.piece_name(piece.piece_type)} on {chess.square_name(sq)} is hanging")
        return threats[:3]

    def _assess_pawn_structure(self, board: chess.Board) -> Dict[str, Any]:
        structure = {"white": {}, "black": {}}
        for color in [chess.WHITE, chess.BLACK]:
            c_name = "white" if color else "black"
            structure[c_name] = {
                "count": len(board.pieces(chess.PAWN, color)),
                "passed": [chess.square_name(sq) for sq in self._find_passed_pawns(board, color)],
                "isolated": [chess.square_name(sq) for sq in self._find_isolated_pawns(board, color)],
                "doubled": self._count_doubled_pawns(board, color),
                "backward": [chess.square_name(sq) for sq in self._find_backward_pawns(board, color)],
            }
        return structure

    # ==================== HELPER METHODS ====================

    def _cp_from_eval(self, eval_dict: Optional[Dict[str, Any]]) -> int:
        if not eval_dict: return 0
        if eval_dict.get("type") == "cp": return int(eval_dict.get("value", 0))
        if eval_dict.get("type") == "mate": return 10000 if int(eval_dict.get("value", 0)) > 0 else -10000
        return 0

    def _find_move_between_boards(self, prev_board: chess.Board, curr_board: chess.Board) -> Optional[chess.Move]:
        for move in prev_board.legal_moves:
            probe = prev_board.copy(stack=False)
            probe.push(move)
            if probe.board_fen() == curr_board.board_fen() and probe.turn == curr_board.turn: return move
        return None

    def _calculate_material_balance(self, board: chess.Board) -> float:
        return sum(self.PIECE_VALUES.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1) for p in board.piece_map().values())

    def _determine_game_phase(self, board: chess.Board) -> str:
        q = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        m = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
        if q >= 2 and m >= 4: return "opening"
        elif q >= 1 or m >= 2: return "middlegame"
        return "endgame"

    def _has_clear_path(self, board: chess.Board, from_sq: int, to_sq: int) -> bool:
        dir = self._get_direction(from_sq, to_sq)
        if not dir: return False
        curr = from_sq
        while True:
            cf, cr = chess.square_file(curr) + dir[0], chess.square_rank(curr) + dir[1]
            if not (0 <= cf < 8 and 0 <= cr < 8): return False
            curr = chess.square(cf, cr)
            if curr == to_sq: return True
            if board.piece_at(curr): return False

    def _has_clear_path_on_file(self, board: chess.Board, from_sq: int, to_sq: int) -> bool:
        if chess.square_file(from_sq) != chess.square_file(to_sq): return False
        fr, tr = chess.square_rank(from_sq), chess.square_rank(to_sq)
        step = 1 if tr > fr else -1
        for r in range(fr + step, tr, step):
            if board.piece_at(chess.square(chess.square_file(from_sq), r)): return False
        return True

    def _get_direction(self, from_sq: int, to_sq: int) -> Optional[Tuple[int, int]]:
        df, dr = chess.square_file(to_sq) - chess.square_file(from_sq), chess.square_rank(to_sq) - chess.square_rank(from_sq)
        if df != 0: df = df // abs(df)
        if dr != 0: dr = dr // abs(dr)
        if df == 0 or dr == 0 or abs(chess.square_file(to_sq) - chess.square_file(from_sq)) == abs(chess.square_rank(to_sq) - chess.square_rank(from_sq)):
            return (df, dr)
        return None

    def _is_on_diagonal(self, sq1: int, sq2: int) -> bool:
        return abs(chess.square_file(sq1) - chess.square_file(sq2)) == abs(chess.square_rank(sq1) - chess.square_rank(sq2))

    def _is_on_same_line(self, sq1: int, sq2: int, sq3: int) -> bool:
        dir1 = self._get_direction(sq1, sq3)
        if not dir1: return False
        dir2 = self._get_direction(sq1, sq2)
        if dir1 != dir2: return False
        dist1 = max(abs(chess.square_file(sq1) - chess.square_file(sq3)), abs(chess.square_rank(sq1) - chess.square_rank(sq3)))
        dist2 = max(abs(chess.square_file(sq1) - chess.square_file(sq2)), abs(chess.square_rank(sq1) - chess.square_rank(sq2)))
        return dist2 < dist1

    def _squares_between(self, from_sq: int, to_sq: int) -> List[int]:
        dir = self._get_direction(from_sq, to_sq)
        if not dir: return []
        sqs, curr = [], from_sq
        while True:
            cf, cr = chess.square_file(curr) + dir[0], chess.square_rank(curr) + dir[1]
            if not (0 <= cf < 8 and 0 <= cr < 8): break
            curr = chess.square(cf, cr)
            if curr == to_sq: break
            sqs.append(curr)
        return sqs

    def _is_good_knight_outpost(self, board: chess.Board, sq: int, color: chess.Color) -> bool:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if color == chess.WHITE and (r < 3 or r > 6): return False
        if color == chess.BLACK and (r < 2 or r > 4): return False
        supported = any(board.piece_at(chess.square(pf, r - 1 if color else r + 1)) and board.piece_at(chess.square(pf, r - 1 if color else r + 1)).color == color for pf in [f-1, f+1] if 0 <= pf < 8)
        enemy_pawns = any(board.piece_at(chess.square(pf, r - 1 if not color else r + 1)) and board.piece_at(chess.square(pf, r - 1 if not color else r + 1)).color != color for pf in [f-1, f+1] if 0 <= pf < 8)
        return supported and not enemy_pawns

    def _bishop_has_open_diagonals(self, board: chess.Board, sq: int, color: chess.Color) -> bool:
        opens = 0
        for df, dr in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            cf, cr = chess.square_file(sq) + df, chess.square_rank(sq) + dr
            while 0 <= cf < 8 and 0 <= cr < 8:
                p = board.piece_at(chess.square(cf, cr))
                if p:
                    if p.color != color: opens += 1
                    break
                opens += 0.5
                cf += df; cr += dr
        return opens >= 2

    def _rook_on_open_file(self, board: chess.Board, sq: int, color: chess.Color) -> bool:
        f = chess.square_file(sq)
        return not any(board.piece_at(chess.square(f, r)) and board.piece_at(chess.square(f, r)).piece_type == chess.PAWN for r in range(8))

    def _find_passed_pawns(self, board: chess.Board, color: chess.Color) -> List[int]:
        passed = []
        for sq in board.pieces(chess.PAWN, color):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            ranks = range(r+1, 8) if color else range(0, r)
            if not any(board.piece_at(chess.square(ef, er)) and board.piece_at(chess.square(ef, er)).color != color and board.piece_at(chess.square(ef, er)).piece_type == chess.PAWN for ef in [f-1, f, f+1] if 0 <= ef < 8 for er in ranks):
                passed.append(sq)
        return passed

    def _find_isolated_pawns(self, board: chess.Board, color: chess.Color) -> List[int]:
        isolated = []
        pawns = set(chess.square_file(sq) for sq in board.pieces(chess.PAWN, color))
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            if (f - 1 not in pawns) and (f + 1 not in pawns): isolated.append(sq)
        return isolated

    def _find_backward_pawns(self, board: chess.Board, color: chess.Color) -> List[int]:
        backward = []
        for sq in board.pieces(chess.PAWN, color):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            prot_r = r - 1 if color else r + 1
            if 0 <= prot_r < 8 and any(board.piece_at(chess.square(pf, prot_r)) and board.piece_at(chess.square(pf, prot_r)).color == color for pf in [f-1, f+1] if 0 <= pf < 8): continue
            att_r = r + 1 if color else r - 1
            if 0 <= att_r < 8 and any(board.piece_at(chess.square(pf, att_r)) and board.piece_at(chess.square(pf, att_r)).color != color for pf in [f-1, f+1] if 0 <= pf < 8):
                backward.append(sq)
        return backward

    def _count_doubled_pawns(self, board: chess.Board, color: chess.Color) -> int:
        files = [0]*8
        for sq in board.pieces(chess.PAWN, color): files[chess.square_file(sq)] += 1
        return sum(f - 1 for f in files if f > 1)

    def _centralization_score(self, sq: int) -> float:
        return 5 - ((chess.square_file(sq) - 3.5)**2 + (chess.square_rank(sq) - 3.5)**2)**0.5

    def _deduplicate_patterns(self, patterns: List[TacticalPattern]) -> List[TacticalPattern]:
        seen, unique = set(), []
        for p in patterns:
            k = (p.pattern_type, tuple(sorted(p.squares)))
            if k not in seen: seen.add(k); unique.append(p)
        return unique

    def _build_result_dict(self, assessment: PositionAssessment, board: chess.Board, prev_board: Optional[chess.Board], move: Optional[chess.Move]) -> Dict[str, Any]:
        return {
            "fen": board.fen(),
            "side_to_move": "white" if board.turn else "black",
            "game_phase": assessment.game_phase,
            "material_balance": assessment.material_balance,
            "king_safety": {
                "white": {"level": assessment.king_safety["white"].safety_level, "description": assessment.king_safety["white"].description, "threats": assessment.king_safety["white"].threats},
                "black": {"level": assessment.king_safety["black"].safety_level, "description": assessment.king_safety["black"].description, "threats": assessment.king_safety["black"].threats}
            },
            "piece_activity": assessment.piece_activity,
            "pawn_structure": assessment.pawn_structure,
            "weak_squares": assessment.weak_squares,
            "tactical_patterns": [{"type": t.pattern_type, "description": t.description, "impact": t.impact.value, "pieces": t.pieces_involved, "squares": t.squares, "narrative": t.narrative} for t in assessment.tactical_patterns],
            "strategic_themes": assessment.strategic_themes,
            "threats": assessment.threats,
        }

    # ==================== ATTACK, PRESSURE, CAPTURES & BRILLIANT ====================

    def detect_attack(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> Dict[str, Any]:
        if not move: return {"is_attacking": False}
        attacker_color = prev_board.turn
        attacks = []
        
        for t_sq in curr_board.piece_map():
            t_piece = curr_board.piece_at(t_sq)
            if not t_piece or t_piece.color == attacker_color: continue
            
            prev_att = prev_board.attackers(attacker_color, t_sq)
            curr_att = curr_board.attackers(attacker_color, t_sq)
            new_attackers = curr_att - prev_att
            
            if not new_attackers: continue
                
            new_attacker_pieces = [curr_board.piece_at(a) for a in new_attackers]
            lowest_attacker = min(new_attacker_pieces, key=lambda p: self.PIECE_VALUES.get(p.piece_type, 0))
            
            target_val = self.PIECE_VALUES.get(t_piece.piece_type, 0)
            attacker_val = self.PIECE_VALUES.get(lowest_attacker.piece_type, 0)
            def_ = list(curr_board.attackers(not attacker_color, t_sq))
            
            if attacker_val < target_val:
                attacks.append({"target_square": chess.square_name(t_sq), "target_piece": chess.piece_name(t_piece.piece_type), "attacker_piece": chess.piece_name(lowest_attacker.piece_type), "is_attacking": True, "attack_type": "favorable_trade", "score": target_val - attacker_val})
            elif len(curr_att) > len(def_):
                attacks.append({"target_square": chess.square_name(t_sq), "target_piece": chess.piece_name(t_piece.piece_type), "attacker_piece": chess.piece_name(lowest_attacker.piece_type), "is_attacking": True, "attack_type": "threat", "score": target_val * 0.9})
            elif attacker_val == target_val:
                attacks.append({"target_square": chess.square_name(t_sq), "target_piece": chess.piece_name(t_piece.piece_type), "attacker_piece": chess.piece_name(lowest_attacker.piece_type), "is_attacking": True, "attack_type": "trade_offer", "score": 0.1})

        if attacks:
            attacks.sort(key=lambda x: x["score"], reverse=True)
            return attacks[0]
        return {"is_attacking": False}

    def detect_pressure(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> Dict[str, Any]:
        if not move: return {"is_pressure": False}
        pressure = []
        for t_sq, t_piece in curr_board.piece_map().items():
            if t_piece.color == prev_board.turn: continue
            prev_att = list(prev_board.attackers(prev_board.turn, t_sq))
            curr_att = list(curr_board.attackers(prev_board.turn, t_sq))
            if len(curr_att) > len(prev_att):
                pressure.append({"target_square": chess.square_name(t_sq), "target_piece": chess.piece_name(t_piece.piece_type), "is_pressure": True, "target_value": self.PIECE_VALUES.get(t_piece.piece_type, 0)})
        if pressure:
            pressure.sort(key=lambda x: x["target_value"], reverse=True)
            return pressure[0]
        return {"is_pressure": False}

    def detect_brilliant_move(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move, sid: str = "") -> Tuple[bool, Dict[str, Any]]:
        if not move or move not in prev_board.legal_moves: return False, {}
        moving_piece = prev_board.piece_at(move.from_square)
        if not moving_piece or moving_piece.piece_type not in {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}: return False, {}

        if not curr_board.is_attacked_by(not prev_board.turn, move.to_square): return False, {}
        if list(curr_board.attackers(prev_board.turn, move.to_square)): return False, {}

        mover_val = self.PIECE_VALUES.get(moving_piece.piece_type, 0)
        cap_val = self.PIECE_VALUES.get(prev_board.piece_at(move.to_square).piece_type, 0) if prev_board.is_capture(move) else 0
        if mover_val <= cap_val: return False, {}

        engine = self._get_engine()
        if not engine: return False, {}

        try:
            engine.set_fen_position(prev_board.fen())
            if engine.get_best_move() != move.uci(): return False, {}
        except Exception: return False, {}

        return True, {"sacrificed_piece": chess.piece_name(moving_piece.piece_type), "sacrificed_on": chess.square_name(move.to_square), "is_free_capture": True, "compensation": "dynamic compensation and massive initiative"}

    def analyze_capture(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> Dict[str, Any]:
        if not prev_board.is_capture(move): return {}
        if prev_board.is_en_passant(move): return {"type": "equal_trade", "desc": "trades pawns via en passant"}

        captured_piece = prev_board.piece_at(move.to_square)
        capturing_piece = prev_board.piece_at(move.from_square)
        if not captured_piece or not capturing_piece: return {}

        cap_val = self.PIECE_VALUES.get(captured_piece.piece_type, 0)
        atk_val = self.PIECE_VALUES.get(capturing_piece.piece_type, 0)
        can_recapture = len(list(curr_board.attackers(not capturing_piece.color, move.to_square))) > 0
        
        if not can_recapture: return {"type": "free_capture", "target": chess.piece_name(captured_piece.piece_type)}
        elif cap_val > atk_val: return {"type": "favorable_trade", "target": chess.piece_name(captured_piece.piece_type)}
        elif cap_val == atk_val: return {"type": "equal_trade", "target": chess.piece_name(captured_piece.piece_type), "attacker": chess.piece_name(capturing_piece.piece_type)}
        else: return {"type": "sacrifice", "target": chess.piece_name(captured_piece.piece_type)}

    def detect_resolved_threats(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> List[str]:
        mover_color = prev_board.turn
        saved_pieces = []
        for sq, piece in prev_board.piece_map().items():
            if piece.color != mover_color or piece.piece_type == chess.KING: continue
            prev_att = list(prev_board.attackers(not mover_color, sq))
            if not prev_att: continue
                
            prev_def = list(prev_board.attackers(mover_color, sq))
            lowest_att = min([prev_board.piece_at(a) for a in prev_att], key=lambda p: self.PIECE_VALUES.get(p.piece_type, 0))
            is_threatened = len(prev_att) > len(prev_def) or self.PIECE_VALUES.get(lowest_att.piece_type, 0) < self.PIECE_VALUES.get(piece.piece_type, 0)
            
            if is_threatened:
                if move.from_square == sq:
                    curr_att = list(curr_board.attackers(not mover_color, move.to_square))
                    curr_def = list(curr_board.attackers(mover_color, move.to_square))
                    if len(curr_att) <= len(curr_def):
                        saved_pieces.append(chess.piece_name(piece.piece_type))
                        continue
                curr_piece = curr_board.piece_at(sq)
                if curr_piece and curr_piece == piece:
                    curr_att = list(curr_board.attackers(not mover_color, sq))
                    curr_def = list(curr_board.attackers(mover_color, sq))
                    still_threatened = False
                    if curr_att:
                        curr_lowest = min([curr_board.piece_at(a) for a in curr_att], key=lambda p: self.PIECE_VALUES.get(p.piece_type, 0))
                        still_threatened = len(curr_att) > len(curr_def) or self.PIECE_VALUES.get(curr_lowest.piece_type, 0) < self.PIECE_VALUES.get(piece.piece_type, 0)
                    if not still_threatened:
                        saved_pieces.append(chess.piece_name(piece.piece_type))
        return list(set(saved_pieces))

    def detect_hung_piece(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> Optional[Dict[str, str]]:
        if not move: return None
        mover_color = prev_board.turn
        enemy_color = not mover_color
        
        if prev_board.is_capture(move):
            cap_piece = prev_board.piece_at(move.to_square)
            mov_piece = prev_board.piece_at(move.from_square)
            if cap_piece and mov_piece and self.PIECE_VALUES.get(cap_piece.piece_type, 0) >= self.PIECE_VALUES.get(mov_piece.piece_type, 0):
                return None
        
        for sq, piece in curr_board.piece_map().items():
            if piece.color != mover_color or piece.piece_type in [chess.KING, chess.PAWN]: continue 
            
            safe_before = len(list(prev_board.attackers(enemy_color, sq))) <= len(list(prev_board.attackers(mover_color, sq)))
            curr_att = list(curr_board.attackers(enemy_color, sq))
            curr_def = list(curr_board.attackers(mover_color, sq))
            
            if len(curr_att) == 1 and curr_board.piece_at(curr_att[0]).piece_type == chess.KING and len(curr_def) > 0: continue
                
            hanging_now = len(curr_att) > len(curr_def)
            if safe_before and hanging_now:
                return {"piece": chess.piece_name(piece.piece_type)}
        return None