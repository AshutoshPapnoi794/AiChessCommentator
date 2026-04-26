from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chess
from semantic_commentary import build_engine_grounded_commentary


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def build_structured_commentary(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    engine_info: Optional[Dict[str, Any]] = None,
) -> str:
    return build_engine_grounded_commentary(
        prev_board=prev_board,
        curr_board=curr_board,
        move=move,
        context=context,
        engine_info=engine_info or {},
    )


def _commentary_from_hanging_piece_blunder(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    quality = str(context.get("quality", "") or "")
    if quality not in {"mistake", "blunder"}:
        return []

    hung = (context.get("tactical", {}) or {}).get("hung_piece") or {}
    piece = str(hung.get("piece", "") or "").strip()
    square = str(hung.get("square", "") or "").strip()
    if not piece or not square:
        return []

    san = str((context.get("move", {}) or {}).get("san", "the move"))
    if piece == "queen":
        return [f"{san} is a serious blunder, simply hanging the queen on {square}."]
    return [f"{san} is a serious mistake, leaving the {piece} on {square} loose."]


def _commentary_from_check_response(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    if not prev_board.is_check():
        return []

    san = str((context.get("move", {}) or {}).get("san", "the move"))
    mover_color = prev_board.turn
    mover_name = "White" if mover_color == chess.WHITE else "Black"
    moving_piece = prev_board.piece_at(move.from_square)
    checker_square = next(iter(prev_board.checkers()), None)
    king_square = prev_board.king(mover_color)
    if checker_square is None or king_square is None:
        return []

    if moving_piece and moving_piece.piece_type == chess.KING:
        lead = f"{san} steps out of the check with the king, which is the most direct way to solve the immediate problem."
    elif move.to_square == checker_square:
        captured_checker = prev_board.piece_at(checker_square)
        lead = f"{san} meets the check by capturing the {describe_piece(captured_checker, checker_square)}."
    elif move.to_square in _squares_between(checker_square, king_square):
        lead = f"{san} blocks the check with the {describe_piece(moving_piece, move.to_square)}."
    else:
        lead = f"{san} deals with the check in a practical way."

    follow = ""
    if curr_board.is_pinned(mover_color, move.to_square):
        king_after = curr_board.king(mover_color)
        if king_after is not None:
            follow = f"Black is out of immediate danger, but the {describe_piece(curr_board.piece_at(move.to_square), move.to_square)} is pinned to the king on {chess.square_name(king_after)}." if mover_color == chess.BLACK else f"White is out of immediate danger, but the {describe_piece(curr_board.piece_at(move.to_square), move.to_square)} is pinned to the king on {chess.square_name(king_after)}."

    return [lead, follow] if follow else [lead]


def _commentary_from_opening_setup(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    opening_name = _normalized_opening_name(str(context.get("opening_name", "") or ""))
    if not opening_name:
        return []

    san = str((context.get("move", {}) or {}).get("san", "the move"))
    mover_name = str((context.get("move", {}) or {}).get("mover", "White"))
    piece = prev_board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return []

    lowered = opening_name.lower()
    if "caro-kann" in lowered and san == "c6":
        return [
            f"{mover_name} chooses the Caro-Kann with {san}, preparing ...d5 from a very solid base rather than meeting e4 symmetrically.",
            "That is the whole opening idea: challenge the center without shutting in the bishop on c8.",
        ]

    if "french defense" in lowered and san == "e6":
        return [
            f"{mover_name} heads for a French Defense setup with {san}, keeping the structure compact and preparing ...d5.",
        ]

    return []


def _commentary_from_pawn_chase(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    piece = prev_board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return []

    mover_color = prev_board.turn
    targets: List[Tuple[int, chess.Piece]] = []
    for square in curr_board.attacks(move.to_square):
        target = curr_board.piece_at(square)
        if not target or target.color == mover_color:
            continue
        if target.piece_type not in {chess.BISHOP, chess.KNIGHT, chess.QUEEN}:
            continue
        targets.append((square, target))

    if not targets:
        return []

    san = str((context.get("move", {}) or {}).get("san", "the move"))
    square, target = targets[0]
    if target.piece_type == chess.BISHOP and chess.square_file(move.to_square) <= 1:
        lead = f"{san} kicks the {describe_piece(target, square)} again and grabs more space on the queenside."
    else:
        lead = f"{san} questions the {describe_piece(target, square)} directly, asking whether it wants to retreat or exchange itself."

    capture_targets = []
    for attacked_square in prev_board.attacks(square):
        victim = prev_board.piece_at(attacked_square)
        if victim and victim.color == mover_color and victim.piece_type in {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}:
            capture_targets.append(describe_piece(victim, attacked_square))

    if capture_targets:
        if target.piece_type == chess.BISHOP and chess.square_file(move.to_square) <= 1:
            follow = f"That can leave the bishop awkward if it has to keep backing up, although trading it for the {capture_targets[0]} is still an option."
        else:
            follow = f"That is the right practical question, because the {describe_piece(target, square)} now has to choose between retreating and trading itself for the {capture_targets[0]}."
        return [lead, follow]

    if target.piece_type == chess.BISHOP and chess.square_file(move.to_square) <= 1:
        return [lead, "Another retreat can leave that bishop short of active squares on the queenside."]

    return [lead]


def _commentary_from_piece_under_attack(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    moving_piece_before = prev_board.piece_at(move.from_square)
    if not moving_piece_before or moving_piece_before.piece_type in {chess.KING, chess.PAWN}:
        return []

    quality = str(context.get("quality", "") or "")
    verified = context.get("verified_facts", {}) or {}
    enemy_color = not prev_board.turn
    attackers = [sq for sq in prev_board.attackers(enemy_color, move.from_square)]
    if not attackers:
        return []

    san = str((context.get("move", {}) or {}).get("san", "the move"))
    moving_piece_after = curr_board.piece_at(move.to_square)

    if prev_board.is_capture(move):
        captured = prev_board.piece_at(move.to_square)
        if captured:
            if verified.get("is_recapture"):
                lead = f"{san} is the natural recapture, taking back the {describe_piece(captured, move.to_square)} and restoring material balance."
                return [lead]
            lead = f"{san} resolves the tension by capturing the {describe_piece(captured, move.to_square)}."
            follow = _expected_recapture_followup(curr_board, move.to_square)
            return [lead, follow] if follow else [lead]
        return []

    if move.from_square != move.to_square:
        moving_piece_after = curr_board.piece_at(move.to_square)
        if quality in {"inaccuracy", "mistake", "blunder"}:
            lead = f"{san} retreats the {describe_piece(moving_piece_after, move.to_square)} from the immediate pressure."
        else:
            lead = f"{san} is a sensible retreat, pulling the {describe_piece(moving_piece_after, move.to_square)} away from immediate harassment."
        follow_parts: List[str] = []

        if quality not in {"inaccuracy", "mistake", "blunder"}:
            pin = _pin_from_attacker(curr_board, move.to_square)
            if pin:
                pinned_piece = pin["pinned_piece"]
                pinned_square = pin["pinned_square"]
                target_piece = pin["target_piece"]
                target_square = pin["target_square"]
                if pin["absolute"]:
                    follow_parts.append(f"It also keeps the {describe_piece(pinned_piece, pinned_square)} pinned to the king on {chess.square_name(target_square)}.")
                else:
                    follow_parts.append(f"It also keeps pressure on the {describe_piece(pinned_piece, pinned_square)} in front of the {describe_piece(target_piece, target_square)}.")

            candidate_capture = _matching_capture_on_attacked_piece(prev_board, move.from_square)
            if candidate_capture and _queen_recap_available_after(prev_board, candidate_capture):
                side = "White" if prev_board.turn == chess.BLACK else "Black"
                follow_parts.append(f"That is cleaner than capturing at once, because {side.lower()}'s queen would come into the game after the recapture.")

        attack_info = (context.get("tactical", {}) or {}).get("attack_info", {}) or {}
        pressure_info = (context.get("tactical", {}) or {}).get("pressure_info", {}) or {}
        if attack_info.get("is_attacking"):
            target_piece = str(attack_info.get("target_piece", "piece"))
            target_square = str(attack_info.get("target_square", "") or "").strip()
            if target_square:
                follow_parts.append(f"From there it also keeps pressure on the {target_piece} on {target_square}.")
            else:
                follow_parts.append(f"From there it also keeps pressure on the {target_piece}.")
        elif pressure_info.get("is_pressure"):
            target_piece = str(pressure_info.get("target_piece", "piece"))
            target_square = str(pressure_info.get("target_square", "") or "").strip()
            if target_square:
                follow_parts.append(f"It still keeps the {target_piece} on {target_square} under some pressure.")

        return [lead, " ".join(follow_parts).strip()] if follow_parts else [lead]

    return []


def _commentary_from_central_pawn_push(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    piece = prev_board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return []
    if prev_board.is_capture(move):
        return []
    if chess.square_file(move.from_square) != chess.square_file(move.to_square):
        return []
    if move.to_square not in {chess.D4, chess.E4, chess.D5, chess.E5}:
        return []

    san = str((context.get("move", {}) or {}).get("san", "the move"))
    mover_name = str((context.get("move", {}) or {}).get("mover", "White"))
    opened = _opened_pieces_for_center_pawn(move.from_square, prev_board.turn)
    lead = f"{mover_name} strikes with {san}, taking space in the center"
    if opened:
        lead += f" and opening lines for {opened}."
    else:
        lead += "."

    follow = _opening_plan_followup(context, san)
    if not follow:
        follow = _reply_hint_sentence(curr_board, engine_info)
    return [lead, follow] if follow else [lead]


def _commentary_from_capture(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    capture_info = (context.get("tactical", {}) or {}).get("capture_analysis", {}) or {}
    if not capture_info:
        return []

    san = str((context.get("move", {}) or {}).get("san", "the move"))
    captured_piece = str(capture_info.get("target", "piece"))
    lines: List[str] = []

    if capture_info.get("type") == "equal_trade":
        lines.append(f"{san} trades for the {captured_piece} and clarifies the position.")
        recapture_note = _expected_recapture_followup(curr_board, move.to_square)
        if recapture_note:
            lines.append(recapture_note)
        else:
            best_reply = _reply_hint_sentence(curr_board, engine_info)
            if best_reply:
                lines.append(best_reply)
        return lines

    if capture_info.get("type") == "free_capture":
        piece_value_map = {"queen": 9, "rook": 5, "bishop": 3, "knight": 3, "pawn": 1}
        val = piece_value_map.get(str(captured_piece).lower(), 0)
        if val >= 5:
            return [f"{san} wins the {captured_piece} outright — the opponent has no way to take back."]
        elif val >= 3:
            return [f"{san} picks up the {captured_piece} for free, winning a piece without compensation."]
        else:
            return [f"{san} snaps off the {captured_piece} with no recapture in sight."]

    if capture_info.get("type") == "winning_capture":
        if captured_piece == "queen":
            return [f"{san} wins the queen for far less material, which is usually decisive."]
        return [f"{san} wins the {captured_piece} for far less material."]

    if capture_info.get("type") == "favorable_trade":
        return [f"{san} is a very sensible exchange, because White comes off better in the trade." if prev_board.turn == chess.WHITE else f"{san} is a very sensible exchange, because Black comes off better in the trade."]

    if capture_info.get("type") == "sacrifice":
        return [f"{san} is a genuine material investment, so the follow-up has to justify it."]

    return []


def _commentary_from_tactic(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    moving_piece = curr_board.piece_at(move.to_square)
    san = str((context.get("move", {}) or {}).get("san", "the move"))
    quality = str(context.get("quality", "") or "")
    hung = (context.get("tactical", {}) or {}).get("hung_piece") or {}

    if quality in {"inaccuracy", "mistake", "blunder"} and hung:
        return []

    pin = _pin_from_attacker(curr_board, move.to_square)
    if pin and pin["absolute"]:
        return [
            f"{san} is a very practical move: the {describe_piece(moving_piece, move.to_square)} pins the {describe_piece(pin['pinned_piece'], pin['pinned_square'])} to the king on {chess.square_name(pin['target_square'])}.",
            f"That means the {describe_piece(pin['pinned_piece'], pin['pinned_square'])} is tied down for the moment.",
        ]
    if pin:
        return [
            f"{san} is a useful pin: the {describe_piece(moving_piece, move.to_square)} fixes the {describe_piece(pin['pinned_piece'], pin['pinned_square'])} in front of the {describe_piece(pin['target_piece'], pin['target_square'])}.",
        ]

    fork = _fork_from_move(curr_board, move)
    if fork:
        return [
            f"{san} is a tactical shot, as the {describe_piece(moving_piece, move.to_square)} hits {fork['targets_text']} at the same time.",
            f"That is awkward to meet cleanly, because one of those targets usually has to give way.",
        ]

    if curr_board.is_check():
        enemy_king_sq = curr_board.king(curr_board.turn)
        if enemy_king_sq is not None:
            return [f"{san} gives check, so the king on {chess.square_name(enemy_king_sq)} has to answer first and ask questions later."]

    return []


def _commentary_from_development(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    verified = context.get("verified_facts", {}) or {}
    if not verified.get("develops_minor_piece"):
        return []

    san = str((context.get("move", {}) or {}).get("san", "the move"))
    moving_piece = curr_board.piece_at(move.to_square)
    support = list(verified.get("central_support", []) or [])
    pressure = list(verified.get("central_pressure", []) or [])
    control = list(verified.get("central_control", []) or [])

    lead = f"{san} develops the {describe_piece(moving_piece, move.to_square)} naturally"
    details: List[str] = []
    if support:
        details.append(f"supports the pawn on {support[0]}")
    if pressure:
        details.append(f"puts more pressure on {pressure[0]}")
    if control:
        details.append(f"adds control over {_join_with_and(control[:2])}")

    if details:
        lead += ", " + _join_with_and(details) + "."
    else:
        lead += " and improves the coordination of the pieces."

    follow = _reply_hint_sentence(curr_board, engine_info)
    return [lead, follow] if follow else [lead]


def _commentary_from_general_improvement(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> List[str]:
    san = str((context.get("move", {}) or {}).get("san", "the move"))
    moving_piece = curr_board.piece_at(move.to_square)
    quality = str(context.get("quality", "") or "")

    if moving_piece and moving_piece.piece_type == chess.PAWN and chess.square_file(move.to_square) <= 1:
        lead = f"{san} gains a little more space on the queenside."
    elif quality in {"inaccuracy", "mistake", "blunder"}:
        lead = f"{san} repositions the {describe_piece(moving_piece, move.to_square)}."
    else:
        lead = f"{san} improves the {describe_piece(moving_piece, move.to_square)} and keeps the position under control."
    follow = _meaningful_structure_note(context)
    if not follow and quality not in {"inaccuracy", "mistake", "blunder"}:
        follow = _reply_hint_sentence(curr_board, engine_info)
    return [lead, follow] if follow else [lead]


def _mistake_or_inaccuracy_note(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> str:
    quality = str(context.get("quality", "") or "")
    if quality not in {"inaccuracy", "mistake", "blunder"}:
        return ""

    best_line_before = _compact_line(str(engine_info.get("best_line_before", "") or "").strip())
    best_line_after = _compact_line(str(engine_info.get("best_line_after", "") or "").strip())
    hung = (context.get("tactical", {}) or {}).get("hung_piece") or {}
    hung_piece = str(hung.get("piece", "") or "").strip()
    hung_square = str(hung.get("square", "") or "").strip()

    if hung_piece and hung_square and best_line_after:
        forcing = _first_move_from_line(str(engine_info.get("best_move_after", "") or "")) or _first_move_from_line(best_line_after) or best_line_after
        side = "Black" if curr_board.turn == chess.BLACK else "White"
        return f"{side} can simply answer with {forcing}, because the {hung_piece} on {hung_square} is loose."

    if best_line_before and quality in {"inaccuracy", "mistake", "blunder"}:
        structure_note = _best_line_structure_note(prev_board, best_line_before)
        if structure_note:
            return f"The cleaner move was {best_line_before}, {structure_note}."
        return f"The cleaner move was {best_line_before}."

    if best_line_after and quality in {"mistake", "blunder"}:
        side = "Black" if curr_board.turn == chess.BLACK else "White"
        return f"{side}'s clean reply is {best_line_after}."

    return ""


def describe_piece(piece: Optional[chess.Piece], square: int, *, with_color: bool = False) -> str:
    if not piece:
        return f"piece on {chess.square_name(square)}"
    base = chess.piece_name(piece.piece_type)
    if piece.piece_type == chess.BISHOP:
        base = f"{_square_color_name(square)}-square bishop"
    name = f"{base} on {chess.square_name(square)}"
    if with_color:
        color = "White" if piece.color == chess.WHITE else "Black"
        return f"{color} {name}"
    return name


def _normalized_opening_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned or cleaned.lower() in {"unknown", "starting position"}:
        return ""
    return cleaned.split(":", 1)[0].strip()


def _reply_hint_sentence(curr_board: chess.Board, engine_info: Dict[str, Any]) -> str:
    best_line = _compact_line(str(engine_info.get("best_line_after", "") or "").strip())
    first_move = _first_move_from_line(best_line)
    if not first_move:
        return ""

    side = "Black" if curr_board.turn == chess.BLACK else "White"
    if first_move == "e5":
        return f"{side} can meet that with ...e5 and challenge the center immediately." if side == "Black" else f"{side} can continue with e5 and grab more space."
    if first_move == "d5":
        return f"{side} is ready for ...d5, hitting the center at once." if side == "Black" else f"{side} can strike with d5 and seize more space."
    if first_move == "c5":
        return f"{side} can counter with ...c5, hitting the center from the side." if side == "Black" else f"{side} can play c5 and expand on the queenside."
    if first_move == "d4":
        return "White can follow with d4 and build the full center."
    if first_move == "Nf3":
        return "White can keep it flexible with Nf3, developing and holding the central tension."
    if first_move in {"cxd5", "exd5"}:
        return f"{side} can simply recapture with {first_move}, so the structure is what really matters."
    if first_move in {"dxe4", "cxd4"}:
        return f"{side} can capture in the center with {first_move} and ask the position a direct question."
    if curr_board.ply() <= 10:
        return f"{side}'s most natural continuation is {first_move}."
    return f"{side}'s cleanest continuation is {best_line}."


def _opening_plan_followup(context: Dict[str, Any], san: str) -> str:
    opening_name = _normalized_opening_name(str(context.get("opening_name", "") or ""))
    if not opening_name:
        return ""

    lowered = opening_name.lower()
    if "caro-kann" in lowered:
        if san == "d4":
            return "White builds the full center, and Black's thematic answer is still ...d5."
        if san == "d5":
            return "That is the key Caro-Kann break, challenging the pawn on e4 while keeping the bishop on c8 free."
    return ""


def _opened_pieces_for_center_pawn(from_square: int, color: chess.Color) -> str:
    file_idx = chess.square_file(from_square)
    if file_idx == chess.FILE_NAMES.index("e"):
        bishop_square = "f1" if color == chess.WHITE else "f8"
        queen_square = "d1" if color == chess.WHITE else "d8"
        return f"the bishop on {bishop_square} and the queen on {queen_square}"
    if file_idx == chess.FILE_NAMES.index("d"):
        bishop_square = "c1" if color == chess.WHITE else "c8"
        queen_square = "d1" if color == chess.WHITE else "d8"
        return f"the bishop on {bishop_square} and the queen on {queen_square}"
    return ""


def _fork_from_move(board: chess.Board, move: chess.Move) -> Optional[Dict[str, str]]:
    moved_piece = board.piece_at(move.to_square)
    if not moved_piece:
        return None

    targets: List[Tuple[int, chess.Piece]] = []
    mover_value = PIECE_VALUES.get(moved_piece.piece_type, 0.0)
    for square in board.attacks(move.to_square):
        target = board.piece_at(square)
        if not target or target.color == moved_piece.color or target.piece_type == chess.PAWN:
            continue
        target_value = PIECE_VALUES.get(target.piece_type, 0.0)
        if target.piece_type == chess.KING or target_value > mover_value or not list(board.attackers(target.color, square)):
            targets.append((square, target))

    if len(targets) < 2:
        return None

    chosen = sorted(targets, key=lambda item: PIECE_VALUES.get(item[1].piece_type, 0.0), reverse=True)[:2]
    return {"targets_text": _join_with_and([describe_piece(piece, square) for square, piece in chosen])}


def _pin_from_attacker(board: chess.Board, attacker_square: int) -> Optional[Dict[str, Any]]:
    attacker = board.piece_at(attacker_square)
    if not attacker or attacker.piece_type not in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
        return None

    directions: List[Tuple[int, int]] = []
    if attacker.piece_type in {chess.ROOK, chess.QUEEN}:
        directions.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])
    if attacker.piece_type in {chess.BISHOP, chess.QUEEN}:
        directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    for file_step, rank_step in directions:
        pinned = _scan_pin_line(board, attacker_square, attacker.color, file_step, rank_step)
        if not pinned:
            continue
        pinned_square, pinned_piece, target_square, target_piece = pinned
        if target_piece.color == attacker.color:
            continue
        if target_piece.piece_type == chess.KING:
            return {
                "absolute": True,
                "pinned_square": pinned_square,
                "pinned_piece": pinned_piece,
                "target_square": target_square,
                "target_piece": target_piece,
            }
        if target_piece.piece_type in {chess.QUEEN, chess.ROOK}:
            if pinned_piece.piece_type == chess.PAWN:
                continue
            return {
                "absolute": False,
                "pinned_square": pinned_square,
                "pinned_piece": pinned_piece,
                "target_square": target_square,
                "target_piece": target_piece,
            }
    return None


def _scan_pin_line(
    board: chess.Board,
    from_square: int,
    color: chess.Color,
    file_step: int,
    rank_step: int,
) -> Optional[Tuple[int, chess.Piece, int, chess.Piece]]:
    current_file = chess.square_file(from_square)
    current_rank = chess.square_rank(from_square)
    pinned_square: Optional[int] = None
    pinned_piece: Optional[chess.Piece] = None

    while True:
        current_file += file_step
        current_rank += rank_step
        if not (0 <= current_file < 8 and 0 <= current_rank < 8):
            return None
        current_square = chess.square(current_file, current_rank)
        piece = board.piece_at(current_square)
        if not piece:
            continue
        if piece.color == color:
            return None
        if pinned_piece is None:
            if piece.piece_type == chess.KING:
                return None
            pinned_square = current_square
            pinned_piece = piece
            continue
        return pinned_square, pinned_piece, current_square, piece


def _squares_between(from_square: int, to_square: int) -> List[int]:
    file_diff = chess.square_file(to_square) - chess.square_file(from_square)
    rank_diff = chess.square_rank(to_square) - chess.square_rank(from_square)
    step_file = 0 if file_diff == 0 else file_diff // abs(file_diff)
    step_rank = 0 if rank_diff == 0 else rank_diff // abs(rank_diff)
    if file_diff and rank_diff and abs(file_diff) != abs(rank_diff):
        return []

    squares: List[int] = []
    current_file = chess.square_file(from_square) + step_file
    current_rank = chess.square_rank(from_square) + step_rank
    while (current_file, current_rank) != (chess.square_file(to_square), chess.square_rank(to_square)):
        squares.append(chess.square(current_file, current_rank))
        current_file += step_file
        current_rank += step_rank
    return squares[:-1] if squares else []


def _legal_recaptures_on_square(board: chess.Board, square: int, color: chess.Color) -> List[chess.Move]:
    recaptures = []
    for move in board.legal_moves:
        if move.to_square != square:
            continue
        piece = board.piece_at(move.from_square)
        if piece and piece.color == color and board.is_capture(move):
            recaptures.append(move)
    return recaptures


def _preferred_recapture_san(board: chess.Board, square: int, color: chess.Color) -> str:
    recapture = _preferred_recapture_move(board, square, color)
    return board.san(recapture) if recapture else ""


def _matching_capture_on_attacked_piece(board: chess.Board, from_square: int) -> Optional[chess.Move]:
    piece = board.piece_at(from_square)
    if not piece:
        return None
    for move in board.legal_moves:
        if move.from_square != from_square or not board.is_capture(move):
            continue
        target = board.piece_at(move.to_square)
        if target and target.color != piece.color and target.piece_type in {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}:
            return move
    return None


def _queen_recap_available_after(board: chess.Board, move: chess.Move) -> bool:
    if move not in board.legal_moves:
        return False
    probe = board.copy(stack=False)
    probe.push(move)
    side_to_move = probe.turn
    for recapture in probe.legal_moves:
        if recapture.to_square != move.to_square or not probe.is_capture(recapture):
            continue
        piece = probe.piece_at(recapture.from_square)
        if piece and piece.color == side_to_move and piece.piece_type == chess.QUEEN:
            return True
    return False


def _expected_recapture_followup(board: chess.Board, square: int) -> str:
    color = board.turn
    chosen = _preferred_recapture_move(board, square, color)
    if not chosen:
        return ""
    recapture_san = board.san(chosen)

    probe = board.copy(stack=False)
    files_before = set(_doubled_files(probe, color))
    probe.push(chosen)
    files_after = set(_doubled_files(probe, color))
    new_files = sorted(files_after - files_before)
    side = "White" if color == chess.WHITE else "Black"

    if new_files:
        return f"{side} will usually answer with {recapture_san}, which leaves doubled pawns on the {new_files[0]}-file."
    return f"{side} will usually answer with {recapture_san}, restoring the material balance."


def _meaningful_structure_note(context: Dict[str, Any]) -> str:
    pawn_structure = ((context.get("analysis", {}) or {}).get("pawn_structure", {}) or {})
    mover = str(((context.get("move", {}) or {}).get("mover", "White")))
    mover_key = "white" if mover == "White" else "black"
    opp_key = "black" if mover == "White" else "white"
    mover_ps = pawn_structure.get(mover_key, {}) or {}
    opp_ps = pawn_structure.get(opp_key, {}) or {}

    passed = list(mover_ps.get("passed", []) or [])
    if passed:
        return f"The long-term asset is the passed pawn on {passed[0]}."

    isolated = list(opp_ps.get("isolated", []) or [])
    if isolated:
        return f"It also points toward the isolated pawn on {isolated[0]}, which could become a lasting target."

    backward = list(opp_ps.get("backward", []) or [])
    if backward:
        return f"The backward pawn on {backward[0]} may become a fixed weakness later on."

    return ""


def _join_with_and(parts: List[str]) -> str:
    cleaned = [part for part in parts if part]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _first_move_from_line(line: str) -> str:
    tokens = [token for token in str(line or "").split() if token]
    return tokens[0] if tokens else ""


def _compact_line(line: str, max_tokens: int = 5) -> str:
    tokens = [token for token in str(line or "").split() if token]
    return " ".join(tokens[:max_tokens])


def _parse_san_line(board: chess.Board, line: str, max_moves: int = 2) -> List[chess.Move]:
    probe = board.copy(stack=False)
    parsed: List[chess.Move] = []
    for token in [token for token in str(line or "").split() if token][:max_moves]:
        try:
            move = probe.parse_san(token)
        except ValueError:
            break
        parsed.append(move)
        probe.push(move)
    return parsed


def _best_line_structure_note(board: chess.Board, line: str) -> str:
    parsed = _parse_san_line(board, line, max_moves=2)
    if len(parsed) < 2 or not board.is_capture(parsed[0]):
        return ""

    first_move = parsed[0]
    first_piece = board.piece_at(first_move.from_square)
    captured_piece = board.piece_at(first_move.to_square)
    if not first_piece or not captured_piece:
        return ""

    after_first = board.copy(stack=False)
    after_first.push(first_move)
    recapturer = after_first.turn
    before_files = set(_doubled_files(after_first, recapturer))

    after_second = after_first.copy(stack=False)
    if parsed[1] not in after_first.legal_moves:
        return ""
    after_second.push(parsed[1])
    after_files = set(_doubled_files(after_second, recapturer))
    new_files = sorted(after_files - before_files)
    if not new_files:
        return ""

    side = "White" if recapturer == chess.WHITE else "Black"
    if captured_piece.piece_type == chess.KNIGHT:
        return f"trading for the knight and damaging {side}'s pawn structure on the {new_files[0]}-file"
    return f"damaging {side}'s pawn structure on the {new_files[0]}-file"


def _doubled_files(board: chess.Board, color: chess.Color) -> List[str]:
    files = [0] * 8
    for square in board.pieces(chess.PAWN, color):
        files[chess.square_file(square)] += 1
    return [chess.FILE_NAMES[index] for index, count in enumerate(files) if count > 1]


def _preferred_recapture_move(board: chess.Board, square: int, color: chess.Color) -> Optional[chess.Move]:
    recaptures = _legal_recaptures_on_square(board, square, color)
    if not recaptures:
        return None
    return max(recaptures, key=lambda move: _recapture_score(board, move, color))


def _recapture_score(board: chess.Board, move: chess.Move, color: chess.Color) -> float:
    piece = board.piece_at(move.from_square)
    if not piece:
        return float("-inf")

    before_files = set(_doubled_files(board, color))
    probe = board.copy(stack=False)
    probe.push(move)
    after_files = set(_doubled_files(probe, color))
    new_doubled = len(after_files - before_files)

    score = 100.0
    score -= new_doubled * 100.0
    score -= PIECE_VALUES.get(piece.piece_type, 0.0)
    return score


def _square_color_name(square: int) -> str:
    return "light" if (chess.square_file(square) + chess.square_rank(square)) % 2 else "dark"