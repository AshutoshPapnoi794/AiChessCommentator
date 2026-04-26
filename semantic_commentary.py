from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chess


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 100.0,
}

BAD_QUALITIES = {"inaccuracy", "mistake", "blunder"}


@dataclass
class AttackTarget:
    square: int
    piece: chess.Piece
    new_attackers: int
    moved_piece_attacks: bool


@dataclass
class StructureSnapshot:
    doubled_files: List[str]
    isolated_pawns: List[str]
    passed_pawns: List[str]


@dataclass
class MoveFacts:
    san: str
    quality: str
    mover_color: chess.Color
    mover_name: str
    opponent_name: str
    move: chess.Move
    moving_piece_before: Optional[chess.Piece]
    moving_piece_after: Optional[chess.Piece]
    captured_piece: Optional[chess.Piece]
    is_capture: bool
    is_check: bool
    is_checkmate: bool
    is_castling: bool
    is_recapture: bool
    opening_name: str
    best_line_before: str
    best_move_before: str
    best_line_after: str
    best_move_after: str
    best_reply_move: Optional[chess.Move]
    best_reply_san: str
    best_reply_capture: Optional[chess.Piece]
    best_reply_gives_check: bool
    best_reply_is_mate: bool
    missed_tactics: Dict[str, Any]
    moved_piece_attacked_before: bool
    moved_piece_attacked_after: bool
    move_is_retreat: bool
    move_is_forward: bool
    new_targets: List[AttackTarget]
    chase_target: Optional[Tuple[int, chess.Piece]]
    chase_trade_target: Optional[Tuple[int, chess.Piece]]
    hung_piece: Dict[str, str]
    white_before: StructureSnapshot
    white_after: StructureSnapshot
    black_before: StructureSnapshot
    black_after: StructureSnapshot
    # Tactical context from engine
    primary_tactic_type: str
    primary_tactic_narrative: str
    move_from_sq: str
    move_to_sq: str


def build_engine_grounded_commentary(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    engine_info: Optional[Dict[str, Any]] = None,
) -> str:
    if not move:
        return ""

    engine_info = engine_info or {}
    facts = _gather_facts(prev_board, curr_board, move, context, engine_info)

    builders = (
        _commentary_for_check_response,
        _commentary_for_bad_move,
        _commentary_for_capture_sequence,
        _commentary_for_check_or_mate,
        _commentary_for_engine_tactic,
        _commentary_for_retreat_or_defense,
        _commentary_for_pawn_chase,
        _commentary_for_direct_attack,
        _commentary_for_opening_principle,
        _commentary_for_structure,
        _commentary_for_generic,
    )

    sentences: List[str] = []
    for builder in builders:
        chunk = builder(prev_board, curr_board, facts)
        if chunk:
            sentences.extend(chunk)
            break

    # Append tactic note if the primary builder didn't already cover it
    if facts.primary_tactic_type and not any(facts.primary_tactic_type in str(s).lower() for s in sentences):
        tactic_note = _tactic_appendage(facts)
        if tactic_note:
            sentences.append(tactic_note)

    unique: List[str] = []
    seen = set()
    for sentence in sentences:
        clean = " ".join(str(sentence or "").split()).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(clean)
    return " ".join(unique[:3]).strip()


def _gather_facts(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
    engine_info: Dict[str, Any],
) -> MoveFacts:
    mover_color = prev_board.turn
    mover_name = "White" if mover_color == chess.WHITE else "Black"
    opponent_name = "Black" if mover_color == chess.WHITE else "White"
    moving_piece_before = prev_board.piece_at(move.from_square)
    moving_piece_after = curr_board.piece_at(move.to_square)
    captured_piece = _captured_piece(prev_board, move)
    quality = str(context.get("quality", "") or "")
    verified = context.get("verified_facts", {}) or {}

    best_line_before = _compact_line(str(engine_info.get("best_line_before", "") or ""))
    best_move_before = str(engine_info.get("best_move_before", "") or _first_token(best_line_before))
    best_line_after = _compact_line(str(engine_info.get("best_line_after", "") or ""))
    best_move_after = str(engine_info.get("best_move_after", "") or _first_token(best_line_after))
    best_reply_move = _parse_san(curr_board, best_move_after)
    best_reply_capture = _captured_piece(curr_board, best_reply_move) if best_reply_move else None
    best_reply_gives_check = False
    best_reply_is_mate = False
    if best_reply_move and best_reply_move in curr_board.legal_moves:
        probe = curr_board.copy(stack=False)
        probe.push(best_reply_move)
        best_reply_gives_check = probe.is_check()
        best_reply_is_mate = probe.is_checkmate()

    enemy_color = not mover_color
    moved_piece_attacked_before = bool(moving_piece_before and prev_board.attackers(enemy_color, move.from_square))
    moved_piece_attacked_after = bool(curr_board.attackers(enemy_color, move.to_square))
    move_is_retreat = _is_retreat(move, mover_color, moving_piece_before)
    move_is_forward = _is_forward(move, mover_color, moving_piece_before)

    # Extract tactical context from the analysis pipeline
    tactical = context.get("tactical", {}) or {}
    primary = tactical.get("primary_tactic") or {}
    primary_tactic_type = str(primary.get("type", "") or "").replace("_", " ")
    primary_tactic_narrative = str(primary.get("narrative", "") or "")
    move_from_sq = str(verified.get("move_from", "") or chess.square_name(move.from_square))
    move_to_sq = str(verified.get("move_to", "") or chess.square_name(move.to_square))

    return MoveFacts(
        san=str((context.get("move", {}) or {}).get("san", "the move")),
        quality=quality,
        mover_color=mover_color,
        mover_name=mover_name,
        opponent_name=opponent_name,
        move=move,
        moving_piece_before=moving_piece_before,
        moving_piece_after=moving_piece_after,
        captured_piece=captured_piece,
        is_capture=prev_board.is_capture(move),
        is_check=curr_board.is_check(),
        is_checkmate=curr_board.is_checkmate(),
        is_castling=prev_board.is_castling(move),
        is_recapture=bool(verified.get("is_recapture")),
        opening_name=_normalized_opening_name(str(context.get("opening_name", "") or "")),
        best_line_before=best_line_before,
        best_move_before=best_move_before,
        best_line_after=best_line_after,
        best_move_after=best_move_after,
        best_reply_move=best_reply_move,
        best_reply_san=best_move_after,
        best_reply_capture=best_reply_capture,
        best_reply_gives_check=best_reply_gives_check,
        best_reply_is_mate=best_reply_is_mate,
        missed_tactics=context.get("missed_tactics") or {},
        moved_piece_attacked_before=moved_piece_attacked_before,
        moved_piece_attacked_after=moved_piece_attacked_after,
        move_is_retreat=move_is_retreat,
        move_is_forward=move_is_forward,
        new_targets=_new_targets(prev_board, curr_board, move),
        chase_target=_pawn_chase_target(prev_board, curr_board, move),
        chase_trade_target=_pawn_chase_trade_target(prev_board, curr_board, move),
        hung_piece=(context.get("tactical", {}) or {}).get("hung_piece") or {},
        white_before=_structure_snapshot(prev_board, chess.WHITE),
        white_after=_structure_snapshot(curr_board, chess.WHITE),
        black_before=_structure_snapshot(prev_board, chess.BLACK),
        black_after=_structure_snapshot(curr_board, chess.BLACK),
        primary_tactic_type=primary_tactic_type,
        primary_tactic_narrative=primary_tactic_narrative,
        move_from_sq=move_from_sq,
        move_to_sq=move_to_sq,
    )


def _commentary_for_check_response(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    if not prev_board.is_check():
        return []

    checker_square = next(iter(prev_board.checkers()), None)
    king_square = prev_board.king(facts.mover_color)
    if checker_square is None or king_square is None:
        return []

    if facts.moving_piece_before and facts.moving_piece_before.piece_type == chess.KING:
        lead = f"{facts.san} steps out of the check with the king."
    elif facts.move.to_square == checker_square:
        captured_checker = prev_board.piece_at(checker_square)
        lead = f"{facts.san} meets the check by capturing the {describe_piece(captured_checker, checker_square)}."
    elif facts.move.to_square in _squares_between(checker_square, king_square):
        lead = f"{facts.san} blocks the check with the {describe_piece(facts.moving_piece_after, facts.move.to_square)}."
    else:
        lead = f"{facts.san} deals with the check."

    if curr_board.is_pinned(facts.mover_color, facts.move.to_square):
        king_square_after = curr_board.king(facts.mover_color)
        if king_square_after is not None:
            return [
                lead,
                f"It also leaves the {describe_piece(facts.moving_piece_after, facts.move.to_square)} pinned to the king on {chess.square_name(king_square_after)}.",
            ]
    return [lead]


def _commentary_for_bad_move(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    if facts.quality not in BAD_QUALITIES:
        return []

    if facts.best_reply_is_mate and facts.best_reply_san:
        return [
            f"{facts.san} is a {facts.quality}, because {facts.opponent_name} has {facts.best_reply_san} and the game is effectively over.",
        ]

    punished_piece = facts.best_reply_capture or _hung_piece_from_facts(prev_board, curr_board, facts)
    if punished_piece and facts.best_reply_san:
        sentences = [_bad_move_hanging_piece_sentence(curr_board, facts, punished_piece)]
        punishment = _bad_move_punishment_sentence(curr_board, facts)
        missed = _bad_move_counterfactual_sentence(prev_board, facts)
        if punishment:
            sentences.append(punishment)
        if missed:
            sentences.append(missed)
        return [sentence for sentence in sentences if sentence]

    if facts.quality in {"mistake", "blunder"} and facts.missed_tactics:
        lead = _bad_move_missed_tactic_lead(facts)
        missed = _bad_move_counterfactual_sentence(prev_board, facts)
        punishment = _bad_move_punishment_sentence(curr_board, facts)
        sentences = [lead]
        if missed:
            sentences.append(missed)
        if punishment:
            sentences.append(punishment)
        return [sentence for sentence in sentences if sentence]

    if facts.move_is_retreat and facts.best_line_before:
        note = _summarize_line_idea(prev_board, facts.best_line_before)
        lead = f"{facts.san} retreats the {describe_piece(facts.moving_piece_after, facts.move.to_square)} from the pressure."
        if note:
            return [lead, f"The stronger line was {facts.best_line_before}, {note}."]
        return [lead, f"The stronger line was {facts.best_line_before}."]

    if facts.best_line_before:
        note = _summarize_line_idea(prev_board, facts.best_line_before)
        if note:
            return [
                f"{facts.san} is not the most accurate continuation.",
                f"The cleaner line was {facts.best_line_before}, {note}.",
            ]
        return [
            f"{facts.san} is not the most accurate continuation.",
            f"The cleaner line was {facts.best_line_before}.",
        ]

    if facts.hung_piece:
        piece = str(facts.hung_piece.get("piece", "piece"))
        square = str(facts.hung_piece.get("square", ""))
        return [f"{facts.san} is inaccurate, because it leaves the {piece} on {square} loose."]

    return [f"{facts.san} is not the most accurate continuation."]


def _commentary_for_check_or_mate(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    if facts.is_checkmate:
        return [f"{facts.san} checkmates immediately."]
    if not facts.is_check:
        return []
    enemy_king = curr_board.king(curr_board.turn)
    if enemy_king is None:
        return [f"{facts.san} gives check."]
    if facts.is_capture and facts.captured_piece:
        return [f"{facts.san} gives check while capturing the {describe_piece(facts.captured_piece, facts.move.to_square)}."]
    return [f"{facts.san} gives check, so the king on {chess.square_name(enemy_king)} has to respond first."]


def _commentary_for_engine_tactic(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    """Fires when the engine detected a primary tactic (fork, pin, skewer, etc.)."""
    if not facts.primary_tactic_type or facts.quality in BAD_QUALITIES:
        return []

    ttype = facts.primary_tactic_type
    piece_desc = describe_piece(facts.moving_piece_after, facts.move.to_square)

    if facts.primary_tactic_narrative:
        return [facts.primary_tactic_narrative]

    if "fork" in ttype:
        targets = facts.new_targets[:2]
        if len(targets) >= 2:
            t1 = describe_piece(targets[0].piece, targets[0].square)
            t2 = describe_piece(targets[1].piece, targets[1].square)
            return [
                f"{facts.san} is a fork — the {piece_desc} attacks the {t1} and the {t2} at the same time.",
                f"One of those pieces will have to be given up.",
            ]
        return [f"{facts.san} creates a fork from {facts.move_to_sq}, hitting multiple targets at once."]

    if "pin" in ttype:
        return [f"{facts.san} creates a pin with the {piece_desc}, restricting the opponent's options."]

    if "skewer" in ttype:
        return [f"{facts.san} sets up a skewer from {facts.move_to_sq} — the more valuable piece must move, exposing the one behind it."]

    if "discovered" in ttype:
        return [f"{facts.san} unleashes a discovered attack by moving the {piece_desc} out of the way."]

    if "double check" in ttype:
        return [f"{facts.san} delivers a double check from {facts.move_to_sq}, which forces the king to move."]

    return [f"{facts.san} sets up a {ttype} from {facts.move_to_sq}."]


def _tactic_appendage(facts: MoveFacts) -> str:
    """Short tactic note appended when the primary builder didn't cover the tactic."""
    if not facts.primary_tactic_type:
        return ""
    ttype = facts.primary_tactic_type
    if facts.primary_tactic_narrative:
        return facts.primary_tactic_narrative
    if "fork" in ttype:
        return f"This also creates a fork from {facts.move_to_sq}."
    if "pin" in ttype:
        return f"It also sets up a pin along the line through {facts.move_to_sq}."
    if "skewer" in ttype:
        return f"There is also a skewer from {facts.move_to_sq}."
    return f"This involves a {ttype}."


def _commentary_for_capture_sequence(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    if not facts.is_capture or not facts.captured_piece or not facts.moving_piece_before:
        return []

    captured_name = describe_piece(facts.captured_piece, facts.move.to_square)
    mover_name = chess.piece_name(facts.moving_piece_before.piece_type)
    recapture = _preferred_recapture_move(curr_board, facts.move.to_square, curr_board.turn)
    recapture_san = curr_board.san(recapture) if recapture else ""
    structure_note = _summarize_recapture_structure(curr_board, recapture)
    captured_value = PIECE_VALUES.get(facts.captured_piece.piece_type, 0.0)
    mover_value = PIECE_VALUES.get(facts.moving_piece_before.piece_type, 0.0)

    if facts.is_recapture:
        if facts.captured_piece.piece_type == chess.PAWN and _is_central_square(facts.move.to_square):
            lead = f"{facts.san} is the natural recapture, trading for the pawn on {chess.square_name(facts.move.to_square)} and restoring material balance."
        else:
            lead = f"{facts.san} is the natural recapture, taking back the {captured_name} and restoring material balance."
        if structure_note:
            return [lead, structure_note]
        return [lead]

    if facts.captured_piece.piece_type == chess.QUEEN or captured_value - mover_value >= 3.5:
        if facts.captured_piece.piece_type == chess.QUEEN:
            return [f"{facts.san} wins the queen for far less material, which is usually decisive."]
        return [f"{facts.san} wins the {captured_name} for far less material."]

    if recapture and captured_value == mover_value:
        if structure_note:
            return [
                f"{facts.san} trades {mover_name} for the {captured_name}.",
                structure_note,
            ]
        if _is_central_square(facts.move.to_square):
            return [
                f"{facts.san} trades for the pawn in the center and clarifies the structure.",
                f"{facts.opponent_name} will usually answer with {recapture_san}, so the pawn formation becomes the main story.",
            ]
        return [
            f"{facts.san} resolves the tension by capturing the {captured_name}.",
            f"{facts.opponent_name} will usually answer with {recapture_san}, restoring the material balance.",
        ]

    if recapture and captured_value > mover_value:
        return [
            f"{facts.san} gives up the {mover_name} for the {captured_name}, and the trade works in {facts.mover_name.lower()}'s favor.",
            structure_note or f"{facts.opponent_name} still has to decide how to recapture.",
        ]

    if recapture:
        return [
            f"{facts.san} captures the {captured_name} first.",
            structure_note or f"{facts.opponent_name} will usually answer with {recapture_san}.",
        ]

    return [f"{facts.san} picks up the {captured_name} with no recapture available."]


def _commentary_for_retreat_or_defense(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    if not facts.moving_piece_before or facts.moving_piece_before.piece_type in {chess.KING, chess.PAWN}:
        return []
    if not facts.moved_piece_attacked_before or facts.is_capture:
        return []

    # ── Zwischenzug / In-Between Move ────────────────────────────────────────
    # The piece was under attack but instead of retreating passively it creates
    # a new threat that is at least as dangerous as the original attacker.
    # This is one of the most important chess dynamics: "pose greater danger."
    if facts.new_targets:
        moving_val = PIECE_VALUES.get(facts.moving_piece_before.piece_type, 0.0)
        best_target = max(
            facts.new_targets,
            key=lambda t: PIECE_VALUES.get(t.piece.piece_type, 0.0)
        )
        target_val = PIECE_VALUES.get(best_target.piece.piece_type, 0.0)
        if target_val >= moving_val:
            target_desc = describe_piece(best_target.piece, best_target.square)
            lead = (
                f"{facts.san} is an in-between move — rather than retreating passively "
                f"from the immediate pressure, it fires back with a counter-threat "
                f"against the {target_desc}."
            )
            follow = (
                f"This forces {facts.opponent_name.lower()} to deal with the new problem "
                f"before they can collect on their original threat."
            )
            return [lead, follow]
    # ── End Zwischenzug ───────────────────────────────────────────────────────

    lead = ""
    if facts.move_is_retreat:
        lead = f"{facts.san} retreats the {describe_piece(facts.moving_piece_after, facts.move.to_square)} from the immediate pressure."
    else:
        lead = f"{facts.san} gets the {describe_piece(facts.moving_piece_after, facts.move.to_square)} out of trouble."

    follow_parts: List[str] = []
    pin = _absolute_or_relative_pin(curr_board, facts.move.to_square)
    if pin:
        follow_parts.append(pin)

    attack_note = _direct_attack_note(facts)
    if attack_note:
        follow_parts.append(attack_note)

    if facts.quality in BAD_QUALITIES and facts.best_line_before:
        better_note = _summarize_line_idea(prev_board, facts.best_line_before)
        if better_note:
            follow_parts.append(f"The stronger line was {facts.best_line_before}, {better_note}.")
        else:
            follow_parts.append(f"The stronger line was {facts.best_line_before}.")

    return [lead, " ".join(part for part in follow_parts if part).strip()] if follow_parts else [lead]


def _commentary_for_pawn_chase(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    if not facts.chase_target or not facts.moving_piece_before or facts.moving_piece_before.piece_type != chess.PAWN:
        return []

    target_square, target_piece = facts.chase_target
    target_desc = describe_piece(target_piece, target_square)
    file_idx = chess.square_file(facts.move.to_square)
    wing_name = ""
    if file_idx in {0, 1}:
        wing_name = "queenside"
    elif file_idx in {6, 7}:
        wing_name = "kingside"
    if wing_name:
        lead = f"{facts.san} drives the {target_desc} again and grabs more space on the {wing_name}."
    else:
        lead = f"{facts.san} asks the {target_desc} a direct question."

    trade_note = ""
    if facts.chase_trade_target:
        trade_square, trade_piece = facts.chase_trade_target
        trade_note = f"The bishop can still choose to trade itself for the {describe_piece(trade_piece, trade_square)}."

    if target_piece.piece_type == chess.BISHOP:
        follow = f"Another retreat can leave that bishop increasingly awkward if it keeps giving ground."
        if trade_note:
            follow = f"{follow} {trade_note}"
        return [lead, follow]
    if trade_note:
        return [lead, trade_note]
    return [lead]


def _commentary_for_direct_attack(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    note = _direct_attack_note(facts)
    if not note:
        return []
    lead = f"{facts.san} creates a direct threat."
    return [lead, note]


def _commentary_for_opening_principle(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    if facts.is_castling:
        side = "queenside" if chess.square_file(facts.move.to_square) < chess.square_file(facts.move.from_square) else "kingside"
        return [f"{facts.san} castles {side}, getting the king to safety and bringing the rook into play."]

    if curr_board.ply() > 14 or not facts.moving_piece_before:
        return []

    opening = facts.opening_name.lower()
    if "caro-kann" in opening and facts.san == "c6":
        return [
            f"{facts.mover_name} chooses the Caro-Kann with {facts.san}, preparing ...d5 from a solid base.",
            "The point is to challenge the center without locking in the bishop on c8.",
        ]
    if "caro-kann" in opening and facts.san == "d5":
        return [
            f"{facts.san} is the key Caro-Kann break, challenging the center immediately.",
            "It does that while keeping the bishop on c8 free.",
        ]

    if facts.moving_piece_before.piece_type == chess.PAWN and facts.move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5}:
        return [f"{facts.san} claims space in the center and opens lines for the pieces behind it."]

    if _is_minor_piece_development(prev_board, facts.move):
        sentence = f"{facts.san} develops the {describe_piece(facts.moving_piece_after, facts.move.to_square)} naturally."
        support = _supported_central_pawn(curr_board, facts.move.to_square, facts.mover_color)
        if support:
            sentence += f" It also supports the pawn on {support}."
        reply_hint = _reply_hint(facts.best_move_after, facts.opponent_name)
        if reply_hint:
            sentence += f" {reply_hint}"
        if facts.opening_name:
            sentence += f" It fits the {facts.opening_name} setup."
        return [sentence]

    return []


def _commentary_for_structure(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    mover_before = facts.white_before if facts.mover_color == chess.WHITE else facts.black_before
    mover_after = facts.white_after if facts.mover_color == chess.WHITE else facts.black_after
    opp_before = facts.black_before if facts.mover_color == chess.WHITE else facts.white_before
    opp_after = facts.black_after if facts.mover_color == chess.WHITE else facts.white_after

    new_passed = sorted(set(mover_after.passed_pawns) - set(mover_before.passed_pawns))
    if new_passed:
        return [f"{facts.san} creates a passed pawn on {new_passed[0]}, which can become a major endgame asset."]

    new_opp_isolated = sorted(set(opp_after.isolated_pawns) - set(opp_before.isolated_pawns))
    if new_opp_isolated:
        return [f"{facts.san} leaves {facts.opponent_name.lower()} with an isolated pawn on {new_opp_isolated[0]}, a weakness that can be targeted later."]

    if facts.moving_piece_before and facts.moving_piece_before.piece_type == chess.ROOK:
        file_state = _file_state(curr_board, facts.move.to_square, facts.mover_color)
        if file_state:
            return [f"{facts.san} puts the rook on the {file_state} {chess.square_name(facts.move.to_square)[0]}-file, where it has more scope."]

    return []


def _commentary_for_generic(
    prev_board: chess.Board,
    curr_board: chess.Board,
    facts: MoveFacts,
) -> List[str]:
    if facts.quality in BAD_QUALITIES and facts.best_line_before:
        note = _summarize_line_idea(prev_board, facts.best_line_before)
        if note:
            return [f"{facts.san} is playable, but it misses the cleaner continuation {facts.best_line_before}, {note}."]
        return [f"{facts.san} is playable, but it misses the cleaner continuation {facts.best_line_before}."]

    if facts.moving_piece_before and facts.moving_piece_before.piece_type == chess.PAWN and facts.san in {"c6", "e6"} and facts.opening_name:
        return [f"{facts.san} supports the central break and keeps the structure compact in the {facts.opening_name}."]

    if facts.moving_piece_after:
        piece_type = facts.moving_piece_after.piece_type
        dest = chess.square_name(facts.move.to_square)
        if piece_type == chess.KNIGHT:
            reply = _reply_hint(facts.best_move_after, facts.opponent_name)
            base = f"{facts.san} repositions the knight to {dest}, searching for a better outpost."
            return [base, reply] if reply else [base]
        if piece_type == chess.BISHOP:
            reply = _reply_hint(facts.best_move_after, facts.opponent_name)
            base = f"{facts.san} shifts the bishop to {dest}, aiming for a more active diagonal."
            return [base, reply] if reply else [base]
        if piece_type == chess.ROOK:
            return [f"{facts.san} improves the rook to {dest}, waiting for lines to open."]
        if piece_type == chess.QUEEN:
            return [f"{facts.san} centralises the queen to {dest}, increasing its scope across the board."]
        if piece_type == chess.PAWN:
            file_letter = dest[0]
            reply = _reply_hint(facts.best_move_after, facts.opponent_name)
            base = f"{facts.san} advances the {file_letter}-pawn to {dest}, slowly altering the structure."
            return [base, reply] if reply else [base]
        return [f"{facts.san} improves the {describe_piece(facts.moving_piece_after, facts.move.to_square)}."]
    return [f"{facts.san} is a waiting move, keeping options open."]


def _bad_move_hanging_piece_sentence(
    curr_board: chess.Board,
    facts: MoveFacts,
    punished_piece: chess.Piece,
) -> str:
    target_square = _capture_square_from_reply(curr_board, facts.best_reply_move)
    piece_name = chess.piece_name(punished_piece.piece_type)
    severity = "blunder" if facts.quality == "blunder" else "mistake"
    if punished_piece.piece_type == chess.QUEEN:
        return f"{facts.san} is a blunder, hanging the queen on {target_square}."
    if facts.quality == "blunder":
        return f"{facts.san} is a blunder, dropping the {piece_name} on {target_square}."
    return f"{facts.san} is a {severity}, leaving the {piece_name} on {target_square} loose."


def _bad_move_missed_tactic_lead(facts: MoveFacts) -> str:
    if facts.move_is_retreat and facts.moving_piece_after:
        return f"{facts.san} retreats the {describe_piece(facts.moving_piece_after, facts.move.to_square)}, but that misses the tactical point."
    if facts.quality == "blunder":
        return f"{facts.san} is a blunder, because it misses the tactical point of the position."
    return f"{facts.san} is a mistake, because it misses the critical idea in the position."


def _bad_move_counterfactual_sentence(prev_board: chess.Board, facts: MoveFacts) -> str:
    missed = facts.missed_tactics or {}
    details = missed.get("tactical_details") or {}
    best_move = str(missed.get("missed_best_move", "") or facts.best_move_before or _first_token(facts.best_line_before)).strip()
    best_line = str(facts.best_line_before or "").strip()

    if best_move:
        idea = _missed_tactic_idea(prev_board, facts, details)
        if idea:
            return f"The move to find was {best_move}, {idea}."
        if best_line:
            note = _summarize_line_idea(prev_board, best_line)
            if note:
                return f"The stronger line was {best_line}, {note}."
            return f"The stronger line was {best_line}."
        return f"The move to find was {best_move}."

    if best_line:
        note = _summarize_line_idea(prev_board, best_line)
        if note:
            return f"The stronger line was {best_line}, {note}."
        return f"The stronger line was {best_line}."

    return ""


def _bad_move_punishment_sentence(curr_board: chess.Board, facts: MoveFacts) -> str:
    if facts.best_reply_is_mate and facts.best_reply_san:
        return f"{facts.opponent_name} has {facts.best_reply_san}, and the game is effectively over."

    if facts.best_reply_san and facts.best_reply_capture:
        target_square = _capture_square_from_reply(curr_board, facts.best_reply_move)
        piece_name = chess.piece_name(facts.best_reply_capture.piece_type)
        if facts.best_reply_capture.piece_type == chess.QUEEN:
            material = "winning the queen"
        else:
            material = f"winning the {piece_name} on {target_square}"
        if facts.best_reply_gives_check:
            material += " with check"
        return f"{facts.opponent_name} can answer with {facts.best_reply_san}, {material}."

    if facts.best_line_after:
        note = _summarize_line_idea(curr_board, facts.best_line_after)
        if note:
            return f"{facts.opponent_name}'s clean reply is {facts.best_line_after}, {note}."
        return f"{facts.opponent_name}'s clean reply is {facts.best_line_after}."

    if facts.best_reply_san:
        if facts.best_reply_gives_check:
            return f"{facts.opponent_name} can answer with {facts.best_reply_san}, seizing the initiative with check."
        return f"{facts.opponent_name} can answer with {facts.best_reply_san} and take over the position."

    return ""


def _missed_tactic_idea(
    prev_board: chess.Board,
    facts: MoveFacts,
    details: Dict[str, Any],
) -> str:
    if details.get("gives_checkmate"):
        return "ending the game at once"

    if details.get("is_capture"):
        captured_piece = str(details.get("captured_piece", "") or "").strip()
        if captured_piece == "queen":
            return "winning the queen immediately"
        if captured_piece:
            phrase = f"winning the {captured_piece}"
            if details.get("gives_check"):
                phrase += " with check"
            return phrase

    if details.get("attacks_valuable"):
        attacked_piece = str(details.get("attacked_piece", "") or "").strip()
        if attacked_piece:
            return f"putting the {attacked_piece} under immediate pressure"

    attack_info = details.get("attack_info") or {}
    if attack_info.get("is_attacking"):
        target_piece = str(attack_info.get("target_piece", "piece") or "piece").strip()
        target_square = str(attack_info.get("target_square", "") or "").strip()
        if target_square:
            return f"starting a direct attack on the {target_piece} on {target_square}"
        return f"starting a direct attack on the {target_piece}"

    tactic_types = [str(item or "").strip() for item in details.get("tactic_types", []) if str(item or "").strip()]
    if tactic_types:
        idea = _tactic_type_phrase(tactic_types[0])
        if idea:
            return idea

    if facts.best_line_before:
        note = _summarize_line_idea(prev_board, facts.best_line_before)
        if note:
            return note

    summary = str((facts.missed_tactics or {}).get("summary", "") or "").strip()
    if "—" in summary:
        tail = summary.split("—", 1)[1].strip()
        if tail:
            return tail.rstrip(".").lower()

    return ""


def _tactic_type_phrase(tactic_type: str) -> str:
    phrases = {
        "fork": "creating a fork",
        "pin": "creating a pin",
        "skewer": "creating a skewer",
        "xray": "using an x-ray tactic",
        "double_check": "forcing the king with a double check",
        "discovered_attack": "uncovering a discovered attack",
        "removal_of_guard": "removing a key defender",
    }
    return phrases.get(str(tactic_type or "").strip(), "")


def describe_piece(piece: Optional[chess.Piece], square: int) -> str:
    if not piece:
        return f"piece on {chess.square_name(square)}"
    base = chess.piece_name(piece.piece_type)
    if piece.piece_type == chess.BISHOP:
        base = f"{_square_color_name(square)}-square bishop"
    return f"{base} on {chess.square_name(square)}"


def _direct_attack_note(facts: MoveFacts) -> str:
    if not facts.new_targets:
        return ""
    target = facts.new_targets[0]
    target_desc = describe_piece(target.piece, target.square)
    if PIECE_VALUES.get(target.piece.piece_type, 0.0) >= 5.0:
        return f"The move puts the {target_desc} under immediate pressure."
    if target.piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
        return f"It also starts asking questions of the {target_desc}."
    return ""


def _new_targets(prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> List[AttackTarget]:
    mover_color = prev_board.turn
    targets: List[AttackTarget] = []
    for square, piece in curr_board.piece_map().items():
        if piece.color == mover_color:
            continue
        before = len(list(prev_board.attackers(mover_color, square)))
        after = len(list(curr_board.attackers(mover_color, square)))
        if after <= before:
            continue
        moved_piece_attacks = bool(move.to_square in curr_board.attackers(mover_color, square))
        targets.append(
            AttackTarget(
                square=square,
                piece=piece,
                new_attackers=after - before,
                moved_piece_attacks=moved_piece_attacks,
            )
        )
    targets.sort(key=lambda item: (PIECE_VALUES.get(item.piece.piece_type, 0.0), item.new_attackers), reverse=True)
    return targets


def _pawn_chase_target(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
) -> Optional[Tuple[int, chess.Piece]]:
    piece = prev_board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN or prev_board.is_capture(move):
        return None
    mover_color = prev_board.turn
    choices: List[Tuple[float, int, chess.Piece]] = []
    for square in curr_board.attacks(move.to_square):
        target = curr_board.piece_at(square)
        if not target or target.color == mover_color or target.piece_type not in {chess.BISHOP, chess.KNIGHT, chess.QUEEN}:
            continue
        choices.append((PIECE_VALUES.get(target.piece_type, 0.0), square, target))
    if not choices:
        return None
    choices.sort(reverse=True)
    _, square, target = choices[0]
    return square, target


def _pawn_chase_trade_target(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
) -> Optional[Tuple[int, chess.Piece]]:
    chase = _pawn_chase_target(prev_board, curr_board, move)
    if not chase:
        return None
    target_square, target_piece = chase
    if target_piece.piece_type != chess.BISHOP:
        return None

    mover_color = prev_board.turn
    choices: List[Tuple[float, int, chess.Piece]] = []
    for square in prev_board.attacks(target_square):
        victim = prev_board.piece_at(square)
        if not victim or victim.color != mover_color or victim.piece_type not in {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}:
            continue
        choices.append((PIECE_VALUES.get(victim.piece_type, 0.0), square, victim))
    if not choices:
        return None
    choices.sort(reverse=True)
    _, square, victim = choices[0]
    return square, victim


def _preferred_recapture_move(board: chess.Board, square: int, color: chess.Color) -> Optional[chess.Move]:
    recaptures = [
        move
        for move in board.legal_moves
        if move.to_square == square and board.is_capture(move) and (board.piece_at(move.from_square) or chess.Piece(chess.PAWN, color)).color == color
    ]
    if not recaptures:
        return None
    return max(recaptures, key=lambda move: _recapture_score(board, move, color))


def _recapture_score(board: chess.Board, move: chess.Move, color: chess.Color) -> float:
    piece = board.piece_at(move.from_square)
    if not piece:
        return float("-inf")
    before = set(_doubled_files(board, color))
    probe = board.copy(stack=False)
    probe.push(move)
    after = set(_doubled_files(probe, color))
    new_doubled = len(after - before)
    score = 100.0
    score -= new_doubled * 100.0
    score -= PIECE_VALUES.get(piece.piece_type, 0.0)
    return score


def _summarize_recapture_structure(board: chess.Board, recapture: Optional[chess.Move]) -> str:
    if not recapture:
        return ""
    color = board.turn
    before = set(_doubled_files(board, color))
    probe = board.copy(stack=False)
    san = board.san(recapture)
    probe.push(recapture)
    after = set(_doubled_files(probe, color))
    new_files = sorted(after - before)
    side = "White" if color == chess.WHITE else "Black"
    if new_files:
        return f"{side} will usually answer with {san}, which leaves doubled pawns on the {new_files[0]}-file."
    return f"{side} will usually answer with {san}, restoring the material balance."


def _summarize_line_idea(board: chess.Board, line: str) -> str:
    if not line:
        return ""
    probe = board.copy(stack=False)
    tokens = [token for token in line.split() if token]
    parsed: List[chess.Move] = []
    for token in tokens[:4]:
        try:
            move = probe.parse_san(token)
        except ValueError:
            break
        parsed.append(move)
        probe.push(move)

    if not parsed:
        return ""

    first = parsed[0]
    first_piece = board.piece_at(first.from_square)
    if not first_piece:
        return ""

    if board.is_capture(first):
        captured = _captured_piece(board, first)
        if captured and captured.piece_type == chess.QUEEN:
            return "winning the queen immediately"
        if len(parsed) >= 2 and board.is_capture(first):
            after_first = board.copy(stack=False)
            after_first.push(first)
            recapturer = after_first.turn
            recapture = parsed[1] if parsed[1] in after_first.legal_moves else None
            if recapture:
                before_doubled = set(_doubled_files(after_first, recapturer))
                after_second = after_first.copy(stack=False)
                after_second.push(recapture)
                after_doubled = set(_doubled_files(after_second, recapturer))
                new_files = sorted(after_doubled - before_doubled)
                if new_files:
                    return f"damaging {'White' if recapturer == chess.WHITE else 'Black'}'s pawn structure on the {new_files[0]}-file"
        if captured:
            return f"trading for the {chess.piece_name(captured.piece_type)} under better circumstances"

    if first_piece.piece_type == chess.PAWN and _is_central_square(first.to_square):
        return "striking in the center at once"

    if first_piece.piece_type in {chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.ROOK}:
        after_first = board.copy(stack=False)
        after_first.push(first)
        targets = _new_targets(board, after_first, first)
        if targets:
            top = targets[0]
            return f"putting the {describe_piece(top.piece, top.square)} under pressure"

    return ""


def _absolute_or_relative_pin(board: chess.Board, attacker_square: int) -> str:
    attacker = board.piece_at(attacker_square)
    if not attacker or attacker.piece_type not in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
        return ""

    directions: List[Tuple[int, int]] = []
    if attacker.piece_type in {chess.ROOK, chess.QUEEN}:
        directions.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])
    if attacker.piece_type in {chess.BISHOP, chess.QUEEN}:
        directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    for file_step, rank_step in directions:
        pin = _scan_pin_line(board, attacker_square, attacker.color, file_step, rank_step)
        if not pin:
            continue
        pinned_square, pinned_piece, target_square, target_piece = pin
        if target_piece.color == attacker.color or pinned_piece.piece_type == chess.PAWN:
            continue
        if target_piece.piece_type == chess.KING:
            return f"It keeps the {describe_piece(pinned_piece, pinned_square)} pinned to the king on {chess.square_name(target_square)}."
        if target_piece.piece_type in {chess.QUEEN, chess.ROOK}:
            return f"It also keeps the {describe_piece(pinned_piece, pinned_square)} tied to the {describe_piece(target_piece, target_square)} behind it."
    return ""


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


def _captured_piece(board: chess.Board, move: Optional[chess.Move]) -> Optional[chess.Piece]:
    if not move or move not in board.legal_moves:
        return None
    if not board.is_capture(move):
        return None
    if board.is_en_passant(move):
        return chess.Piece(chess.PAWN, not board.turn)
    return board.piece_at(move.to_square)


def _capture_square_from_reply(board: chess.Board, move: Optional[chess.Move]) -> str:
    if not move:
        return ""
    if board.is_en_passant(move):
        rank = chess.square_rank(move.to_square) + (-1 if board.turn == chess.WHITE else 1)
        square = chess.square(chess.square_file(move.to_square), rank)
        return chess.square_name(square)
    return chess.square_name(move.to_square)


def _hung_piece_from_facts(prev_board: chess.Board, curr_board: chess.Board, facts: MoveFacts) -> Optional[chess.Piece]:
    piece_name = str(facts.hung_piece.get("piece", "") or "")
    square_name = str(facts.hung_piece.get("square", "") or "")
    if not piece_name or not square_name:
        return None
    try:
        square = chess.parse_square(square_name)
    except ValueError:
        return None
    piece = curr_board.piece_at(square)
    if piece and chess.piece_name(piece.piece_type) == piece_name:
        return piece
    return None


def _parse_san(board: chess.Board, san: str) -> Optional[chess.Move]:
    if not san:
        return None
    try:
        move = board.parse_san(san)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def _compact_line(line: str, max_tokens: int = 5) -> str:
    tokens = [token for token in str(line or "").split() if token]
    return " ".join(tokens[:max_tokens])


def _first_token(line: str) -> str:
    tokens = [token for token in str(line or "").split() if token]
    return tokens[0] if tokens else ""


def _square_color_name(square: int) -> str:
    return "light" if (chess.square_file(square) + chess.square_rank(square)) % 2 else "dark"


def _normalized_opening_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned or cleaned.lower() in {"unknown", "starting position"}:
        return ""
    return cleaned.split(":", 1)[0].strip()


def _is_central_square(square: int) -> bool:
    return square in {chess.D4, chess.E4, chess.D5, chess.E5}


def _supported_central_pawn(board: chess.Board, square: int, color: chess.Color) -> str:
    for target in board.attacks(square):
        piece = board.piece_at(target)
        if piece and piece.color == color and piece.piece_type == chess.PAWN and target in {chess.D4, chess.E4, chess.D5, chess.E5}:
            return chess.square_name(target)
    return ""


def _is_minor_piece_development(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type not in {chess.KNIGHT, chess.BISHOP}:
        return False
    home_rank = 0 if board.turn == chess.WHITE else 7
    return chess.square_rank(move.from_square) == home_rank


def _is_retreat(move: chess.Move, color: chess.Color, piece: Optional[chess.Piece]) -> bool:
    if not piece or piece.piece_type == chess.KING:
        return False
    if piece.piece_type == chess.KNIGHT:
        return False
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    return to_rank < from_rank if color == chess.WHITE else to_rank > from_rank


def _is_forward(move: chess.Move, color: chess.Color, piece: Optional[chess.Piece]) -> bool:
    if not piece or piece.piece_type == chess.KING:
        return False
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    return to_rank > from_rank if color == chess.WHITE else to_rank < from_rank


def _structure_snapshot(board: chess.Board, color: chess.Color) -> StructureSnapshot:
    return StructureSnapshot(
        doubled_files=_doubled_files(board, color),
        isolated_pawns=[chess.square_name(square) for square in _isolated_pawns(board, color)],
        passed_pawns=[chess.square_name(square) for square in _passed_pawns(board, color)],
    )


def _doubled_files(board: chess.Board, color: chess.Color) -> List[str]:
    files = [0] * 8
    for square in board.pieces(chess.PAWN, color):
        files[chess.square_file(square)] += 1
    return [chess.FILE_NAMES[index] for index, count in enumerate(files) if count > 1]


def _isolated_pawns(board: chess.Board, color: chess.Color) -> List[int]:
    pawns = list(board.pieces(chess.PAWN, color))
    files_with_pawns = {chess.square_file(square) for square in pawns}
    isolated: List[int] = []
    for square in pawns:
        file_idx = chess.square_file(square)
        if (file_idx - 1) not in files_with_pawns and (file_idx + 1) not in files_with_pawns:
            isolated.append(square)
    return sorted(isolated)


def _passed_pawns(board: chess.Board, color: chess.Color) -> List[int]:
    enemy = not color
    passed: List[int] = []
    for square in board.pieces(chess.PAWN, color):
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        is_passed = True
        for enemy_square in board.pieces(chess.PAWN, enemy):
            enemy_file = chess.square_file(enemy_square)
            enemy_rank = chess.square_rank(enemy_square)
            if abs(enemy_file - file_idx) > 1:
                continue
            if color == chess.WHITE and enemy_rank > rank_idx:
                is_passed = False
                break
            if color == chess.BLACK and enemy_rank < rank_idx:
                is_passed = False
                break
        if is_passed:
            passed.append(square)
    return sorted(passed)


def _file_state(board: chess.Board, square: int, color: chess.Color) -> str:
    file_idx = chess.square_file(square)
    friendly_pawns = 0
    enemy_pawns = 0
    for rank in range(8):
        piece = board.piece_at(chess.square(file_idx, rank))
        if not piece or piece.piece_type != chess.PAWN:
            continue
        if piece.color == color:
            friendly_pawns += 1
        else:
            enemy_pawns += 1
    if friendly_pawns == 0 and enemy_pawns == 0:
        return "open"
    if friendly_pawns == 0:
        return "semi-open"
    return ""


def _reply_hint(move_san: str, side: str) -> str:
    if move_san == "e5":
        return f"{side} can still challenge the center with e5."
    if move_san == "d5":
        return f"{side} can still strike with d5."
    if move_san == "c5":
        return f"{side} can still challenge from the side with c5."
    return ""


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
    # The while-loop already stops before appending to_square, so squares
    # already contains exactly the intermediate squares.  The previous
    # squares[:-1] was an off-by-one that dropped the square adjacent to the
    # king -- often the only legal blocking square -- causing check-blocking
    # commentary to misfire.
    return squares