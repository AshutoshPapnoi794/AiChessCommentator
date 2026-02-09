import re
from typing import Any, Dict, List, Optional

import chess

from commentary_attack import infer_played_move
from commentary_engine import engine_best_move_and_eval_for_color

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}
FORCING_TARGET_TYPES = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}


def _target_capture_trade_profile(post_board: chess.Board, attacker_sq: int, target_sq: int) -> Dict[str, Any]:
    attacker_piece = post_board.piece_at(attacker_sq)
    target_piece = post_board.piece_at(target_sq)
    if not attacker_piece or not target_piece:
        return {
            'target_can_capture_attacker': False,
            'capture_is_bad': False,
            'capture_delta_for_target': 0,
        }

    probe = post_board.copy(stack=False)
    probe.turn = target_piece.color
    target_capture = next(
        (
            mv
            for mv in probe.legal_moves
            if mv.from_square == target_sq and mv.to_square == attacker_sq and probe.is_capture(mv)
        ),
        None,
    )
    if target_capture is None:
        return {
            'target_can_capture_attacker': False,
            'capture_is_bad': False,
            'capture_delta_for_target': 0,
        }

    after_capture = probe.copy(stack=False)
    after_capture.push(target_capture)

    recapture_probe = after_capture.copy(stack=False)
    recapture_probe.turn = attacker_piece.color
    has_recapture = any(
        mv.to_square == attacker_sq and recapture_probe.is_capture(mv)
        for mv in recapture_probe.legal_moves
    )

    attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0)
    target_value = PIECE_VALUES.get(target_piece.piece_type, 0)
    capture_delta_for_target = attacker_value - (target_value if has_recapture else 0)
    return {
        'target_can_capture_attacker': True,
        'capture_is_bad': capture_delta_for_target < 0,
        'capture_delta_for_target': capture_delta_for_target,
    }


def _safe_retreat_count(post_board: chess.Board, attacker_color: chess.Color, target_sq: int) -> int:
    target_piece = post_board.piece_at(target_sq)
    if not target_piece:
        return 0

    probe = post_board.copy(stack=False)
    probe.turn = target_piece.color
    retreats = [mv for mv in probe.legal_moves if mv.from_square == target_sq and not probe.is_capture(mv)]
    safe = 0
    for mv in retreats:
        after = probe.copy(stack=False)
        after.push(mv)
        if not after.is_attacked_by(attacker_color, mv.to_square):
            safe += 1
    return safe


def build_forcing_signal(
    sid: str,
    prev_board: chess.Board,
    current_board: chess.Board,
    move_hint_san: Optional[str] = None,
) -> Dict[str, Any]:
    played_move = infer_played_move(prev_board, current_board, move_hint_san)
    if not played_move:
        return {'kind': None, 'summary': 'None'}

    attacker = current_board.piece_at(played_move.to_square)
    if attacker is None:
        return {'kind': None, 'summary': 'None'}
    if attacker.piece_type == chess.KING:
        return {'kind': None, 'summary': 'None'}

    attacker_color = attacker.color
    defender_color = not attacker_color
    attacker_sq = played_move.to_square

    legal_probe = current_board.copy(stack=False)
    legal_probe.turn = attacker_color
    legal_capture_targets = {
        mv.to_square for mv in legal_probe.legal_moves if mv.from_square == attacker_sq and legal_probe.is_capture(mv)
    }

    candidates: List[Dict[str, Any]] = []
    for sq in current_board.attacks(attacker_sq):
        if sq not in legal_capture_targets:
            continue
        target_piece = current_board.piece_at(sq)
        if not target_piece or target_piece.color == attacker_color:
            continue
        if target_piece.piece_type not in FORCING_TARGET_TYPES:
            continue

        trade_profile = _target_capture_trade_profile(current_board, attacker_sq, sq)
        safe_retreats = _safe_retreat_count(current_board, attacker_color, sq)
        candidates.append(
            {
                'target_sq': sq,
                'target_name': chess.piece_name(target_piece.piece_type),
                'target_value': PIECE_VALUES.get(target_piece.piece_type, 0),
                'target_can_capture_attacker': trade_profile['target_can_capture_attacker'],
                'target_capture_is_bad': trade_profile['capture_is_bad'],
                'safe_retreat_count': safe_retreats,
            }
        )

    if not candidates:
        return {'kind': None, 'summary': 'None'}

    best = engine_best_move_and_eval_for_color(sid, current_board, defender_color, depth=10)
    best_move = best['move'] if best and best.get('move') else None

    def addressed(cand: Dict[str, Any]) -> bool:
        if best_move is None:
            return False
        return best_move.from_square == cand['target_sq'] or best_move.to_square == attacker_sq

    addressed_candidates = [c for c in candidates if addressed(c)]
    if not addressed_candidates:
        # Avoid static "no safe retreat" trap calls here; this caused false positives
        # on defended/pinned pieces. Trapped-piece handling is delegated to
        # attack validation with engine confirmation.
        return {'kind': None, 'summary': 'None'}

    target = sorted(addressed_candidates, key=lambda x: (-x['target_value'], x['safe_retreat_count']))[0]
    attacker_name = chess.piece_name(attacker.piece_type)
    target_sq_name = chess.square_name(target['target_sq'])
    attacker_sq_name = chess.square_name(attacker_sq)

    if best_move and best_move.from_square == target['target_sq'] and current_board.is_capture(best_move):
        return {
            'kind': 'forcing_trade',
            'summary': (
                f"{attacker_name.capitalize()} on {attacker_sq_name} attacks the {target['target_name']} on {target_sq_name}, "
                f"effectively forcing an exchange."
            ),
            'target_square': target_sq_name,
            'target_name': target['target_name'],
        }

    if best_move and best_move.to_square == attacker_sq and current_board.is_capture(best_move):
        return {
            'kind': 'forcing_trade',
            'summary': (
                f"{attacker_name.capitalize()} on {attacker_sq_name} attacks the {target['target_name']} on {target_sq_name}, "
                f"and the clean reply is to trade it off."
            ),
            'target_square': target_sq_name,
            'target_name': target['target_name'],
        }

    if best_move and best_move.from_square == target['target_sq']:
        return {
            'kind': 'forcing_retreat',
            'summary': (
                f"{attacker_name.capitalize()} on {attacker_sq_name} attacks the {target['target_name']} on {target_sq_name}, "
                f"forcing that piece to retreat."
            ),
            'target_square': target_sq_name,
            'target_name': target['target_name'],
        }

    return {'kind': None, 'summary': 'None'}


def build_capture_check_signal(
    prev_board: chess.Board,
    current_board: chess.Board,
    move_hint_san: Optional[str] = None,
) -> Dict[str, Any]:
    played_move = infer_played_move(prev_board, current_board, move_hint_san)
    if not played_move or not prev_board.is_capture(played_move):
        return {'kind': None, 'summary': 'None'}

    mover = current_board.piece_at(played_move.to_square)
    if mover is None:
        return {'kind': None, 'summary': 'None'}

    if prev_board.is_en_passant(played_move):
        captured_sq = played_move.to_square - 8 if mover.color == chess.WHITE else played_move.to_square + 8
    else:
        captured_sq = played_move.to_square

    captured_piece = prev_board.piece_at(captured_sq)
    if captured_piece is None:
        return {'kind': None, 'summary': 'None'}

    captured_name = chess.piece_name(captured_piece.piece_type)
    captured_sq_name = chess.square_name(captured_sq)
    mover_name = chess.piece_name(mover.piece_type)

    defenders = prev_board.attackers(captured_piece.color, captured_sq)
    was_hanging = len(defenders) == 0
    gives_check = current_board.is_check()
    capture_resolves_check = prev_board.is_check() and captured_sq in set(prev_board.checkers())

    recapture_probe = current_board.copy(stack=False)
    recapture_probe.turn = not mover.color
    recapture_exists = any(
        mv.to_square == played_move.to_square and recapture_probe.is_capture(mv)
        for mv in recapture_probe.legal_moves
    )
    is_free_capture = was_hanging and not recapture_exists

    if capture_resolves_check:
        file_idx = chess.square_file(played_move.to_square)
        file_name = chr(ord('a') + file_idx)
        pawns_on_file_after = sum(
            1 for sq in current_board.pieces(chess.PAWN, mover.color) if chess.square_file(sq) == file_idx
        )
        doubled_on_file = pawns_on_file_after >= 2
        if mover.piece_type == chess.PAWN and captured_piece.piece_type in {chess.BISHOP, chess.KNIGHT} and doubled_on_file:
            return {
                'kind': 'recapture_checker_structure',
                'summary': (
                    f"Pawn recaptures the checking {captured_name} on {captured_sq_name}, completing the minor-piece trade "
                    f"but leaving doubled pawns on the {file_name}-file."
                ),
            }
        return {
            'kind': 'recapture_checker',
            'summary': f"{mover_name.capitalize()} recaptures the checking {captured_name} on {captured_sq_name}.",
        }

    if is_free_capture and gives_check:
        return {
            'kind': 'free_capture_check',
            'summary': f"{mover_name.capitalize()} wins the hanging {captured_name} on {captured_sq_name} with check.",
        }
    if gives_check:
        return {
            'kind': 'capture_check',
            'summary': f"{mover_name.capitalize()} captures on {captured_sq_name} with check.",
        }
    if is_free_capture:
        return {
            'kind': 'free_capture',
            'summary': f"{mover_name.capitalize()} picks up a hanging {captured_name} on {captured_sq_name}.",
        }

    return {'kind': None, 'summary': 'None'}


def enforce_move_event_consistency(
    commentary: str,
    forcing_signal: Dict[str, Any],
    capture_check_signal: Dict[str, Any],
) -> str:
    text = (commentary or "").strip()
    lower = text.lower()

    forcing_kind = forcing_signal.get('kind')
    forcing_summary = (forcing_signal.get('summary') or "None").strip()
    capture_kind = capture_check_signal.get('kind')
    capture_summary = (capture_check_signal.get('summary') or "None").strip()

    plain_move = bool(re.match(r"^(white|black)\s+makes\s+the\s+move\s+[^.!?]+[.!?]?$", lower))
    if plain_move:
        if capture_kind:
            return capture_summary
        if forcing_kind:
            return forcing_summary

    if capture_kind == 'free_capture_check':
        if "check" not in lower or not any(tok in lower for tok in ("hanging", "wins", "free", "picks up")):
            return capture_summary
    elif capture_kind == 'capture_check':
        if "check" not in lower:
            return capture_summary
    elif capture_kind == 'free_capture':
        if not any(tok in lower for tok in ("hanging", "wins", "free", "picks up")):
            return capture_summary
    elif capture_kind in {'recapture_checker', 'recapture_checker_structure'}:
        if not any(tok in lower for tok in ("recapt", "trade", "checking")):
            return capture_summary

    if forcing_kind == 'forcing_retreat':
        if not any(tok in lower for tok in ("retreat", "forced", "move away", "back off")):
            return forcing_summary
    elif forcing_kind == 'forcing_trade':
        if not any(tok in lower for tok in ("trade", "exchange", "swap")):
            return forcing_summary
    elif forcing_kind == 'forcing_trap':
        if not any(tok in lower for tok in ("trapped", "no safe", "no retreat")):
            return forcing_summary

    if forcing_kind and "pin" in lower and not any(tok in lower for tok in ("trade", "exchange", "retreat", "trapped")):
        return forcing_summary

    return text
