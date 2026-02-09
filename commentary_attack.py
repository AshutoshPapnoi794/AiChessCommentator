from typing import Any, Dict, Optional

import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}
ATTACKABLE_VALUABLE_TYPES = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}


def _target_capture_trade_profile(post_board: chess.Board, attacker_sq: int, target_sq: int) -> Dict[str, Any]:
    attacker_piece = post_board.piece_at(attacker_sq)
    target_piece = post_board.piece_at(target_sq)
    if not attacker_piece or not target_piece:
        return {
            'target_can_capture_attacker': False,
            'has_recapture': False,
            'capture_delta_for_target': 0,
            'capture_is_losing': False,
        }

    target_probe = post_board.copy(stack=False)
    target_probe.turn = target_piece.color
    target_capture = None
    for move in target_probe.legal_moves:
        if move.from_square == target_sq and move.to_square == attacker_sq and target_probe.is_capture(move):
            target_capture = move
            break

    if target_capture is None:
        return {
            'target_can_capture_attacker': False,
            'has_recapture': False,
            'capture_delta_for_target': 0,
            'capture_is_losing': False,
        }

    after_target_capture = target_probe.copy(stack=False)
    after_target_capture.push(target_capture)

    recapture_probe = after_target_capture.copy(stack=False)
    recapture_probe.turn = attacker_piece.color
    has_recapture = any(
        move.to_square == attacker_sq and recapture_probe.is_capture(move)
        for move in recapture_probe.legal_moves
    )

    attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0)
    target_value = PIECE_VALUES.get(target_piece.piece_type, 0)
    capture_delta_for_target = attacker_value - (target_value if has_recapture else 0)

    return {
        'target_can_capture_attacker': True,
        'has_recapture': has_recapture,
        'capture_delta_for_target': capture_delta_for_target,
        'capture_is_losing': capture_delta_for_target < 0,
    }


def _target_mobility_profile(post_board: chess.Board, attacker_color: chess.Color, target_sq: int) -> Dict[str, Any]:
    target_piece = post_board.piece_at(target_sq)
    if not target_piece:
        return {'target_legal_move_count': 0, 'safe_retreat_count': 0, 'is_trapped': False}

    probe = post_board.copy(stack=False)
    probe.turn = target_piece.color
    target_moves = [m for m in probe.legal_moves if m.from_square == target_sq]
    retreat_moves = [m for m in target_moves if not probe.is_capture(m)]
    safe_retreat_count = 0

    for move in retreat_moves:
        after = probe.copy(stack=False)
        after.push(move)
        if not after.is_attacked_by(attacker_color, move.to_square):
            safe_retreat_count += 1

    return {
        'target_legal_move_count': len(target_moves),
        'safe_retreat_count': safe_retreat_count,
        # Static fallback only: truly trapped if no legal move at all.
        # Engine-confirmed trapping is handled later in validation.
        'is_trapped': len(target_moves) == 0,
    }


def _classify_attack_or_pressure(
    is_hanging: bool,
    attacker_piece_type: int,
    target_piece_type: int,
    trade_profile: Dict[str, Any],
    support_profile: Dict[str, Any],
) -> str:
    attacker_value = PIECE_VALUES.get(attacker_piece_type, 0)
    target_value = PIECE_VALUES.get(target_piece_type, 0)

    # If a higher-value target can safely take the attacker, this is a bad attack attempt.
    if target_value > attacker_value and trade_profile.get('target_can_capture_attacker') and not trade_profile.get('capture_is_losing'):
        return 'attack_blunder'

    # Calling it an "attack" on a hanging piece is meaningful when the attacker is
    # not higher value than the target (e.g., pawn->knight, bishop->rook, equal trades).
    # High-value pieces poking lower-value hanging pieces are usually not the tactical
    # motif we want to narrate as attack.
    if is_hanging:
        if attacker_value <= target_value:
            return 'attack'
        return 'none'

    if target_value > attacker_value:
        if not trade_profile.get('target_can_capture_attacker'):
            return 'attack'
        if trade_profile.get('capture_is_losing'):
            return 'attack'
        return 'none'

    # Pressure means added attacker count + exchange benefit.
    # Benefit requires numerical superiority, or equal numbers with value superiority.
    our_attackers_post = support_profile.get('our_attackers_post', 0)
    their_defenders_post = support_profile.get('their_defenders_post', 0)
    has_numerical_edge = our_attackers_post > their_defenders_post
    has_value_edge_at_equal_numbers = (
        our_attackers_post == their_defenders_post and attacker_value < target_value
    )
    if support_profile.get('attacker_added_pressure') and (has_numerical_edge or has_value_edge_at_equal_numbers):
        return 'pressure'
    return 'none'


def same_position_state(a: chess.Board, b: chess.Board) -> bool:
    return (
        a.board_fen() == b.board_fen()
        and a.turn == b.turn
        and a.castling_rights == b.castling_rights
        and a.ep_square == b.ep_square
    )


def infer_played_move(prev_board: chess.Board, current_board: chess.Board, move_hint_san: Optional[str] = None) -> Optional[chess.Move]:
    if move_hint_san:
        try:
            hinted_move = prev_board.parse_san(move_hint_san)
            trial = prev_board.copy(stack=False)
            trial.push(hinted_move)
            if same_position_state(trial, current_board):
                return hinted_move
        except ValueError:
            pass
        else:
            return hinted_move

    for move in prev_board.legal_moves:
        trial = prev_board.copy(stack=False)
        trial.push(move)
        if same_position_state(trial, current_board):
            return move
    return None


def Attackedby(prev_board: chess.Board, current_board: chess.Board, move_hint_san: Optional[str] = None) -> Dict[str, Any]:
    played_move = infer_played_move(prev_board, current_board, move_hint_san)
    if not played_move:
        return {'summary': '', 'kind': None, 'targets': [], 'played_move_uci': None}

    trial_post = prev_board.copy(stack=False)
    trial_post.push(played_move)
    post_board = current_board if same_position_state(trial_post, current_board) else trial_post

    attacker = post_board.piece_at(played_move.to_square)
    if attacker is None:
        return {'summary': '', 'kind': None, 'targets': [], 'played_move_uci': played_move.uci()}

    legal_probe = post_board.copy(stack=False)
    legal_probe.turn = attacker.color
    legal_capture_targets = {
        move.to_square
        for move in legal_probe.legal_moves
        if move.from_square == played_move.to_square and legal_probe.is_capture(move)
    }

    targets = []
    for sq in post_board.attacks(played_move.to_square):
        if sq not in legal_capture_targets:
            continue
        target_piece = post_board.piece_at(sq)
        if not target_piece:
            continue
        if target_piece.color == attacker.color:
            continue
        if target_piece.piece_type not in ATTACKABLE_VALUABLE_TYPES:
            continue

        defenders = post_board.attackers(target_piece.color, sq)
        is_hanging = len(defenders) == 0
        trade_profile = _target_capture_trade_profile(post_board, played_move.to_square, sq)
        mobility_profile = _target_mobility_profile(post_board, attacker.color, sq)
        our_attackers_post = len(post_board.attackers(attacker.color, sq))
        our_attackers_prev = len(prev_board.attackers(attacker.color, sq))
        their_defenders_post = len(defenders)
        support_profile = {
            'our_attackers_post': our_attackers_post,
            'our_attackers_prev': our_attackers_prev,
            'their_defenders_post': their_defenders_post,
            'attacker_added_pressure': our_attackers_post > our_attackers_prev,
        }
        kind = _classify_attack_or_pressure(
            is_hanging=is_hanging,
            attacker_piece_type=attacker.piece_type,
            target_piece_type=target_piece.piece_type,
            trade_profile=trade_profile,
            support_profile=support_profile,
        )
        if kind in {'attack', 'pressure'} and mobility_profile['is_trapped']:
            kind = 'attack_trapped'
        targets.append({
            'piece_type': target_piece.piece_type,
            'piece_name': chess.piece_name(target_piece.piece_type),
            'square': chess.square_name(sq),
            'is_hanging': is_hanging,
            'value': PIECE_VALUES.get(target_piece.piece_type, 0),
            'kind': kind,
            'target_can_capture_attacker': trade_profile['target_can_capture_attacker'],
            'capture_is_losing': trade_profile['capture_is_losing'],
            'capture_delta_for_target': trade_profile['capture_delta_for_target'],
            'target_legal_move_count': mobility_profile['target_legal_move_count'],
            'safe_retreat_count': mobility_profile['safe_retreat_count'],
            'is_trapped': mobility_profile['is_trapped'],
            'our_attackers_post': our_attackers_post,
            'our_attackers_prev': our_attackers_prev,
            'their_defenders_post': their_defenders_post,
            'attacker_added_pressure': support_profile['attacker_added_pressure'],
        })

    if not targets:
        return {'summary': '', 'kind': None, 'targets': [], 'played_move_uci': played_move.uci()}

    priority = {
        'attack_trapped': 0,
        'attack': 1,
        'pressure': 2,
        'attack_blunder': 3,
        'none': 9,
    }
    targets.sort(key=lambda t: (priority.get(t['kind'], 9), 0 if t['is_hanging'] else 1, -t['value']))
    actionable_targets = [t for t in targets if t.get('kind') != 'none']
    if not actionable_targets:
        return {'summary': '', 'kind': None, 'targets': targets, 'played_move_uci': played_move.uci()}
    top_target = actionable_targets[0]

    attacker_name = chess.piece_name(attacker.piece_type)
    attacker_sq = chess.square_name(played_move.to_square)
    if top_target['kind'] == 'attack_trapped':
        summary = (
            f"{attacker_name.capitalize()} on {attacker_sq} attacks a trapped "
            f"{top_target['piece_name']} on {top_target['square']}."
        )
    elif top_target['kind'] == 'attack':
        if top_target['is_hanging']:
            summary = (
                f"{attacker_name.capitalize()} on {attacker_sq} attacks "
                f"a hanging {top_target['piece_name']} on {top_target['square']}."
            )
        elif top_target.get('capture_is_losing'):
            summary = (
                f"{attacker_name.capitalize()} on {attacker_sq} attacks the defended "
                f"{top_target['piece_name']} on {top_target['square']}; capturing back loses material."
            )
        else:
            summary = (
                f"{attacker_name.capitalize()} on {attacker_sq} attacks the defended "
                f"{top_target['piece_name']} on {top_target['square']}."
            )
    elif top_target['kind'] == 'attack_blunder':
        summary = (
            f"{attacker_name.capitalize()} on {attacker_sq} pseudo-attacks "
            f"{top_target['piece_name']} on {top_target['square']}, but it can capture back safely."
        )
    else:
        summary = (
            f"{attacker_name.capitalize()} on {attacker_sq} pressures the defended "
            f"{top_target['piece_name']} on {top_target['square']}."
        )

    return {
        'summary': summary,
        'kind': top_target['kind'],
        'targets': targets,
        'played_move_uci': played_move.uci(),
    }
