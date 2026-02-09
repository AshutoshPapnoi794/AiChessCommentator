from typing import Any, Dict, List, Optional

import chess

from commentary_attack import infer_played_move, same_position_state
from commentary_engine import engine_best_line_san, engine_best_move_and_eval_for_color, engine_eval_cp_for_color
from commentary_text import is_noneish_text
from narrative import GameNarrativeMemory

THREAT_GAIN_CP = 170
TACTICAL_THREAT_GAIN_CP = 55
CHECK_THREAT_GAIN_CP = 40
FORK_THREAT_GAIN_CP = 30
MATE_CP_THRESHOLD = 1500
CONCEDED_THREAT_GAIN_CP = 70

THREAT_RESPONSE_MISTAKE_CP = 80
THREAT_RESPONSE_BLUNDER_CP = 170
THREAT_RESPONSE_DEFENSE_CP = 70

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _captured_piece_on_move(board: chess.Board, move: chess.Move, mover_color: chess.Color) -> Optional[chess.Piece]:
    if not board.is_capture(move):
        return None
    if board.is_en_passant(move):
        captured_sq = move.to_square - 8 if mover_color == chess.WHITE else move.to_square + 8
    else:
        captured_sq = move.to_square
    return board.piece_at(captured_sq)


def _attacked_enemy_targets_after_move(
    board_before: chess.Board,
    move: chess.Move,
    mover_color: chess.Color,
) -> List[Dict[str, Any]]:
    board_after = board_before.copy(stack=False)
    board_after.push(move)
    attacker_sq = move.to_square
    targets: List[Dict[str, Any]] = []

    for sq in board_after.attacks(attacker_sq):
        piece = board_after.piece_at(sq)
        if not piece or piece.color == mover_color:
            continue
        value = PIECE_VALUES.get(piece.piece_type, 0)
        if value <= 0:
            continue
        targets.append(
            {
                'square': chess.square_name(sq),
                'piece_type': piece.piece_type,
                'piece_name': chess.piece_name(piece.piece_type),
                'value': value,
            }
        )

    targets.sort(key=lambda x: (-x['value'], x['square']))
    return targets


def _fork_profile(targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    has_king = any(t['piece_type'] == chess.KING for t in targets)
    material_targets = [t for t in targets if t['piece_type'] in {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}]
    if has_king and material_targets:
        return {
            'is_fork': True,
            'king_included': True,
            'targets': material_targets,
        }
    if len(material_targets) >= 2:
        return {
            'is_fork': True,
            'king_included': False,
            'targets': material_targets,
        }
    return {
        'is_fork': False,
        'king_included': has_king,
        'targets': material_targets,
    }


def _mover_side_name(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


def _capture_trade_profile_for_mover(
    board_before_capture: chess.Board,
    move: chess.Move,
    mover_color: chess.Color,
) -> Dict[str, Any]:
    captured_piece = _captured_piece_on_move(board_before_capture, move, mover_color)
    mover_piece = board_before_capture.piece_at(move.from_square)
    if not captured_piece or not mover_piece:
        return {
            'captured_piece': None,
            'captured_value': 0,
            'attacker_value': 0,
            'recapture_exists': False,
            'trade_delta_for_mover': 0,
        }

    board_after = board_before_capture.copy(stack=False)
    board_after.push(move)
    recapture_probe = board_after.copy(stack=False)
    recapture_probe.turn = not mover_color
    recapture_exists = any(
        mv.to_square == move.to_square and recapture_probe.is_capture(mv)
        for mv in recapture_probe.legal_moves
    )

    captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
    attacker_value = PIECE_VALUES.get(mover_piece.piece_type, 0)
    trade_delta_for_mover = captured_value - (attacker_value if recapture_exists else 0)
    return {
        'captured_piece': captured_piece,
        'captured_value': captured_value,
        'attacker_value': attacker_value,
        'recapture_exists': recapture_exists,
        'trade_delta_for_mover': trade_delta_for_mover,
    }


def _is_piece_immediately_capturable_after_move(
    board_before_move: chess.Board,
    move: chess.Move,
    mover_color: chess.Color,
) -> bool:
    board_after_move = board_before_move.copy(stack=False)
    if move not in board_after_move.legal_moves:
        return False
    board_after_move.push(move)
    target_sq = move.to_square
    defender_color = not mover_color
    probe = board_after_move.copy(stack=False)
    probe.turn = defender_color
    return any(
        cap.to_square == target_sq and probe.is_capture(cap)
        for cap in probe.legal_moves
    )


def build_threat_signal(sid: str, prev_board: chess.Board, current_board: chess.Board) -> Dict[str, Any]:
    # Threats are from the mover's perspective: what they threaten next if ignored.
    mover_color = prev_board.turn
    played_move = infer_played_move(prev_board, current_board)
    escaped_check_with_king = False
    if played_move:
        moved_piece = prev_board.piece_at(played_move.from_square)
        escaped_check_with_king = bool(
            prev_board.is_check() and moved_piece and moved_piece.color == mover_color and moved_piece.piece_type == chess.KING
        )

    null_probe = current_board.copy(stack=False)
    null_probe.turn = mover_color
    best = engine_best_move_and_eval_for_color(sid, null_probe, mover_color, depth=12)
    if not best or not best.get('move'):
        return {'kind': None, 'summary': 'None', 'must_mention': False}

    best_move = best['move']
    best_san = best.get('san') or best_move.uci()
    cp_after_threat = best.get('cp')
    cp_now_for_mover = engine_eval_cp_for_color(sid, current_board, mover_color, depth=10)

    if cp_after_threat is None:
        return {'kind': None, 'summary': 'None', 'must_mention': False}

    danger_gain_cp = int(cp_after_threat - cp_now_for_mover) if cp_now_for_mover is not None else 0
    gives_check = null_probe.gives_check(best_move)

    capture_profile = _capture_trade_profile_for_mover(null_probe, best_move, mover_color)
    captured_piece = capture_profile['captured_piece']
    captured_value = capture_profile['captured_value']
    captured_name = chess.piece_name(captured_piece.piece_type) if captured_piece else ""
    captured_sq_name = chess.square_name(best_move.to_square) if captured_piece else ""
    attacker_value = capture_profile['attacker_value']
    recapture_exists = capture_profile['recapture_exists']
    trade_delta_for_mover = capture_profile['trade_delta_for_mover']

    attacked_targets = _attacked_enemy_targets_after_move(null_probe, best_move, mover_color)
    fork_profile = _fork_profile(attacked_targets)
    fork_targets = fork_profile['targets']
    has_major_fork_target = any(t['value'] >= 5 for t in fork_targets)

    side_name = _mover_side_name(mover_color)
    kind = None
    must_mention = False
    summary = "None"

    if cp_after_threat >= MATE_CP_THRESHOLD and danger_gain_cp >= CHECK_THREAT_GAIN_CP:
        kind = 'mate_threat'
        must_mention = True
        summary = f"{side_name} threatens {best_san}, launching a mating attack."
    elif fork_profile['is_fork'] and (danger_gain_cp >= FORK_THREAT_GAIN_CP or (gives_check and has_major_fork_target)):
        kind = 'fork_threat'
        must_mention = True
        first = fork_targets[0]
        second = fork_targets[1] if len(fork_targets) > 1 else None
        if fork_profile.get('king_included') and first:
            summary = (
                f"{side_name} threatens {best_san}, forking the king and the {first['piece_name']} "
                f"on {first['square']}."
            )
        elif second:
            summary = (
                f"{side_name} threatens {best_san}, creating a fork on the {first['piece_name']} on {first['square']} "
                f"and the {second['piece_name']} on {second['square']}."
            )
        else:
            summary = f"{side_name} threatens {best_san}, creating a tactical fork."
    elif gives_check and captured_piece and captured_value >= 3 and danger_gain_cp >= CHECK_THREAT_GAIN_CP:
        kind = 'check_tactical_threat'
        must_mention = True
        summary = f"{side_name} threatens {best_san}, a checking capture that wins material."
    elif gives_check and danger_gain_cp >= CHECK_THREAT_GAIN_CP:
        kind = 'check_threat'
        must_mention = True
        summary = f"{side_name} threatens {best_san}, a forcing check."
    elif (
        captured_piece
        and captured_value >= 3
        and danger_gain_cp >= TACTICAL_THREAT_GAIN_CP
        and attacker_value < captured_value
        and trade_delta_for_mover > 0
    ):
        kind = 'material_threat'
        summary = f"{side_name} threatens {best_san}, winning the {captured_name} on {captured_sq_name}."
    elif danger_gain_cp >= THREAT_GAIN_CP:
        kind = 'positional_threat'
        summary = f"{side_name} threatens {best_san}, a strong improving idea if ignored."

    if not kind:
        return {'kind': None, 'summary': 'None', 'must_mention': False}
    if kind == 'positional_threat':
        return {'kind': None, 'summary': 'None', 'must_mention': False}

    # If the move simply escaped check with the king, do not turn it into a
    # generic material/positional threat sentence.
    if escaped_check_with_king and kind in {'material_threat', 'positional_threat'}:
        return {'kind': None, 'summary': 'None', 'must_mention': False}

    # Equal recapturable exchanges are trade proposals, not meaningful threats.
    if kind == 'material_threat' and trade_delta_for_mover <= 0 and recapture_exists:
        return {'kind': None, 'summary': 'None', 'must_mention': False}
    if kind == 'fork_threat' and _is_piece_immediately_capturable_after_move(null_probe, best_move, mover_color):
        return {'kind': None, 'summary': 'None', 'must_mention': False}

    if kind in {'mate_threat', 'fork_threat', 'check_tactical_threat', 'material_threat'}:
        board_after = null_probe.copy(stack=False)
        if best_move in board_after.legal_moves:
            board_after.push(best_move)
            line = engine_best_line_san(sid, board_after, max_plies=3, depth=10)
            if line:
                summary = f"{summary} If ignored: {line}."

    return {
        'kind': kind,
        'summary': summary,
        'must_mention': must_mention,
        'threat_move_san': best_san,
        'threat_move_uci': best_move.uci(),
        'threat_creator_color': mover_color,
        'danger_gain_cp': danger_gain_cp,
    }


def assess_threat_response_to_pending(
    sid: str,
    pending: Optional[Dict[str, Any]],
    prev_board: chess.Board,
    current_board: chess.Board,
    move_hint_san: Optional[str],
) -> Dict[str, Any]:
    result = {'summary': '', 'label': None, 'kind': None, 'matched': False, 'stale': False}
    if not pending:
        return result

    try:
        pending_post_board = chess.Board(pending.get('post_fen', ''))
    except ValueError:
        result['stale'] = True
        return result

    if not same_position_state(pending_post_board, prev_board):
        result['stale'] = True
        return result

    result['matched'] = True
    played_move = infer_played_move(prev_board, current_board, move_hint_san)
    if not played_move:
        return result

    threat_move_uci = str(pending.get('threat_move_uci') or '').strip()
    threat_move_san = str(pending.get('threat_move_san') or '').strip()
    threat_creator_color = pending.get('threat_creator_color')
    if threat_creator_color not in {chess.WHITE, chess.BLACK}:
        return result

    defender_color = prev_board.turn
    best_defense = engine_best_move_and_eval_for_color(sid, prev_board, defender_color, depth=12)
    played_is_best = bool(best_defense and best_defense.get('move') == played_move)

    null_probe = prev_board.copy(stack=False)
    null_probe.turn = threat_creator_color
    threat_move = None
    if threat_move_uci:
        try:
            parsed = chess.Move.from_uci(threat_move_uci)
            if parsed in null_probe.legal_moves:
                threat_move = parsed
        except ValueError:
            threat_move = None

    cp_after_response = engine_eval_cp_for_color(sid, current_board, defender_color, depth=12)
    cp_if_ignored = None
    if threat_move is not None:
        board_if_ignored = null_probe.copy(stack=False)
        board_if_ignored.push(threat_move)
        cp_if_ignored = engine_eval_cp_for_color(sid, board_if_ignored, defender_color, depth=12)

    if cp_after_response is not None and cp_if_ignored is not None:
        saved_cp = cp_after_response - cp_if_ignored
        if saved_cp <= -THREAT_RESPONSE_BLUNDER_CP:
            result['label'] = 'blunder'
            result['kind'] = 'threat_ignored'
            stronger = best_defense.get('san') if best_defense else 'a defensive move'
            result['summary'] = (
                f"This misses the {threat_move_san} threat and is a blunder; stronger was {stronger}."
            )
            return result
        if saved_cp <= -THREAT_RESPONSE_MISTAKE_CP:
            result['label'] = 'mistake'
            result['kind'] = 'threat_ignored'
            stronger = best_defense.get('san') if best_defense else 'a defensive move'
            result['summary'] = (
                f"This underestimates the {threat_move_san} threat; stronger was {stronger}."
            )
            return result
        if saved_cp >= THREAT_RESPONSE_DEFENSE_CP:
            result['label'] = 'fine'
            result['kind'] = 'threat_defended'
            result['summary'] = f"{move_hint_san or played_move.uci()} is a solid defensive move, addressing the {threat_move_san} threat."
            return result

    if played_is_best and threat_move_san:
        result['label'] = 'fine'
        result['kind'] = 'threat_defended'
        result['summary'] = f"{move_hint_san or played_move.uci()} is the best defensive response to {threat_move_san}."
        return result

    return result


def update_pending_threat_response(
    memory: GameNarrativeMemory,
    threat_data: Dict[str, Any],
    post_board: chess.Board,
    ply: int,
) -> None:
    kind = threat_data.get('kind')
    if not kind:
        memory.pending_threat_response = None
        return

    threat_move_san = str(threat_data.get('threat_move_san') or '').strip()
    threat_move_uci = str(threat_data.get('threat_move_uci') or '').strip()
    if not threat_move_san or not threat_move_uci:
        memory.pending_threat_response = None
        return

    memory.pending_threat_response = {
        'set_ply': ply,
        'post_fen': post_board.fen(),
        'kind': kind,
        'summary': threat_data.get('summary'),
        'must_mention': bool(threat_data.get('must_mention')),
        'threat_move_san': threat_move_san,
        'threat_move_uci': threat_move_uci,
        'threat_creator_color': threat_data.get('threat_creator_color'),
    }


def enforce_threat_response_consistency(commentary: str, threat_response: Dict[str, Any]) -> str:
    label = threat_response.get('label')
    kind = threat_response.get('kind')
    summary = (threat_response.get('summary') or '').strip()
    if not kind or is_noneish_text(kind) or not summary or is_noneish_text(summary):
        return commentary

    lower = (commentary or '').lower()
    if label == 'fine' and any(tok in lower for tok in ('blunder', 'mistake')):
        return commentary
    if label == 'blunder' and 'blunder' not in lower:
        return summary
    if label == 'mistake' and all(tok not in lower for tok in ('mistake', 'blunder')):
        return summary

    if kind == 'threat_defended':
        if not any(tok in lower for tok in ('defend', 'avoid', 'prevent', 'address', 'threat')):
            return summary
    if kind == 'threat_ignored':
        if not any(tok in lower for tok in ('threat', 'danger', 'miss', 'underestimate')):
            return summary

    return commentary


def build_conceded_threat_signal(sid: str, prev_board: chess.Board, current_board: chess.Board) -> Dict[str, Any]:
    mover_color = prev_board.turn
    opponent_color = current_board.turn

    cp_now_for_mover = engine_eval_cp_for_color(sid, current_board, mover_color, depth=10)
    best = engine_best_move_and_eval_for_color(sid, current_board, opponent_color, depth=12)
    if cp_now_for_mover is None or not best or not best.get('move'):
        return {'kind': None, 'summary': 'None'}

    best_move = best['move']
    best_san = best.get('san') or best_move.uci()
    cp_after_for_opp = best.get('cp')
    if cp_after_for_opp is None:
        return {'kind': None, 'summary': 'None'}

    cp_after_for_mover = -cp_after_for_opp
    danger_gain_cp = cp_now_for_mover - cp_after_for_mover
    if danger_gain_cp < CONCEDED_THREAT_GAIN_CP:
        return {'kind': None, 'summary': 'None'}

    gives_check = current_board.gives_check(best_move)
    captured_piece = _captured_piece_on_move(current_board, best_move, opponent_color)
    captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0) if captured_piece else 0

    if gives_check and captured_piece and captured_value >= 3:
        kind = 'conceded_tactical_check'
        summary = f"This move allows {best_san}, a checking tactic that wins material."
    elif captured_piece and captured_value >= 3:
        kind = 'conceded_material'
        summary = f"This move allows {best_san}, conceding material."
    elif gives_check:
        kind = 'conceded_check'
        summary = f"This move allows {best_san}, handing over the initiative with check."
    else:
        kind = 'conceded_tactic'
        summary = f"This move allows {best_san}, a strong tactical reply."

    board_after = current_board.copy(stack=False)
    if best_move in board_after.legal_moves:
        board_after.push(best_move)
        line = engine_best_line_san(sid, board_after, max_plies=3, depth=10)
        if line:
            summary = f"{summary} Likely continuation: {line}."

    return {
        'kind': kind,
        'summary': summary,
        'danger_gain_cp': int(danger_gain_cp),
        'opponent_move_san': best_san,
    }


def enforce_conceded_threat_consistency(commentary: str, conceded_data: Dict[str, Any]) -> str:
    kind = conceded_data.get('kind')
    summary = (conceded_data.get('summary') or '').strip()
    if not kind or is_noneish_text(kind) or not summary or is_noneish_text(summary):
        return commentary

    lower = (commentary or '').lower()
    if not any(tok in lower for tok in ('allow', 'concede', 'threat', 'tactic', 'initiative')):
        return summary
    return commentary


def enforce_threat_consistency(commentary: str, threat_data: Dict[str, Any]) -> str:
    kind = threat_data.get('kind')
    if not kind:
        return commentary

    summary = (threat_data.get('summary') or '').strip()
    if not summary:
        return commentary

    text = (commentary or '').strip()
    lower = text.lower()
    san = str(threat_data.get('threat_move_san') or '').strip().lower()

    if threat_data.get('must_mention'):
        if kind == 'mate_threat' and not any(tok in lower for tok in ('mate', 'mating', 'checkmate')):
            return summary
        if kind == 'fork_threat' and not any(tok in lower for tok in ('fork', 'double attack', 'king and', 'threat')):
            return summary
        if kind.startswith('check') and 'check' not in lower and 'threat' not in lower:
            return summary

    if not any(tok in lower for tok in ('threat', 'idea', 'watch', 'danger')):
        if san and san not in lower:
            return summary

    return commentary
