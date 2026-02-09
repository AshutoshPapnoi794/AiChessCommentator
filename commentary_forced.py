from typing import Any, Dict, Optional

import chess

from commentary_attack import ATTACKABLE_VALUABLE_TYPES, PIECE_VALUES, infer_played_move, same_position_state
from commentary_engine import (
    engine_best_line_san,
    engine_best_move_and_eval_for_color,
    engine_eval_cp_after_move_for_color,
    engine_eval_cp_for_color,
)
from narrative import GameNarrativeMemory

FORCED_RESPONSE_MISTAKE_CP = 80
FORCED_RESPONSE_BLUNDER_CP = 170
FORCED_RESPONSE_FORCE_GAP_CP = 80


def _describe_active_counterplay(
    current_board: chess.Board,
    defender_color: chess.Color,
    move_to_sq: int,
    pending: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    moved_piece = current_board.piece_at(move_to_sq)
    if moved_piece is None or moved_piece.color != defender_color:
        return None

    enemy_color = not defender_color
    attacked_squares = set(current_board.attacks(move_to_sq))
    active_targets = []
    moved_piece_value = PIECE_VALUES.get(moved_piece.piece_type, 0)

    for sq in attacked_squares:
        piece = current_board.piece_at(sq)
        if not piece or piece.color != enemy_color:
            continue
        if piece.piece_type not in ATTACKABLE_VALUABLE_TYPES:
            continue
        defenders = len(current_board.attackers(enemy_color, sq))
        attackers = len(current_board.attackers(defender_color, sq))
        if attackers < defenders:
            continue

        # Ignore pseudo-activity where the target can simply take the moved piece
        # without losing material in return.
        target_can_capture = False
        target_capture_is_bad = False
        probe = current_board.copy(stack=False)
        probe.turn = enemy_color
        for mv in probe.legal_moves:
            if mv.from_square == sq and mv.to_square == move_to_sq and probe.is_capture(mv):
                target_can_capture = True
                after_capture = probe.copy(stack=False)
                after_capture.push(mv)
                recapture_probe = after_capture.copy(stack=False)
                recapture_probe.turn = defender_color
                has_recapture = any(
                    rec.to_square == move_to_sq and recapture_probe.is_capture(rec)
                    for rec in recapture_probe.legal_moves
                )
                target_value = PIECE_VALUES.get(piece.piece_type, 0)
                capture_delta = moved_piece_value - (target_value if has_recapture else 0)
                target_capture_is_bad = capture_delta < 0
                break

        if target_can_capture and not target_capture_is_bad and PIECE_VALUES.get(piece.piece_type, 0) <= moved_piece_value:
            continue

        active_targets.append({
            'square': chess.square_name(sq),
            'piece_name': chess.piece_name(piece.piece_type),
            'piece_value': PIECE_VALUES.get(piece.piece_type, 0),
        })

    attacker_square_name = pending.get('attacker_square') if pending else None
    attacks_original_attacker = False
    if attacker_square_name:
        try:
            original_attacker_sq = chess.parse_square(attacker_square_name)
            if original_attacker_sq in attacked_squares:
                original_attacker = current_board.piece_at(original_attacker_sq)
                attacks_original_attacker = bool(
                    original_attacker
                    and original_attacker.color == enemy_color
                    and original_attacker.piece_type in ATTACKABLE_VALUABLE_TYPES
                )
        except ValueError:
            attacks_original_attacker = False

    if not attacks_original_attacker and not active_targets:
        return None

    active_targets.sort(key=lambda x: (-x['piece_value'], x['square']))
    top = active_targets[0] if active_targets else None
    return {
        'attacks_original_attacker': attacks_original_attacker,
        'top_target': top,
    }


def assess_forced_response_to_hanging_attack(
    sid: str,
    pending: Optional[Dict[str, Any]],
    prev_board: chess.Board,
    current_board: chess.Board,
    move_hint_san: Optional[str],
) -> Dict[str, Any]:
    result = {'summary': '', 'label': None, 'response_kind': None, 'matched': False, 'stale': False}
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

    target_square_name = pending.get('target_square')
    if not target_square_name:
        return result
    try:
        target_sq = chess.parse_square(target_square_name)
    except ValueError:
        return result

    defender_color = prev_board.turn
    target_piece = prev_board.piece_at(target_sq)
    if not target_piece or target_piece.color != defender_color:
        result['label'] = 'fine'
        result['response_kind'] = 'resolved_elsewhere'
        result['summary'] = "The hanging-piece threat was resolved tactically before a direct save was needed."
        return result

    target_moves = [m for m in prev_board.legal_moves if m.from_square == target_sq]
    if not target_moves:
        result['label'] = 'fine'
        result['response_kind'] = 'forced'
        result['summary'] = "The attacked piece had no legal move; the position was forcing."
        return result

    evaluated_target_moves = []
    for move in target_moves:
        cp = engine_eval_cp_after_move_for_color(sid, prev_board, move, defender_color, depth=12)
        if cp is None:
            continue
        try:
            san = prev_board.san(move)
        except ValueError:
            san = move.uci()
        evaluated_target_moves.append({
            'move': move,
            'san': san,
            'cp': cp,
            'is_capture': prev_board.is_capture(move),
        })

    if not evaluated_target_moves:
        if played_move.from_square == target_sq:
            result['label'] = 'fine'
            if prev_board.is_capture(played_move):
                result['response_kind'] = 'forcing_trade'
                result['summary'] = "The attacked piece traded immediately; this practical defense is acceptable."
            else:
                result['response_kind'] = 'piece_retreated'
                result['summary'] = "The attacked piece moved away from danger."
        return result

    best_target = max(evaluated_target_moves, key=lambda x: x['cp'])
    capture_options = [x for x in evaluated_target_moves if x['is_capture']]
    best_capture = max(capture_options, key=lambda x: x['cp']) if capture_options else None

    played_cp = engine_eval_cp_for_color(sid, current_board, defender_color, depth=12)
    if played_cp is None:
        return result

    cp_loss = best_target['cp'] - played_cp
    played_is_capture = prev_board.is_capture(played_move)
    played_from_target = played_move.from_square == target_sq

    best_any = engine_best_move_and_eval_for_color(sid, prev_board, defender_color, depth=12)
    if best_any and (best_any['cp'] - best_target['cp']) > FORCED_RESPONSE_FORCE_GAP_CP:
        result['label'] = 'fine'
        result['response_kind'] = 'not_forced'
        result['summary'] = (
            f"The attacked {pending.get('target_piece_name', 'piece')} was not truly forced to move; "
            f"{best_any['san']} was a stronger counterplay resource."
        )
        return result

    if played_is_capture and cp_loss >= FORCED_RESPONSE_BLUNDER_CP:
        line_after_played = engine_best_line_san(sid, current_board, max_plies=4, depth=12)
        result['label'] = 'blunder'
        result['response_kind'] = 'forcing_trade'
        result['summary'] = (
            f"{move_hint_san or played_move.uci()} grabs material but is a blunder; "
            f"better was {best_target['san']}. {line_after_played or 'Tactical punishment follows.'}"
        )
        return result

    if (not played_from_target) and (not played_is_capture) and best_capture:
        capture_gain_over_played = best_capture['cp'] - played_cp
        if capture_gain_over_played >= FORCED_RESPONSE_MISTAKE_CP:
            board_after_best_capture = prev_board.copy(stack=False)
            board_after_best_capture.push(best_capture['move'])
            best_capture_line = engine_best_line_san(sid, board_after_best_capture, max_plies=4, depth=12)
            result['label'] = 'mistake'
            result['response_kind'] = 'ignored_threat'
            result['summary'] = (
                f"Ignoring the attacked piece was a mistake; forcing trade with {best_capture['san']} was stronger. "
                f"{best_capture_line or 'Engine prefers the capture continuation.'}"
            )
            return result

    if played_from_target and played_is_capture and cp_loss <= FORCED_RESPONSE_MISTAKE_CP:
        line_after_played = engine_best_line_san(sid, current_board, max_plies=4, depth=12)
        result['label'] = 'fine'
        result['response_kind'] = 'forcing_trade'
        result['summary'] = (
            f"The attacked piece forced a trade and it is fine; engine sees a stable continuation: "
            f"{line_after_played or 'roughly equal continuation'}."
        )
        return result

    if played_from_target and (not played_is_capture):
        active_counter = _describe_active_counterplay(current_board, defender_color, played_move.to_square, pending)
        if active_counter:
            top_target = active_counter.get('top_target') or {}
            target_text = (
                f"{top_target.get('piece_name')} on {top_target.get('square')}"
                if top_target.get('piece_name') and top_target.get('square')
                else "enemy pieces"
            )
            suffix = " while also hitting the original attacker" if active_counter.get('attacks_original_attacker') else ""
            if cp_loss >= FORCED_RESPONSE_BLUNDER_CP:
                result['label'] = 'blunder'
                result['response_kind'] = 'active_counter'
                result['summary'] = (
                    f"This active jump is still a blunder; better was {best_target['san']} against the threat."
                )
                return result
            if cp_loss >= FORCED_RESPONSE_MISTAKE_CP:
                result['label'] = 'mistake'
                result['response_kind'] = 'active_counter'
                result['summary'] = (
                    f"The piece moved to an active square targeting {target_text}{suffix}, "
                    f"but stronger was {best_target['san']}."
                )
                return result
            result['label'] = 'fine'
            result['response_kind'] = 'active_counter'
            result['summary'] = (
                f"The attacked piece chose an active square, targeting {target_text}{suffix}; this defense is acceptable."
            )
            return result

        safe_retreated = not current_board.is_attacked_by(not defender_color, played_move.to_square)
        if cp_loss >= FORCED_RESPONSE_BLUNDER_CP:
            result['label'] = 'blunder'
            result['response_kind'] = 'piece_retreated'
            result['summary'] = (
                f"The retreat was a blunder; {best_target['san']} was required against the attack."
            )
            return result
        if cp_loss >= FORCED_RESPONSE_MISTAKE_CP:
            result['label'] = 'mistake'
            result['response_kind'] = 'piece_retreated'
            result['summary'] = (
                f"The piece retreated, but it was a mistake; stronger was {best_target['san']}."
            )
            return result
        result['label'] = 'fine'
        result['response_kind'] = 'piece_retreated'
        if safe_retreated:
            result['summary'] = "The attacked piece retreated to a safe square; this defense is acceptable."
        else:
            result['summary'] = "The piece retreated, but the square is still contested; engine still considers it acceptable."
        return result

    if cp_loss >= FORCED_RESPONSE_BLUNDER_CP:
        result['label'] = 'blunder'
        result['summary'] = f"This reply to the hanging-piece threat is a blunder; better was {best_target['san']}."
        result['response_kind'] = 'other'
    elif cp_loss >= FORCED_RESPONSE_MISTAKE_CP:
        result['label'] = 'mistake'
        result['summary'] = f"This reply is a mistake; {best_target['san']} was the better response to the threat."
        result['response_kind'] = 'other'
    else:
        result['label'] = 'fine'
        result['summary'] = "This response to the hanging-piece threat is acceptable by engine evaluation."
        result['response_kind'] = 'other'

    return result


def update_pending_forced_response(
    memory: GameNarrativeMemory,
    attackedby_data: Dict[str, Any],
    post_board: chess.Board,
    attacker_color: chess.Color,
    ply: int,
) -> None:
    targets = attackedby_data.get('targets') or []
    top = targets[0] if targets else None
    if attackedby_data.get('kind') in {'attack', 'attack_trapped'} and top:
        played_move_uci = attackedby_data.get('played_move_uci') or ""
        attacker_square = played_move_uci[2:4] if isinstance(played_move_uci, str) and len(played_move_uci) >= 4 else None
        memory.pending_forced_response = {
            'set_ply': ply,
            'post_fen': post_board.fen(),
            'attacker_color': attacker_color,
            'target_square': top.get('square'),
            'target_piece_name': top.get('piece_name'),
            'target_is_trapped': top.get('is_trapped', False),
            'attack_summary': attackedby_data.get('summary'),
            'attacker_square': attacker_square,
        }
        return

    memory.pending_forced_response = None


def enforce_forced_response_consistency(commentary: str, forced_response: Dict[str, Any]) -> str:
    label = forced_response.get('label')
    response_kind = forced_response.get('response_kind')
    summary = forced_response.get('summary') or ""
    if not label or not summary:
        return commentary

    lower = (commentary or "").lower()
    if label == 'blunder' and "blunder" not in lower:
        return summary
    if label == 'mistake' and ("mistake" not in lower and "blunder" not in lower):
        return summary
    if label == 'fine' and any(word in lower for word in ("blunder", "mistake")):
        return summary
    if response_kind == 'forcing_trade' and "trade" not in lower:
        return summary
    if response_kind == 'piece_retreated' and "retreat" not in lower:
        return summary
    if response_kind == 'active_counter' and not any(token in lower for token in ("counter", "active", "jump")):
        return summary
    if response_kind == 'ignored_threat' and not any(token in lower for token in ("ignored", "threat", "attack")):
        return summary
    return commentary
