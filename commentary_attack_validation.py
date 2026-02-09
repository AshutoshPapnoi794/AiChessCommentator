from typing import Any, Dict, Optional

import chess

from analysis_socket import commentary_engines, commentary_locks
from commentary_attack import PIECE_VALUES, infer_played_move, same_position_state
from commentary_engine import engine_best_line_san, engine_eval_cp_after_move_for_color, engine_eval_cp_for_board

TACTICAL_PROTECTION_DROP_CP = 80
ATTACK_BLUNDER_SAFE_CP = 40
TRAPPED_CONFIRM_LOSS_CP = 120
TRAP_CAPTURE_TOLERANCE_CP = 40


def _cp_for_color(cp_white: int, color: chess.Color) -> int:
    return cp_white if color == chess.WHITE else -cp_white


def _is_move_tactically_safe_for_attacker_capture(
    sid: str,
    board_after_target_move: chess.Board,
    attacker_color: chess.Color,
    target_square: int,
) -> bool:
    if board_after_target_move.turn != attacker_color:
        return False

    capture_moves = [
        move
        for move in board_after_target_move.legal_moves
        if move.to_square == target_square and board_after_target_move.is_capture(move)
    ]
    if not capture_moves:
        return False

    cp_before = engine_eval_cp_for_board(sid, board_after_target_move, depth=10)
    if cp_before is None:
        return False
    cp_before_for_attacker = _cp_for_color(cp_before, attacker_color)

    best_capture_cp = None
    for move in capture_moves:
        cp_after = engine_eval_cp_after_move_for_color(
            sid,
            board_after_target_move,
            move,
            attacker_color,
            depth=12,
        )
        if cp_after is None:
            continue
        if best_capture_cp is None or cp_after > best_capture_cp:
            best_capture_cp = cp_after

    if best_capture_cp is None:
        return False

    # Capturing is considered tactically sound if it does not significantly worsen the attacker's eval.
    return best_capture_cp >= (cp_before_for_attacker - TRAP_CAPTURE_TOLERANCE_CP)


def _has_any_safe_escape_square(
    sid: str,
    post_board: chess.Board,
    attacker_color: chess.Color,
    target_sq: int,
    target_moves: list[chess.Move],
) -> bool:
    defender_color = not attacker_color

    for move in target_moves:
        board_after_target_move = post_board.copy(stack=False)
        board_after_target_move.turn = defender_color
        if move not in board_after_target_move.legal_moves:
            continue
        board_after_target_move.push(move)

        moved_piece = board_after_target_move.piece_at(move.to_square)
        if moved_piece is None or moved_piece.color != defender_color:
            continue

        board_after_target_move.turn = attacker_color
        attacker_has_capture = any(
            m.to_square == move.to_square and board_after_target_move.is_capture(m)
            for m in board_after_target_move.legal_moves
        )
        if not attacker_has_capture:
            return True

        if not _is_move_tactically_safe_for_attacker_capture(
            sid,
            board_after_target_move,
            attacker_color,
            move.to_square,
        ):
            return True

    return False


def validate_hanging_target_with_stockfish(
    sid: str,
    prev_board: chess.Board,
    current_board: chess.Board,
    move_hint_san: Optional[str],
    attackedby_data: Dict[str, Any],
) -> Dict[str, Any]:
    if attackedby_data.get('kind') not in {'attack', 'attack_trapped', 'attack_blunder', 'pressure'}:
        return attackedby_data
    if not commentary_engines.get(sid) or not commentary_locks.get(sid):
        return attackedby_data

    targets = attackedby_data.get('targets') or []
    top = targets[0] if targets else None
    if not top:
        return attackedby_data

    played_move = infer_played_move(prev_board, current_board, move_hint_san)
    if not played_move:
        return attackedby_data

    trial_post = prev_board.copy(stack=False)
    trial_post.push(played_move)
    post_board = current_board if same_position_state(trial_post, current_board) else trial_post

    attacker = post_board.piece_at(played_move.to_square)
    if attacker is None:
        return attackedby_data

    target_square_name = top.get('square')
    if not target_square_name:
        return attackedby_data

    try:
        target_sq = chess.parse_square(target_square_name)
    except ValueError:
        return attackedby_data

    validation = attackedby_data.setdefault('engine_validation', {})

    # 0) Trap confirmation with engine:
    # trapped if target has no legal move from its square, or every such move is heavily losing.
    target_probe = post_board.copy(stack=False)
    target_probe.turn = not attacker.color
    target_moves = [m for m in target_probe.legal_moves if m.from_square == target_sq]
    trap_confirmed = False
    best_target_move_san = None
    best_target_delta_cp = None

    cp_before_target_white = engine_eval_cp_for_board(sid, target_probe, depth=10)
    if cp_before_target_white is not None:
        cp_before_target = _cp_for_color(cp_before_target_white, not attacker.color)
        move_deltas = []
        for move in target_moves:
            after = target_probe.copy(stack=False)
            after.push(move)
            cp_after_white = engine_eval_cp_for_board(sid, after, depth=12)
            if cp_after_white is None:
                continue
            cp_after = _cp_for_color(cp_after_white, not attacker.color)
            delta = cp_after - cp_before_target
            try:
                san = target_probe.san(move)
            except ValueError:
                san = move.uci()
            move_deltas.append((delta, san))

        if not target_moves:
            trap_confirmed = True
        elif move_deltas:
            best_delta, best_san = max(move_deltas, key=lambda x: x[0])
            best_target_move_san = best_san
            best_target_delta_cp = best_delta
            trap_confirmed = all(delta <= -TRAPPED_CONFIRM_LOSS_CP for delta, _ in move_deltas)

        # Additional trap rule:
        # if there are no safe retreat squares and every target move can still be met by a tactically
        # sound immediate capture, treat the piece as trapped.
        static_no_safe_retreat = (top.get('safe_retreat_count') == 0)
        has_safe_escape = False
        if target_moves and static_no_safe_retreat:
            has_safe_escape = _has_any_safe_escape_square(
                sid=sid,
                post_board=post_board,
                attacker_color=attacker.color,
                target_sq=target_sq,
                target_moves=target_moves,
            )
            if not has_safe_escape:
                trap_confirmed = True

        validation.update({
            'trap_confirmed': trap_confirmed,
            'trap_best_target_move': best_target_move_san,
            'trap_best_target_delta_cp': best_target_delta_cp,
            'trap_static_no_safe_retreat': static_no_safe_retreat if target_moves else None,
            'trap_has_safe_escape': has_safe_escape if target_moves else None,
        })

    if trap_confirmed and attackedby_data.get('kind') in {'attack', 'pressure'}:
        attacker_name = chess.piece_name(attacker.piece_type)
        attackedby_data['kind'] = 'attack_trapped'
        attackedby_data['summary'] = (
            f"{attacker_name.capitalize()} on {chess.square_name(played_move.to_square)} attacks a trapped "
            f"{top['piece_name']} on {target_square_name}."
        )
    elif attackedby_data.get('kind') == 'attack_trapped' and not trap_confirmed:
        # Conservative downgrade if trap not engine-confirmed.
        attackedby_data['kind'] = 'attack'
        attackedby_data['summary'] = (
            f"{chess.piece_name(attacker.piece_type).capitalize()} on {chess.square_name(played_move.to_square)} "
            f"attacks {top['piece_name']} on {target_square_name}."
        )

    # 1) Target captures attacker check (detect bad pseudo-attacks on higher-value pieces).
    target_capture_move = None
    for move in target_probe.legal_moves:
        if move.from_square == target_sq and move.to_square == played_move.to_square and target_probe.is_capture(move):
            target_capture_move = move
            break

    if target_capture_move is not None:
        board_after_target_capture = target_probe.copy(stack=False)
        board_after_target_capture.push(target_capture_move)

        cp_before_target = engine_eval_cp_for_board(sid, target_probe, depth=10)
        cp_after_target_capture = engine_eval_cp_for_board(sid, board_after_target_capture, depth=12)
        if cp_before_target is not None and cp_after_target_capture is not None:
            target_capture_delta_cp = (
                (cp_after_target_capture - cp_before_target)
                if (not attacker.color) == chess.WHITE
                else (cp_before_target - cp_after_target_capture)
            )
            target_capture_line = engine_best_line_san(sid, board_after_target_capture, max_plies=4, depth=12)
            try:
                target_capture_san = target_probe.san(target_capture_move)
            except ValueError:
                target_capture_san = target_capture_move.uci()

            validation.update({
                'target_capture_delta_cp': target_capture_delta_cp,
                'target_capture_san': target_capture_san,
                'target_capture_line': target_capture_line,
            })

            attacker_value = PIECE_VALUES.get(attacker.piece_type, 0)
            target_value = PIECE_VALUES.get(top.get('piece_type') or 0, 0)
            if target_value > attacker_value and target_capture_delta_cp >= -ATTACK_BLUNDER_SAFE_CP:
                attackedby_data['kind'] = 'attack_blunder'
                attackedby_data['summary'] = (
                    f"That attack is unsound: the {top['piece_name']} on {target_square_name} can capture the attacker safely "
                    f"({target_capture_san})."
                )
                return attackedby_data

    # 2) Attacker captures target check (detect tactical protection with a concrete refutation line).
    attacker_probe = post_board.copy(stack=False)
    attacker_probe.turn = attacker.color
    attacker_capture_move = None
    for move in attacker_probe.legal_moves:
        if move.from_square == played_move.to_square and move.to_square == target_sq and attacker_probe.is_capture(move):
            attacker_capture_move = move
            break

    if attacker_capture_move is None:
        return attackedby_data

    board_after_attacker_capture = attacker_probe.copy(stack=False)
    board_after_attacker_capture.push(attacker_capture_move)

    cp_before_capture = engine_eval_cp_for_board(sid, attacker_probe, depth=10)
    cp_after_capture = engine_eval_cp_for_board(sid, board_after_attacker_capture, depth=12)
    if cp_before_capture is None or cp_after_capture is None:
        return attackedby_data

    capture_delta_cp = (
        (cp_after_capture - cp_before_capture)
        if attacker.color == chess.WHITE
        else (cp_before_capture - cp_after_capture)
    )
    refutation_line = engine_best_line_san(sid, board_after_attacker_capture, max_plies=4, depth=12)

    try:
        capture_san = attacker_probe.san(attacker_capture_move)
    except ValueError:
        capture_san = attacker_capture_move.uci()

    validation.update({
        'cp_before_capture': cp_before_capture,
        'cp_after_capture': cp_after_capture,
        'capture_delta_cp': capture_delta_cp,
        'capture_san': capture_san,
        'refutation_line': refutation_line,
    })

    if attackedby_data.get('kind') == 'attack_trapped':
        attacker_name = chess.piece_name(attacker.piece_type)
        trapped_summary = (
            f"{attacker_name.capitalize()} on {chess.square_name(played_move.to_square)} attacks a trapped "
            f"{top['piece_name']} on {target_square_name}; it has no safe retreat."
        )
        if capture_delta_cp >= 80:
            trapped_summary += " The piece is not tactically defended and capturing it wins clear material."
        elif capture_delta_cp >= -ATTACK_BLUNDER_SAFE_CP:
            trapped_summary += " The piece is not tactically defended and capturing it is tactically sound."
        attackedby_data['summary'] = trapped_summary

    if capture_delta_cp <= -TACTICAL_PROTECTION_DROP_CP:
        attacker_name = chess.piece_name(attacker.piece_type)
        attackedby_data['kind'] = 'tactically_protected'
        attackedby_data['summary'] = (
            f"{attacker_name.capitalize()} on {chess.square_name(played_move.to_square)} appears to attack "
            f"{top['piece_name']} on {target_square_name}, but it is tactically defended. "
            f"If {capture_san}, then {refutation_line or 'the tactical reply strongly punishes the capture'}."
        )

    return attackedby_data


def enforce_attack_pressure_consistency(
    commentary: str,
    attackedby_data: Dict[str, Any],
    trade_kind: str = "none",
) -> str:
    kind = attackedby_data.get('kind')
    targets = attackedby_data.get('targets') or []
    top = targets[0] if targets else None
    lower = (commentary or "").lower()
    has_attack_word = "attack" in lower or "attacking" in lower or "threat" in lower
    has_pressure_word = "pressur" in lower

    if kind is None:
        if trade_kind not in {"", "none", None}:
            return commentary
        if has_attack_word or has_pressure_word:
            return "No immediate attack or pressure is created by that move."
        return commentary

    if kind == 'tactically_protected':
        validation = attackedby_data.get('engine_validation') or {}
        capture_san = validation.get('capture_san')
        refutation_line = validation.get('refutation_line')
        if top and capture_san and refutation_line:
            return (
                f"It looks like an attack on {top['piece_name']} {top['square']}, but it is tactically protected: "
                f"after {capture_san}, {refutation_line}."
            )
        if top:
            return f"It looks like an attack on {top['piece_name']} {top['square']}, but the piece is tactically protected."
        return "The target looks hanging, but tactical resources make the capture unsound."

    if kind == 'attack_blunder':
        validation = attackedby_data.get('engine_validation') or {}
        cap = validation.get('target_capture_san')
        line = validation.get('target_capture_line')
        if top and cap and line:
            return (
                f"This attack is a blunder: {top['piece_name']} {top['square']} can safely take it with {cap}, "
                f"then {line}."
            )
        if top:
            return f"This attack is unsound; the {top['piece_name']} on {top['square']} can capture back safely."
        return "This attack is unsound because the target can capture back safely."

    if kind == 'attack':
        if not has_attack_word or has_pressure_word:
            if top:
                if top.get('is_hanging'):
                    return f"That move attacks a hanging {top['piece_name']} on {top['square']}, creating an immediate tactical threat."
                if top.get('capture_is_losing'):
                    return f"That move attacks the defended {top['piece_name']} on {top['square']}; capturing it back would lose material."
                return f"That move attacks the defended {top['piece_name']} on {top['square']}."
            return "That move creates a direct attack on an enemy piece."
        return commentary

    if kind == 'attack_trapped':
        validation = attackedby_data.get('engine_validation') or {}
        capture_delta_cp = validation.get('capture_delta_cp')
        if top:
            canonical = f"That move attacks a trapped {top['piece_name']} on {top['square']}; it has no safe retreat squares."
        else:
            canonical = "That move attacks a trapped enemy piece with no safe retreat."
        if capture_delta_cp is not None:
            if capture_delta_cp >= 80:
                canonical += " The piece is not tactically defended and can be won."
            elif capture_delta_cp >= -ATTACK_BLUNDER_SAFE_CP:
                canonical += " The piece is not tactically defended; capturing it is sound."
        if "trapped" not in lower or not has_attack_word or has_pressure_word:
            return canonical
        return commentary

    if kind == 'pressure':
        if top:
            canonical = f"That move pressures the defended {top['piece_name']} on {top['square']}, increasing positional tension."
        else:
            canonical = "That move increases pressure on a defended enemy piece."
        if not has_pressure_word or has_attack_word:
            return canonical
        return commentary

    contradiction = any(
        token in lower
        for token in (
            "not attacking",
            "not pressuring",
            "no attack",
            "no pressure",
            "not threatening",
        )
    )
    if not contradiction:
        return commentary

    return commentary
