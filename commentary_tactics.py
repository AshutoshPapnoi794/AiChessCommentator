import re
from typing import Any, Dict, List, Optional, Tuple

import chess

from commentary_attack import infer_played_move
from commentary_engine import engine_eval_cp_after_move_for_color, engine_eval_cp_for_color
from runtime import logger

PIN_RELATIVE_CAPTURE_LOSS_CP = 80


def _pin_key(pin: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(pin.get('type', '')),
        str(pin.get('pinner_square', '')),
        str(pin.get('pinned_square', '')),
        str(pin.get('valuable_piece_square', '')),
    )


def _piece_label(symbol: Any) -> str:
    try:
        piece = chess.Piece.from_symbol(str(symbol))
    except ValueError:
        return str(symbol or "piece")
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color} {chess.piece_name(piece.piece_type)}"


def _choose_capture_move_if_legal(
    board_for_pinned_side: chess.Board,
    from_sq: int,
    to_sq: int,
    piece_type: int,
) -> Optional[chess.Move]:
    if piece_type == chess.PAWN and chess.square_rank(to_sq) in {0, 7}:
        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            move = chess.Move(from_sq, to_sq, promotion=promo)
            if move in board_for_pinned_side.legal_moves:
                return move
        return None

    move = chess.Move(from_sq, to_sq)
    if move in board_for_pinned_side.legal_moves:
        return move
    return None


def _find_pin_exploitations(
    sid: str,
    current_board: chess.Board,
    played_move: chess.Move,
    pins_curr: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    mover = current_board.piece_at(played_move.to_square)
    if mover is None:
        return []

    mover_color = mover.color
    events: List[Dict[str, Any]] = []

    for pin in pins_curr:
        pinned_sq_name = pin.get('pinned_square')
        if not pinned_sq_name:
            continue
        try:
            pinned_sq = chess.parse_square(pinned_sq_name)
        except ValueError:
            continue

        pinned_piece = current_board.piece_at(pinned_sq)
        if not pinned_piece or pinned_piece.color == mover_color:
            continue

        if played_move.to_square not in current_board.attacks(pinned_sq):
            continue

        probe = current_board.copy(stack=False)
        probe.turn = pinned_piece.color
        capture_move = _choose_capture_move_if_legal(
            board_for_pinned_side=probe,
            from_sq=pinned_sq,
            to_sq=played_move.to_square,
            piece_type=pinned_piece.piece_type,
        )

        if capture_move is None:
            reason = "absolute pin" if str(pin.get('type')) == "Absolute" else "pin constraint"
            events.append({
                'reason': reason,
                'pin': pin,
                'pinned_piece': pinned_piece,
                'pinned_square': chess.square_name(pinned_sq),
                'landed_square': chess.square_name(played_move.to_square),
            })
            continue

        if str(pin.get('type')) == "Relative":
            cp_before = engine_eval_cp_for_color(sid, probe, pinned_piece.color, depth=10)
            cp_after = engine_eval_cp_after_move_for_color(sid, probe, capture_move, pinned_piece.color, depth=12)
            if cp_before is not None and cp_after is not None and (cp_after - cp_before) <= -PIN_RELATIVE_CAPTURE_LOSS_CP:
                events.append({
                    'reason': "relative pin loss",
                    'pin': pin,
                    'pinned_piece': pinned_piece,
                    'pinned_square': chess.square_name(pinned_sq),
                    'landed_square': chess.square_name(played_move.to_square),
                })

    return events


def detect_tactical_events(
    sid: str,
    prev_board: chess.Board,
    current_board: chess.Board,
    move_hint_san: Optional[str],
    tactics_payload: Dict[str, Any],
    pins_prev: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pins_curr = (tactics_payload.get('static_analysis', {}) or {}).get('pins') or []
    forks_curr = (tactics_payload.get('static_analysis', {}) or {}).get('forks') or []
    analysis_of_move = tactics_payload.get('analysis_of_move', {}) or {}
    has_discovered_attack = bool(analysis_of_move.get('discovered_attacks'))

    prev_keys = {_pin_key(pin) for pin in pins_prev}
    created_pins = [pin for pin in pins_curr if _pin_key(pin) not in prev_keys]

    played_move = infer_played_move(prev_board, current_board, move_hint_san)
    exploited_pins: List[Dict[str, Any]] = []
    created_forks: List[Dict[str, Any]] = []
    if played_move:
        exploited_pins = _find_pin_exploitations(sid, current_board, played_move, pins_curr)
        to_sq_name = chess.square_name(played_move.to_square)
        created_forks = [fork for fork in forks_curr if str(fork.get('forking_piece_square')) == to_sq_name]

    if exploited_pins:
        top = exploited_pins[0]
        pin = top['pin']
        pinner_piece = _piece_label(pin.get('pinner_piece'))
        pinned_piece = _piece_label(pin.get('pinned_piece'))
        summary = (
            f"Pin exploited: {pinner_piece} keeps {pinned_piece} on {top['pinned_square']} pinned, "
            f"so the piece on {top['landed_square']} cannot be captured safely ({top['reason']})."
        )
        return {
            'kind': 'pin_exploited',
            'summary': summary,
            'created_pins': created_pins,
            'created_forks': created_forks,
            'exploited_pins': exploited_pins,
            'has_discovered_attack': has_discovered_attack,
        }

    if created_forks:
        top_fork = created_forks[0]
        forking_piece = _piece_label(top_fork.get('forking_piece'))
        target_labels: List[str] = []
        for target in top_fork.get('targets', [])[:3]:
            try:
                target_labels.append(f"{_piece_label(target.get('piece'))} on {target.get('square')}")
            except Exception:
                continue
        target_text = ", ".join(target_labels) if target_labels else "multiple valuable pieces"
        winning_targets = top_fork.get('winning_targets') or []
        if winning_targets:
            winning_text = ", ".join(str(sq) for sq in winning_targets[:3])
            summary = (
                f"Fork created: {forking_piece} on {top_fork.get('forking_piece_square')} attacks {target_text}; "
                f"at least one target cannot be saved in one move ({winning_text})."
            )
        else:
            summary = (
                f"Fork created: {forking_piece} on {top_fork.get('forking_piece_square')} attacks {target_text}."
            )
        return {
            'kind': 'fork_created',
            'summary': summary,
            'created_pins': created_pins,
            'created_forks': created_forks,
            'exploited_pins': exploited_pins,
            'has_discovered_attack': has_discovered_attack,
        }

    if created_pins:
        pin = created_pins[0]
        summary = (
            f"New {str(pin.get('type', 'pin')).lower()} pin created: "
            f"{_piece_label(pin.get('pinner_piece'))} pins "
            f"{_piece_label(pin.get('pinned_piece'))} on {pin.get('pinned_square')} "
            f"to {_piece_label(pin.get('valuable_piece'))}."
        )
        return {
            'kind': 'pin_created',
            'summary': summary,
            'created_pins': created_pins,
            'created_forks': created_forks,
            'exploited_pins': exploited_pins,
            'has_discovered_attack': has_discovered_attack,
        }

    if has_discovered_attack:
        return {
            'kind': 'discovered_attack',
            'summary': "Discovered attack motif created.",
            'created_pins': created_pins,
            'created_forks': created_forks,
            'exploited_pins': exploited_pins,
            'has_discovered_attack': True,
        }

    return {
        'kind': 'none',
        'summary': "None",
        'created_pins': created_pins,
        'created_forks': created_forks,
        'exploited_pins': exploited_pins,
        'has_discovered_attack': has_discovered_attack,
    }


def build_tactics_signal(
    sid: str,
    prev_board: chess.Board,
    current_board: chess.Board,
    move_hint_san: Optional[str],
    tactics_analyzer: Any,
    lock: Any = None,
) -> str:
    if not tactics_analyzer:
        return "None"

    try:
        pins_prev = tactics_analyzer._detect_positional_pins(prev_board)  # pylint: disable=protected-access
        if lock:
            with lock:
                payload = tactics_analyzer.analyze(current_board.fen(), prev_board.fen())
        else:
            payload = tactics_analyzer.analyze(current_board.fen(), prev_board.fen())
        events = detect_tactical_events(
            sid=sid,
            prev_board=prev_board,
            current_board=current_board,
            move_hint_san=move_hint_san,
            tactics_payload=payload or {},
            pins_prev=pins_prev or [],
        )
        return str(events.get('summary') or "None")
    except Exception as exc:
        logger.debug("Tactical event signal failed: %s", exc)
        return "None"


def enforce_tactics_event_consistency(commentary: str, tactics_signal: str) -> str:
    text = (commentary or "").strip()
    if not text:
        return text

    if (tactics_signal or "").strip().lower() != "none":
        return text

    lower = text.lower()
    if not (
        re.search(r"\bpin\b", lower)
        or re.search(r"\bpinned\b", lower)
        or re.search(r"\bdiscovered attack\b", lower)
        or re.search(r"\bfork\b", lower)
        or re.search(r"\bforking\b", lower)
        or re.search(r"\bdouble attack\b", lower)
    ):
        return text

    cleaned = text
    cleaned = re.sub(r",?\s*as the pin[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*while the pin[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*with the pin[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*as the [^.,;]*pinned[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*while the [^.,;]*pinned[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*with the [^.,;]*pinned[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*as a discovered attack[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*while a discovered attack[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*as the fork[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*while the fork[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*with the fork[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*as the double attack[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*while the double attack[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;")

    if not cleaned:
        return "A useful move."
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned
