from typing import Any, Dict, Optional

import chess

from commentary_attack import PIECE_VALUES, infer_played_move, same_position_state


def _is_open_file(board: chess.Board, file_idx: int) -> bool:
    for rank in range(8):
        sq = chess.square(file_idx, rank)
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN:
            return False
    return True


def detect_trade_context(
    prev_board: chess.Board,
    current_board: chess.Board,
    move_hint_san: Optional[str] = None,
) -> Dict[str, Any]:
    played_move = infer_played_move(prev_board, current_board, move_hint_san)
    if not played_move:
        return {'kind': None, 'summary': ''}

    trial_post = prev_board.copy(stack=False)
    trial_post.push(played_move)
    post_board = current_board if same_position_state(trial_post, current_board) else trial_post

    mover_piece = prev_board.piece_at(played_move.from_square)
    captured_piece = prev_board.piece_at(played_move.to_square) if prev_board.is_capture(played_move) else None
    if mover_piece is None or captured_piece is None:
        return {'kind': None, 'summary': ''}

    mover_color = mover_piece.color
    opponent_color = not mover_color
    mover_name = "White" if mover_color == chess.WHITE else "Black"
    to_sq_name = chess.square_name(played_move.to_square)

    if captured_piece.piece_type != chess.QUEEN:
        # Center pawn tension release / pawn trade initiation.
        if (
            mover_piece.piece_type == chess.PAWN
            and captured_piece.piece_type == chess.PAWN
            and chess.square_file(played_move.to_square) in {3, 4}  # d/e files
        ):
            to_sq_name = chess.square_name(played_move.to_square)
            summary = (
                f"{mover_name} captures in the center on {to_sq_name}, "
                f"releasing central tension and initiating a pawn trade."
            )
            return {'kind': 'center_pawn_trade', 'summary': summary}
        return {'kind': None, 'summary': ''}

    post_probe = post_board.copy(stack=False)
    post_probe.turn = opponent_color
    recapture_moves = [
        m
        for m in post_probe.legal_moves
        if m.to_square == played_move.to_square and post_probe.is_capture(m)
    ]
    recapture_piece_values = []
    for move in recapture_moves:
        recapturer = post_probe.piece_at(move.from_square)
        if recapturer:
            recapture_piece_values.append(PIECE_VALUES.get(recapturer.piece_type, 99))
    immediate_reasonable_recapture = bool(recapture_piece_values) and min(recapture_piece_values) <= PIECE_VALUES[chess.QUEEN]

    white_queens_after = len(post_board.pieces(chess.QUEEN, chess.WHITE))
    black_queens_after = len(post_board.pieces(chess.QUEEN, chess.BLACK))
    both_queens_off = white_queens_after == 0 and black_queens_after == 0

    if both_queens_off:
        file_idx = chess.square_file(played_move.to_square)
        file_name = chr(ord('a') + file_idx)
        if mover_piece.piece_type == chess.ROOK:
            if _is_open_file(post_board, file_idx):
                summary = (
                    f"{mover_name} recaptures the queen with the rook on {to_sq_name}, "
                    f"and now controls the open {file_name}-file."
                )
            else:
                summary = f"{mover_name} recaptures the queen with the rook on {to_sq_name}, completing the queen trade."
        else:
            summary = f"{mover_name} recaptures the queen on {to_sq_name}, and queens are traded off the board."
        return {'kind': 'queen_trade_completed', 'summary': summary}

    if mover_piece.piece_type == chess.QUEEN and immediate_reasonable_recapture:
        summary = (
            f"{mover_name} trades queens with {move_hint_san or played_move.uci()}; "
            f"the queen on {to_sq_name} is immediately recapturable."
        )
        return {'kind': 'queen_trade_initiated', 'summary': summary}

    return {'kind': None, 'summary': ''}


def enforce_trade_consistency(commentary: str, trade_data: Dict[str, Any]) -> str:
    kind = trade_data.get('kind')
    summary = (trade_data.get('summary') or '').strip()
    if not kind or not summary:
        return commentary

    lower = (commentary or "").lower()

    if kind == 'center_pawn_trade':
        lower = (commentary or "").lower()
        if not any(token in lower for token in ("center", "central", "pawn trade", "tension", "captures")):
            return summary
        return commentary

    if kind == 'queen_trade_completed':
        return summary

    if kind == 'queen_trade_initiated':
        bad_tokens = (
            "hanging rook",
            "undefended rook",
            "hanging",
            "undefended",
            "attacks the rook",
            "attacks rook",
        )
        if any(token in lower for token in bad_tokens):
            return summary
        if "queen" not in lower or "trade" not in lower:
            return summary

    return commentary
