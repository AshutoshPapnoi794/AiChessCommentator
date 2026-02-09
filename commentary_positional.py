import logging
from typing import List, Optional

import chess

from commentary_attack import infer_played_move

logger = logging.getLogger(__name__)

CORE_CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
KEY_CONTROL_SQUARES = {
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.D4,
    chess.E4,
    chess.F4,
    chess.C5,
    chess.D5,
    chess.E5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
}


def _format_square_list(squares: List[int], limit: int = 2) -> str:
    if not squares:
        return ""
    names = [chess.square_name(sq) for sq in squares[:limit]]
    if len(names) == 1:
        return names[0]
    return " and ".join(names)


def _opened_slider_lines(
    prev_board: chess.Board,
    current_board: chess.Board,
    moved_from_square: int,
    mover_color: chess.Color,
) -> List[str]:
    opened: List[str] = []
    for piece_type in (chess.QUEEN, chess.BISHOP, chess.ROOK):
        for slider_sq in prev_board.pieces(piece_type, mover_color):
            if slider_sq == moved_from_square:
                continue
            prev_slider = prev_board.piece_at(slider_sq)
            curr_slider = current_board.piece_at(slider_sq)
            if not prev_slider or not curr_slider:
                continue
            if curr_slider.color != mover_color or curr_slider.piece_type != prev_slider.piece_type:
                continue

            prev_attacks = prev_board.attacks(slider_sq)
            if moved_from_square not in prev_attacks:
                continue
            curr_attacks = current_board.attacks(slider_sq)
            if curr_attacks - prev_attacks:
                opened.append(f"{chess.piece_name(prev_slider.piece_type)} on {chess.square_name(slider_sq)}")

    dedup: List[str] = []
    for item in opened:
        if item not in dedup:
            dedup.append(item)
    return dedup[:2]


def _is_development_move(piece: chess.Piece, from_sq: int, to_sq: int, color: chess.Color) -> bool:
    from_rank = chess.square_rank(from_sq)
    to_rank = chess.square_rank(to_sq)
    if piece.piece_type == chess.KNIGHT:
        home_rank = 0 if color == chess.WHITE else 7
        return from_rank == home_rank and to_rank != home_rank
    if piece.piece_type == chess.BISHOP:
        home_rank = 0 if color == chess.WHITE else 7
        return from_rank == home_rank and to_rank != home_rank
    return False


def build_positional_signal(
    prev_board: chess.Board,
    current_board: chess.Board,
    move_hint_san: Optional[str] = None,
) -> str:
    played_move = infer_played_move(prev_board, current_board, move_hint_san)
    if not played_move:
        return "None"

    moved_piece_before = prev_board.piece_at(played_move.from_square)
    moved_piece_after = current_board.piece_at(played_move.to_square)
    if moved_piece_before is None or moved_piece_after is None:
        return "None"

    mover_color = moved_piece_after.color
    segments: List[str] = []

    if prev_board.is_castling(played_move):
        return "Improves king safety by castling and helps rook coordination."

    if moved_piece_before.piece_type == chess.PAWN:
        to_sq = played_move.to_square
        to_name = chess.square_name(to_sq)
        if to_sq in CORE_CENTER:
            segments.append(f"claims central space with {to_name}")

        controlled = [sq for sq in current_board.attacks(to_sq) if sq in KEY_CONTROL_SQUARES]
        controlled_text = _format_square_list(sorted(controlled))
        if controlled_text:
            segments.append(f"controls {controlled_text}")

        opened_lines = _opened_slider_lines(prev_board, current_board, played_move.from_square, mover_color)
        if opened_lines:
            if len(opened_lines) == 1:
                segments.append(f"opens a line for {opened_lines[0]}")
            else:
                segments.append(f"opens lines for {opened_lines[0]} and {opened_lines[1]}")

    if _is_development_move(moved_piece_before, played_move.from_square, played_move.to_square, mover_color):
        piece_name = chess.piece_name(moved_piece_before.piece_type)
        to_name = chess.square_name(played_move.to_square)
        segments.append(f"develops the {piece_name} to {to_name}")

    if moved_piece_before.piece_type == chess.ROOK:
        file_idx = chess.square_file(played_move.to_square)
        file_squares = [chess.square(file_idx, rank) for rank in range(8)]
        if not any(
            (piece := current_board.piece_at(sq)) and piece.piece_type == chess.PAWN
            for sq in file_squares
        ):
            segments.append(f"places the rook on an open {chr(ord('a') + file_idx)}-file")

    if not segments:
        return "None"

    text = "; ".join(segments)
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def enforce_positional_signal_consistency(
    commentary: str,
    positional_signal: str,
    attack_kind: str,
    trade_kind: str,
    forced_label: str,
    tactics_signal: str,
    opening_update: str = "None",
    forcing_signal: str = "None",
    threat_signal: str = "None",
    threat_response_signal: str = "None",
    conceded_signal: str = "None",
) -> str:
    text = (commentary or "").strip()
    if not text:
        text = ""

    signal = (positional_signal or "").strip()
    if not signal or signal.lower() == "none":
        return text

    # Let tactical/trade/forced moments dominate the sentence.
    if attack_kind not in {"", "none"}:
        return text
    if trade_kind not in {"", "none"}:
        return text
    if forced_label not in {"", "none", None}:
        return text
    if (tactics_signal or "").strip().lower() != "none":
        return text
    if (opening_update or "").strip().lower() != "none":
        return text
    if (forcing_signal or "").strip().lower() != "none":
        return text
    if (threat_signal or "").strip().lower() != "none":
        return text
    if (threat_response_signal or "").strip().lower() != "none":
        return text
    if (conceded_signal or "").strip().lower() != "none":
        return text

    lower = text.lower()
    if any(token in lower for token in ("center", "central", "develop", "opens", "open file", "king safety", "castl")):
        return text

    try:
        if not text:
            return signal
        return signal
    except Exception as exc:
        logger.debug("Positional signal enforcement failed: %s", exc)
        return text


def compact_positional_signal_for_prompt(signal: str) -> str:
    text = (signal or "").strip()
    if not text or text.lower() == "none":
        return "None"
    raw = text.strip(". ")
    lowered = raw.lower()
    if lowered == "improves king safety by castling and helps rook coordination":
        return "castle=king safety, rook coordination"

    segments = [seg.strip() for seg in raw.split(";") if seg.strip()]
    compact_segments: List[str] = []
    for seg in segments:
        low = seg.lower()
        if low.startswith("claims central space with "):
            compact_segments.append("center=" + seg[len("claims central space with "):].strip())
        elif low.startswith("controls "):
            squares = seg[len("controls "):].replace(" and ", ",")
            compact_segments.append("controls=" + squares)
        elif low.startswith("opens lines for "):
            pieces = seg[len("opens lines for "):]
            compact_segments.append("opens=" + pieces)
        elif low.startswith("opens a line for "):
            piece = seg[len("opens a line for "):]
            compact_segments.append("opens=" + piece)
        elif low.startswith("develops the "):
            compact_segments.append("develops=" + seg[len("develops the "):])
        elif low.startswith("places the rook on an open "):
            compact_segments.append("rook_file=" + seg[len("places the rook on an open "):])
        else:
            compact_segments.append(seg)

    compact = ", ".join(compact_segments).strip()
    return compact or raw
