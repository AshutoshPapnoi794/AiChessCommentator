import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess

from analysis_socket import tactics_lock
from commentary_attack import Attackedby
from commentary_attack_validation import enforce_attack_pressure_consistency, validate_hanging_target_with_stockfish
from commentary_engine import get_commentary_eval
from commentary_forced import (
    assess_forced_response_to_hanging_attack,
    enforce_forced_response_consistency,
    update_pending_forced_response,
)
from commentary_momentum import enforce_momentum_consistency
from commentary_move_events import (
    build_capture_check_signal,
    build_forcing_signal,
    enforce_move_event_consistency,
)
from commentary_opening import build_opening_update, enforce_opening_update_consistency
from commentary_positional import (
    build_positional_signal,
    compact_positional_signal_for_prompt,
    enforce_positional_signal_consistency,
)
from commentary_tactics import build_tactics_signal, enforce_tactics_event_consistency
from commentary_text import ensure_sentence_punctuation, is_noneish_text, sanitize_commentary_text
from commentary_threats import (
    assess_threat_response_to_pending,
    build_conceded_threat_signal,
    build_threat_signal,
    enforce_conceded_threat_consistency,
    enforce_threat_consistency,
    enforce_threat_response_consistency,
    update_pending_threat_response,
)
from commentary_trade import detect_trade_context, enforce_trade_consistency
from narrative import GameNarrativeMemory
from runtime import STOCKFISH_PATH, logger, tactics_analyzer


ProgressCallback = Optional[Callable[[int, str], None]]


def _emit_progress(progress_cb: ProgressCallback, pct: int, message: str) -> None:
    if progress_cb:
        progress_cb(max(0, min(100, int(pct))), message)


def _clamp_words(text: str, max_words: int = 30) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,.;:") + "."


def _safe_text(value: Any, fallback: str = "None") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    return re.sub(r"\s+", " ", text)


def _is_noneish(value: Any) -> bool:
    return is_noneish_text(_safe_text(value, fallback="None"))


def _to_san_moves(moves: List[Any]) -> List[str]:
    san_moves: List[str] = []
    for mv in moves:
        if isinstance(mv, dict):
            san = mv.get("san")
        else:
            san = mv
        san_text = _safe_text(san, fallback="")
        if san_text:
            san_moves.append(san_text)
    return san_moves


def _move_quality_from_eval_delta(eval_prev: int, eval_curr: int, is_white_move: bool) -> str:
    diff = (eval_curr - eval_prev) if is_white_move else (eval_prev - eval_curr)
    if diff < -200:
        return "Blunder"
    if diff < -80:
        return "Mistake"
    if diff > 30:
        return "Good/Improves Position"
    return "Neutral"


def _fallback_commentary_line(context: Dict[str, Any]) -> str:
    trade = context.get('trade_data') or {}
    trade_summary = _safe_text(trade.get('summary'), fallback="")
    if trade.get('kind') in {'queen_trade_initiated', 'queen_trade_completed', 'center_pawn_trade'} and trade_summary:
        return _clamp_words(trade_summary, max_words=30)

    forced = context.get('forced_response') or {}
    forced_summary = _safe_text(forced.get('summary'), fallback="")
    forced_label = forced.get('label')
    if forced_label in {'blunder', 'mistake'} and forced_summary:
        return _clamp_words(forced_summary, max_words=35)
    threat_response = context.get('threat_response') or {}
    threat_response_summary = _safe_text(threat_response.get('summary'), fallback="")
    threat_response_label = threat_response.get('label')
    if threat_response_label in {'blunder', 'mistake', 'fine'} and threat_response_summary:
        return _clamp_words(threat_response_summary, max_words=35)
    conceded_summary = _safe_text(context.get('conceded_signal'), fallback="None")
    if conceded_summary != "None":
        return _clamp_words(conceded_summary, max_words=35)

    attackedby = context.get('attackedby_data') or {}
    attack_kind = attackedby.get('kind')
    attack_summary = _safe_text(attackedby.get('summary'), fallback="")
    if attack_kind in {'attack_trapped', 'attack', 'pressure', 'attack_blunder', 'tactically_protected'} and attack_summary:
        return _clamp_words(attack_summary, max_words=35 if attack_kind == 'tactically_protected' else 28)

    forcing_summary = _safe_text(context.get('forcing_signal'), fallback="None")
    if forcing_summary != "None":
        return _clamp_words(forcing_summary, max_words=32)

    capture_check_summary = _safe_text(context.get('capture_check_signal'), fallback="None")
    if capture_check_summary != "None":
        return _clamp_words(capture_check_summary, max_words=32)

    threat_summary = _safe_text(context.get('threat_signal'), fallback="None")
    if threat_summary != "None":
        return _clamp_words(threat_summary, max_words=32)

    move_san = context.get('move_san', 'Move')
    quality = context.get('move_quality', 'Neutral')
    if quality == "Blunder":
        return f"{move_san} is a blunder that concedes a major swing in the position."
    if quality == "Mistake":
        return f"{move_san} is a mistake; there was a stronger continuation available."
    positional_signal = _safe_text(context.get('positional_signal'), fallback="None")
    if positional_signal != "None":
        return _clamp_words(positional_signal, max_words=32)
    if quality == "Good/Improves Position":
        return f"{move_san} is a useful improving move that keeps pressure on your plan."
    return f"{move_san} improves piece coordination."


def _extract_json_blob(raw_text: str) -> Optional[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return None

    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        candidates.append(obj_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_model_commentary_map(raw_text: str) -> Dict[int, str]:
    parsed = _extract_json_blob(raw_text)
    if not parsed:
        return {}

    entries = parsed.get('commentaries') or parsed.get('lines') or parsed.get('moves') or []
    if not isinstance(entries, list):
        return {}

    out: Dict[int, str] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        ply = item.get('ply')
        text = item.get('text') or item.get('commentary') or item.get('line') or ""
        try:
            ply_num = int(ply)
        except (TypeError, ValueError):
            continue
        line = _safe_text(text, fallback="")
        if ply_num > 0 and line and not _is_noneish(line):
            out[ply_num] = line
    return out


def _context_line(context: Dict[str, Any]) -> str:
    parts = [
        f"P{context['ply']}",
        f"move={context['move_san']}",
        f"side={context['side_to_move']}",
        f"quality={context['move_quality']}",
    ]
    if not _is_noneish(context.get('opening_update')):
        parts.append(f"opening_update={_safe_text(context.get('opening_update'))}")
    if str(context.get('attack_kind') or 'none') != 'none':
        parts.append(f"attack_kind={context['attack_kind']}")
        if not _is_noneish(context.get('attack_signal')):
            parts.append(f"attack_signal={_safe_text(context.get('attack_signal'))}")
    if str(context.get('trade_kind') or 'none') != 'none':
        parts.append(f"trade_kind={context['trade_kind']}")
        if not _is_noneish(context.get('trade_signal')):
            parts.append(f"trade_signal={_safe_text(context.get('trade_signal'))}")
    if str(context.get('forced_label') or 'none') != 'none':
        parts.append(f"forced_label={context['forced_label']}")
        if str(context.get('forced_kind') or 'none') != 'none':
            parts.append(f"forced_kind={context['forced_kind']}")
        if not _is_noneish(context.get('forced_signal')):
            parts.append(f"forced_signal={_safe_text(context.get('forced_signal'))}")
    if str(context.get('threat_response_label') or 'none') != 'none':
        parts.append(f"threat_response_label={context['threat_response_label']}")
        if str(context.get('threat_response_kind') or 'none') != 'none':
            parts.append(f"threat_response_kind={context['threat_response_kind']}")
        if not _is_noneish(context.get('threat_response_signal')):
            parts.append(f"threat_response_signal={_safe_text(context.get('threat_response_signal'))}")
    if not _is_noneish(context.get('conceded_signal')):
        parts.append(f"conceded_signal={_safe_text(context.get('conceded_signal'))}")
    if not _is_noneish(context.get('forcing_signal')):
        parts.append(f"forcing_signal={_safe_text(context.get('forcing_signal'))}")
    if not _is_noneish(context.get('capture_check_signal')):
        parts.append(f"capture_check_signal={_safe_text(context.get('capture_check_signal'))}")
    if not _is_noneish(context.get('threat_signal')):
        parts.append(f"threat_signal={_safe_text(context.get('threat_signal'))}")
    if not _is_noneish(context.get('tactics')):
        parts.append(f"tactics={_safe_text(context.get('tactics'))}")
    if not _is_noneish(context.get('momentum')):
        parts.append(f"momentum={_safe_text(context.get('momentum'))}")
    if not _is_noneish(context.get('positional_signal')):
        parts.append(f"positional_signal={compact_positional_signal_for_prompt(_safe_text(context.get('positional_signal')))}")
    return " | ".join(parts)


def build_batch_contexts(
    sid: str,
    fens: List[str],
    moves: List[Any],
    openings: List[Any],
    progress_cb: ProgressCallback = None,
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    san_moves = _to_san_moves(moves)
    total_plies = min(len(san_moves), max(0, len(fens) - 1))
    if total_plies <= 0:
        return [], {}

    _emit_progress(progress_cb, 2, "Preparing full-game analysis context...")

    evals: List[int] = [0] * (total_plies + 1)
    if STOCKFISH_PATH:
        for idx in range(total_plies + 1):
            try:
                evals[idx] = get_commentary_eval(sid, fens[idx])
            except Exception as exc:
                logger.debug("Batch eval failed at index %s: %s", idx, exc)
                evals[idx] = evals[idx - 1] if idx > 0 else 0
            _emit_progress(
                progress_cb,
                5 + int(35 * (idx / max(1, total_plies))),
                f"Evaluating position {idx}/{total_plies}...",
            )

    memory = GameNarrativeMemory()
    last_opening_announced: Optional[str] = None
    contexts: List[Dict[str, Any]] = []
    move_quality_map: Dict[int, str] = {}

    for ply in range(1, total_plies + 1):
        prev_fen = fens[ply - 1]
        current_fen = fens[ply]
        move_san = san_moves[ply - 1]

        try:
            prev_board = chess.Board(prev_fen)
            current_board = chess.Board(current_fen)
        except ValueError:
            continue

        forced_response = assess_forced_response_to_hanging_attack(
            sid,
            memory.pending_forced_response,
            prev_board,
            current_board,
            move_san,
        )
        if forced_response.get('matched') or forced_response.get('stale'):
            memory.pending_forced_response = None
        threat_response = assess_threat_response_to_pending(
            sid,
            memory.pending_threat_response,
            prev_board,
            current_board,
            move_san,
        )
        if threat_response.get('matched') or threat_response.get('stale'):
            memory.pending_threat_response = None

        attackedby_data = Attackedby(prev_board, current_board, move_san)
        attackedby_data = validate_hanging_target_with_stockfish(
            sid,
            prev_board,
            current_board,
            move_san,
            attackedby_data,
        )
        trade_data = detect_trade_context(prev_board, current_board, move_san)

        eval_prev = evals[ply - 1]
        eval_curr = evals[ply]
        move_quality = _move_quality_from_eval_delta(eval_prev, eval_curr, prev_board.turn == chess.WHITE)
        if attackedby_data.get('kind') == 'attack_blunder':
            move_quality = "Blunder"
        if forced_response.get('label') == 'blunder':
            move_quality = "Blunder"
        elif forced_response.get('label') == 'mistake' and move_quality != "Blunder":
            move_quality = "Mistake"
        if threat_response.get('label') == 'blunder':
            move_quality = "Blunder"
        elif threat_response.get('label') == 'mistake' and move_quality != "Blunder":
            move_quality = "Mistake"

        move_quality_map[ply] = move_quality.lower() if move_quality else "neutral"

        if STOCKFISH_PATH:
            memory.update_eval(eval_curr, ply)
            momentum = memory.detect_momentum(eval_curr, prev_board.turn) or "None"
        else:
            momentum = "None"

        opening_name = openings[ply] if ply < len(openings) else "Unknown"
        opening_update, last_opening_announced = build_opening_update(opening_name, last_opening_announced)
        raw_positional_signal = build_positional_signal(prev_board, current_board, move_san)
        forcing_signal = build_forcing_signal(sid, prev_board, current_board, move_san)
        capture_check_signal = build_capture_check_signal(prev_board, current_board, move_san)
        try:
            threat_signal = build_threat_signal(sid, prev_board, current_board)
        except Exception as exc:
            logger.error("Threat signal failed for %s on ply %s: %s", sid, ply, exc)
            threat_signal = {'kind': None, 'summary': 'None', 'must_mention': False}
        try:
            conceded_data = build_conceded_threat_signal(sid, prev_board, current_board)
        except Exception as exc:
            logger.error("Conceded-threat signal failed for %s on ply %s: %s", sid, ply, exc)
            conceded_data = {'kind': None, 'summary': 'None'}
        conceded_kind = conceded_data.get('kind') or 'none'
        conceded_signal = conceded_data.get('summary') or "None"
        if move_quality not in {"Mistake", "Blunder"}:
            conceded_kind = 'none'
            conceded_signal = "None"
        tactics_signal = build_tactics_signal(
            sid=sid,
            prev_board=prev_board,
            current_board=current_board,
            move_hint_san=move_san,
            tactics_analyzer=tactics_analyzer,
            lock=tactics_lock,
        )
        if (forcing_signal.get('kind') or "none") != "none" and " pin created" in tactics_signal.lower():
            tactics_signal = "None"
        has_non_positional_event = any(
            (
                (attackedby_data.get('kind') or 'none') != 'none',
                (trade_data.get('kind') or 'none') != 'none',
                (forced_response.get('label') or 'none') != 'none',
                (threat_response.get('label') or 'none') != 'none',
                conceded_kind != 'none',
                (forcing_signal.get('kind') or 'none') != 'none',
                (capture_check_signal.get('kind') or 'none') != 'none',
                (threat_signal.get('kind') or 'none') != 'none',
                tactics_signal != "None",
            )
        )
        positional_signal = "None" if has_non_positional_event else raw_positional_signal

        update_pending_forced_response(memory, attackedby_data, current_board, prev_board.turn, ply)
        update_pending_threat_response(memory, threat_signal, current_board, ply)

        context = {
            'ply': ply,
            'move_san': move_san,
            'side_to_move': "White" if prev_board.turn == chess.WHITE else "Black",
            'opening_name': opening_name,
            'opening_update': opening_update,
            'move_quality': move_quality,
            'momentum': momentum,
            'attack_kind': attackedby_data.get('kind') or 'none',
            'attack_signal': attackedby_data.get('summary') if (attackedby_data.get('kind') or 'none') != 'none' else "None",
            'trade_kind': trade_data.get('kind') or 'none',
            'trade_signal': trade_data.get('summary') or "None",
            'forced_label': forced_response.get('label') or 'none',
            'forced_kind': forced_response.get('response_kind') or 'none',
            'forced_signal': forced_response.get('summary') or "None",
            'threat_response_label': threat_response.get('label') or 'none',
            'threat_response_kind': threat_response.get('kind') or 'none',
            'threat_response_signal': threat_response.get('summary') or "None",
            'conceded_kind': conceded_kind,
            'conceded_signal': conceded_signal,
            'positional_signal': positional_signal,
            'forcing_kind': forcing_signal.get('kind') or 'none',
            'forcing_signal': forcing_signal.get('summary') or "None",
            'capture_check_kind': capture_check_signal.get('kind') or 'none',
            'capture_check_signal': capture_check_signal.get('summary') or "None",
            'threat_kind': threat_signal.get('kind') or 'none',
            'threat_signal': threat_signal.get('summary') or "None",
            'tactics': tactics_signal,
            'attackedby_data': attackedby_data,
            'trade_data': trade_data,
            'forced_response': forced_response,
            'threat_response': threat_response,
            'forcing_data': forcing_signal,
            'capture_check_data': capture_check_signal,
            'threat_data': threat_signal,
        }
        contexts.append(context)

        _emit_progress(
            progress_cb,
            42 + int(38 * (ply / max(1, total_plies))),
            f"Building commentary context {ply}/{total_plies}...",
        )

    return contexts, move_quality_map


def generate_batch_commentary(
    text_model: Any,
    contexts: List[Dict[str, Any]],
    progress_cb: ProgressCallback = None,
) -> Dict[int, str]:
    if not contexts:
        return {}

    _emit_progress(progress_cb, 82, "Generating fluent commentary in one API call...")

    context_lines = [_context_line(ctx) for ctx in contexts]
    prompt = (
        "You are Grandmaster Insight. Convert compact chess context lines into fluent one-or-two-sentence commentary.\n"
        "Only active signals are listed; missing signals are absent by design.\n"
        "Use ONLY provided facts.\n"
        "Output strict JSON only: {\"commentaries\":[{\"ply\":<int>,\"text\":\"<sentence>\"}, ...]} with one entry per ply.\n"
        "Keep lines natural and concise (target 14-34 words, max 40).\n"
        "Prioritize motifs in this order: capture_check_signal > forcing_signal > threat_response_signal > conceded_signal > threat_signal > trade > attack/pressure > tactics > opening_update > positional_signal.\n"
        "Do not invent absent motifs. Keep attack vs pressure terminology consistent. Avoid dry templates like 'White makes the move X.'\n"
        "No engine references and no move variations.\n\n"
        "CONTEXT LINES:\n"
        + "\n".join(context_lines)
    )

    model_map: Dict[int, str] = {}
    if text_model:
        try:
            logger.info("AI batch commentary prompt [plies=%s]:\n%s", len(contexts), prompt)
            resp = text_model.generate_content(prompt)
            raw = (getattr(resp, 'text', '') or '').strip()
            model_map = _parse_model_commentary_map(raw)
        except Exception as exc:
            logger.error("Batch commentary generation failed: %s", exc)

    commentary_map: Dict[int, str] = {}
    for ctx in contexts:
        ply = ctx['ply']
        line = sanitize_commentary_text(_safe_text(model_map.get(ply), fallback=""))
        if _is_noneish(line):
            line = ""
        if not line:
            line = _fallback_commentary_line(ctx)

        line = enforce_trade_consistency(line, ctx.get('trade_data') or {})
        line = enforce_attack_pressure_consistency(
            line,
            ctx.get('attackedby_data') or {},
            trade_kind=ctx.get('trade_kind') or "none",
        )
        line = enforce_forced_response_consistency(line, ctx.get('forced_response') or {})
        line = enforce_threat_response_consistency(line, ctx.get('threat_response') or {})
        line = enforce_conceded_threat_consistency(
            line,
            {'kind': ctx.get('conceded_kind') or 'none', 'summary': ctx.get('conceded_signal') or "None"},
        )
        line = enforce_move_event_consistency(
            commentary=line,
            forcing_signal=ctx.get('forcing_data') or {},
            capture_check_signal=ctx.get('capture_check_data') or {},
        )
        line = enforce_threat_consistency(line, ctx.get('threat_data') or {})
        line = enforce_tactics_event_consistency(line, ctx.get('tactics') or "None")
        line = enforce_momentum_consistency(line, ctx.get('momentum') or "None")
        line = enforce_opening_update_consistency(
            commentary=line,
            opening_update=ctx.get('opening_update') or "None",
            opening_name=ctx.get('opening_name'),
        )
        line = enforce_positional_signal_consistency(
            commentary=line,
            positional_signal=ctx.get('positional_signal') or "None",
            attack_kind=ctx.get('attack_kind') or "none",
            trade_kind=ctx.get('trade_kind') or "none",
            forced_label=ctx.get('forced_label') or "none",
            tactics_signal=ctx.get('tactics') or "None",
            opening_update=ctx.get('opening_update') or "None",
            forcing_signal=ctx.get('forcing_signal') or "None",
            threat_signal=ctx.get('threat_signal') or "None",
            threat_response_signal=ctx.get('threat_response_signal') or "None",
            conceded_signal=ctx.get('conceded_signal') or "None",
        )
        line = sanitize_commentary_text(line)
        if not line:
            line = _fallback_commentary_line(ctx)
        line = ensure_sentence_punctuation(line)
        if not line:
            line = ensure_sentence_punctuation(f"{ctx.get('move_san', 'Move')} improves piece coordination.")

        commentary_map[ply] = _clamp_words(line, max_words=35)

    _emit_progress(progress_cb, 98, "Finalizing commentary cache...")
    return commentary_map
