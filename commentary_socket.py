import base64
import io
import struct
from typing import Any, Dict

import chess
from flask import request

from analysis_socket import tactics_lock
from commentary_attack import Attackedby
from commentary_attack_validation import enforce_attack_pressure_consistency, validate_hanging_target_with_stockfish
from commentary_engine import classify_move_quality, get_commentary_eval
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
from narrative import GameNarrativeMemory, session_memories
from runtime import STOCKFISH_PATH, logger, socketio, tactics_analyzer, text_model, tts_client, types


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    bits_per_sample = 16
    sample_rate = 24000
    try:
        if "rate=" in mime_type:
            sample_rate = int(mime_type.split("rate=")[1])
    except (ValueError, IndexError):
        pass

    num_channels = 1
    data_size = len(audio_data)
    block_align = num_channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def _first_non_noneish(*values: Any) -> str:
    for value in values:
        text = sanitize_commentary_text(value)
        if text and not is_noneish_text(text):
            return text
    return ""


@socketio.on('get_move_quality')
def handle_move_quality(data):
    sid = request.sid
    if not isinstance(data, dict) or not STOCKFISH_PATH:
        return

    fen_before = data.get('fen_before')
    fen_after = data.get('fen_after')
    ply = int(data.get('ply', 0) or 0)
    request_id = data.get('requestId')

    if not fen_before or not fen_after:
        return

    try:
        cp_before = get_commentary_eval(sid, fen_before)
        cp_after = get_commentary_eval(sid, fen_after)
        is_white_move = chess.Board(fen_before).turn == chess.WHITE
        classification = classify_move_quality(cp_before, cp_after, is_white_move)
    except Exception as exc:
        logger.debug("Move quality evaluation failed for %s on ply %s: %s", sid, ply, exc)
        return

    socketio.emit(
        'move_quality_result',
        {'classification': classification, 'ply': ply, 'requestId': request_id},
        room=sid,
    )


@socketio.on('get_ai_commentary')
def handle_ai_commentary(data):
    sid = request.sid
    if not isinstance(data, dict):
        socketio.emit('ai_commentary_error', {'message': 'Invalid commentary payload.', 'ply': 0}, room=sid)
        return

    if not text_model:
        socketio.emit(
            'ai_commentary_text_result',
            {'commentary': 'AI unavailable.', 'ply': data.get('ply', 0)},
            room=sid,
        )
        return

    memory = session_memories.setdefault(sid, GameNarrativeMemory())

    ply = int(data.get('ply', 0) or 0)
    current_fen = data.get('current_fen')
    prev_fen = data.get('previous_fen')
    human_move = data.get('humanMove')

    if not current_fen or not prev_fen or not human_move:
        socketio.emit('ai_commentary_error', {'message': 'Incomplete commentary payload.', 'ply': ply}, room=sid)
        return

    eval_curr_val = 0
    eval_prev_val = 0
    move_quality = "Neutral"

    try:
        prev_board = chess.Board(prev_fen)
        current_board = chess.Board(current_fen)
    except ValueError:
        socketio.emit('ai_commentary_error', {'message': 'Invalid position payload.', 'ply': ply}, room=sid)
        return

    forced_response = assess_forced_response_to_hanging_attack(
        sid,
        memory.pending_forced_response,
        prev_board,
        current_board,
        human_move,
    )
    if forced_response.get('matched') or forced_response.get('stale'):
        memory.pending_forced_response = None
    forced_response_text = forced_response.get('summary') or "None"
    threat_response = assess_threat_response_to_pending(
        sid,
        memory.pending_threat_response,
        prev_board,
        current_board,
        human_move,
    )
    if threat_response.get('matched') or threat_response.get('stale'):
        memory.pending_threat_response = None
    threat_response_text = threat_response.get('summary') or "None"

    attackedby_data = Attackedby(prev_board, current_board, human_move)
    attackedby_data = validate_hanging_target_with_stockfish(sid, prev_board, current_board, human_move, attackedby_data)
    attack_pressure_kind = attackedby_data.get('kind') or "none"
    top_attack_target = (attackedby_data.get('targets') or [{}])[0]
    attack_pressure_target = (
        f"{top_attack_target.get('piece_name')} on {top_attack_target.get('square')}"
        if top_attack_target.get('piece_name') and top_attack_target.get('square')
        else "none"
    )
    attack_pressure_text = (
        attackedby_data.get('summary')
        if attack_pressure_kind != "none"
        else "None"
    )
    trade_data = detect_trade_context(prev_board, current_board, human_move)
    trade_kind = trade_data.get('kind') or "none"
    trade_text = trade_data.get('summary') or "None"
    opening_name = str(data.get('opening', 'Unknown') or 'Unknown').strip()
    opening_update, next_opening_name = build_opening_update(opening_name, memory.last_opening_announced)
    if opening_update != "None":
        memory.last_opening_announced = next_opening_name
    positional_signal = build_positional_signal(prev_board, current_board, human_move)
    forcing_signal = build_forcing_signal(sid, prev_board, current_board, human_move)
    forcing_text = forcing_signal.get('summary') or "None"
    forcing_kind = forcing_signal.get('kind') or "none"
    capture_check_signal = build_capture_check_signal(prev_board, current_board, human_move)
    capture_check_text = capture_check_signal.get('summary') or "None"
    try:
        threat_data = build_threat_signal(sid, prev_board, current_board)
    except Exception as exc:
        logger.error("Threat signal failed for %s on ply %s: %s", sid, ply, exc)
        threat_data = {'kind': None, 'summary': 'None', 'must_mention': False}
    threat_text = threat_data.get('summary') or "None"
    try:
        conceded_data = build_conceded_threat_signal(sid, prev_board, current_board)
    except Exception as exc:
        logger.error("Conceded-threat signal failed for %s on ply %s: %s", sid, ply, exc)
        conceded_data = {'kind': None, 'summary': 'None'}
    conceded_text = conceded_data.get('summary') or "None"

    if STOCKFISH_PATH:
        try:
            eval_prev_val = get_commentary_eval(sid, prev_fen)
            eval_curr_val = get_commentary_eval(sid, current_fen)
            memory.update_eval(eval_curr_val, ply)

            is_white = prev_board.turn == chess.WHITE
            diff = (eval_curr_val - eval_prev_val) if is_white else (eval_prev_val - eval_curr_val)

            if diff < -200:
                move_quality = "Blunder"
            elif diff < -80:
                move_quality = "Mistake"
            elif diff > 30:
                move_quality = "Good/Improves Position"
        except Exception as exc:
            logger.debug("Commentary eval failed for %s on ply %s: %s", sid, ply, exc)

    if attackedby_data.get('kind') == 'attack_blunder':
        move_quality = "Blunder"

    forced_label = forced_response.get('label')
    if forced_label == 'blunder':
        move_quality = "Blunder"
    elif forced_label == 'mistake' and move_quality != "Blunder":
        move_quality = "Mistake"
    threat_response_label = threat_response.get('label')
    if threat_response_label == 'blunder':
        move_quality = "Blunder"
    elif threat_response_label == 'mistake' and move_quality != "Blunder":
        move_quality = "Mistake"

    conceded_kind = conceded_data.get('kind') or "none"
    if move_quality not in {"Mistake", "Blunder"}:
        conceded_kind = "none"
        conceded_text = "None"

    update_pending_forced_response(memory, attackedby_data, current_board, prev_board.turn, ply)
    update_pending_threat_response(memory, threat_data, current_board, ply)
    momentum_text = memory.detect_momentum(eval_curr_val, prev_board.turn) or "None"
    tactics_str = build_tactics_signal(
        sid=sid,
        prev_board=prev_board,
        current_board=current_board,
        move_hint_san=human_move,
        tactics_analyzer=tactics_analyzer,
        lock=tactics_lock,
    )
    if forcing_kind != "none" and " pin created" in tactics_str.lower():
        tactics_str = "None"
    has_non_positional_event = any(
        (
            attack_pressure_kind != "none",
            trade_kind != "none",
            (forced_label or "none") != "none",
            (threat_response_label or "none") != "none",
            conceded_kind != "none",
            forcing_kind != "none",
            (capture_check_signal.get('kind') or "none") != "none",
            (threat_data.get('kind') or "none") != "none",
            tactics_str != "None",
        )
    )
    positional_for_prompt = "None" if has_non_positional_event else positional_signal
    has_forced = (forced_label or "none") != "none"
    has_threat_response = (threat_response_label or "none") != "none"
    has_conceded = conceded_kind != "none"
    has_attack = attack_pressure_kind != "none"
    has_trade = trade_kind != "none"
    has_tactics = tactics_str != "None"
    has_forcing = forcing_text != "None"
    has_capture_check = capture_check_text != "None"
    has_threat = threat_text != "None"
    has_opening_update = opening_update != "None"
    has_momentum = momentum_text != "None"

    context_lines = [
        f"- Previous Comment: \"{memory.last_commentary}\"",
        f"- Move Quality: {move_quality}",
        f"- Move Played: {human_move} (ply {ply})",
    ]
    if has_opening_update:
        context_lines.append(f"- Opening Update: {opening_update}")
    if has_attack:
        context_lines.append(f"- Attack/Pressure: kind={attack_pressure_kind}; target={attack_pressure_target}; signal={attack_pressure_text}")
    if has_trade:
        context_lines.append(f"- Trade: kind={trade_kind}; signal={trade_text}")
    if has_forced:
        context_lines.append(f"- Forced Response: {forced_response_text}")
    if has_threat_response:
        context_lines.append(f"- Threat Response: {threat_response_text}")
    if has_conceded:
        context_lines.append(f"- Conceded Threat: {conceded_text}")
    if has_forcing:
        context_lines.append(f"- Forcing Motif: {forcing_text}")
    if has_capture_check:
        context_lines.append(f"- Capture/Check Motif: {capture_check_text}")
    if has_threat:
        context_lines.append(f"- Threat: {threat_text}")
    if has_tactics:
        context_lines.append(f"- Tactical Event: {tactics_str}")
    if has_momentum:
        context_lines.append(f"- Momentum Update: {momentum_text}")
    if positional_for_prompt != "None":
        context_lines.append(f"- Positional Intent: {compact_positional_signal_for_prompt(positional_for_prompt)}")
    context_block = "\n    ".join(context_lines)

    prompt = f"""
    You are 'Grandmaster Insight', a natural and energetic chess coach.
    Only active signals are listed below; missing signals are absent by design.
    Use only these facts.

    CONTEXT:
    {context_block}

    TASK:
    - Write one or two fluent sentences (about 14-34 words, max 40).
    - Prioritize concrete motifs in this order: capture/check, forcing motif, threat response, conceded threat, threat, trade, attack/pressure, tactical event, opening update, positional intent.
    - Keep attack vs pressure wording consistent with context.
    - Do not invent pins, forks, threats, or trades not listed.
    - Avoid dry templates like "White makes the move X."
    - No engine references and no move variations.
    """

    try:
        logger.info("AI commentary prompt [sid=%s, ply=%s]:\n%s", sid, ply, prompt)
        resp = text_model.generate_content(prompt)
        commentary = (getattr(resp, 'text', '') or '').strip()
        commentary = enforce_trade_consistency(commentary, trade_data)
        commentary = enforce_attack_pressure_consistency(commentary, attackedby_data, trade_kind=trade_kind)
        commentary = enforce_forced_response_consistency(commentary, forced_response)
        commentary = enforce_threat_response_consistency(commentary, threat_response)
        commentary = enforce_conceded_threat_consistency(commentary, conceded_data)
        commentary = enforce_move_event_consistency(commentary, forcing_signal, capture_check_signal)
        commentary = enforce_threat_consistency(commentary, threat_data)
        commentary = enforce_tactics_event_consistency(commentary, tactics_str)
        commentary = enforce_momentum_consistency(commentary, momentum_text)
        commentary = enforce_opening_update_consistency(commentary, opening_update, opening_name)
        commentary = enforce_positional_signal_consistency(
            commentary=commentary,
            positional_signal=positional_signal,
            attack_kind=attack_pressure_kind,
            trade_kind=trade_kind,
            forced_label=forced_label or "none",
            tactics_signal=tactics_str,
            opening_update=opening_update,
            forcing_signal=forcing_text,
            threat_signal=threat_text,
            threat_response_signal=threat_response_text,
            conceded_signal=conceded_text,
        )
        commentary = sanitize_commentary_text(commentary)
        if not commentary:
            commentary = _first_non_noneish(
                capture_check_text,
                forcing_text,
                threat_response_text,
                conceded_text,
                threat_text,
                trade_text,
                attack_pressure_text,
                tactics_str,
                opening_update,
                positional_signal,
            )
        if not commentary:
            commentary = f"{human_move} keeps piece coordination improving."
        commentary = ensure_sentence_punctuation(commentary)
        memory.add_commentary(ply, commentary)

        socketio.emit('ai_commentary_text_result', {'commentary': commentary, 'ply': ply}, room=sid)

        if data.get('audio_enabled') and tts_client:
            tts_model_name = "gemini-2.5-flash-preview-tts"
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=commentary)])]
            config = types.GenerateContentConfig(
                response_modalities=["audio"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                    )
                ),
            )
            audio_buffer = io.BytesIO()
            first_chunk = True
            mime_type = "audio/L16;rate=24000"

            for chunk in tts_client.models.generate_content_stream(model=tts_model_name, contents=contents, config=config):
                if (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                    and (part := chunk.candidates[0].content.parts[0]).inline_data
                ):
                    if first_chunk:
                        mime_type = part.inline_data.mime_type
                        first_chunk = False
                    audio_buffer.write(part.inline_data.data)

            if audio_buffer.getbuffer().nbytes > 0:
                wav_data = convert_to_wav(audio_buffer.getvalue(), mime_type)
                socketio.emit(
                    'ai_commentary_audio_result',
                    {'audio_data': base64.b64encode(wav_data).decode('utf-8'), 'ply': ply},
                    room=sid,
                )

    except Exception as exc:
        logger.error("AI generation error on ply %s: %s", ply, exc)
        socketio.emit('ai_commentary_error', {'message': 'AI commentary failed.', 'ply': ply}, room=sid)
