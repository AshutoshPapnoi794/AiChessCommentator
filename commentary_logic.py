"""
Deterministic commentary planning helpers.

This module turns board state plus verified chess facts into a commentary
blueprint that can stand on its own without any LLM reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chess


CENTRAL_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
FIANCHETTO_TARGETS = {
    chess.WHITE: {chess.G3: chess.F1, chess.B3: chess.C1},
    chess.BLACK: {chess.G6: chess.F8, chess.B6: chess.C8},
}
HOME_SQUARES = {
    chess.WHITE: {
        chess.KNIGHT: {chess.B1, chess.G1},
        chess.BISHOP: {chess.C1, chess.F1},
        chess.ROOK: {chess.A1, chess.H1},
        chess.QUEEN: {chess.D1},
    },
    chess.BLACK: {
        chess.KNIGHT: {chess.B8, chess.G8},
        chess.BISHOP: {chess.C8, chess.F8},
        chess.ROOK: {chess.A8, chess.H8},
        chess.QUEEN: {chess.D8},
    },
}
NATURAL_KNIGHT_SQUARES = {
    chess.WHITE: {chess.C3, chess.F3, chess.D2, chess.E2},
    chess.BLACK: {chess.C6, chess.F6, chess.D7, chess.E7},
}
KEYWORD_STOPWORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "if", "in", "into",
    "is", "it", "its", "just", "not", "of", "on", "or", "so", "than", "that", "the",
    "their", "them", "there", "this", "to", "up", "way", "with",
}

# Lightweight piece-value table used for heuristic comparisons inside commentary
_PIECE_VALUE_MAP: dict = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass
class CommentaryPoint:
    category: str
    family: str
    priority: int
    lead: str
    support: str = ""
    keywords: List[str] = field(default_factory=list)


@dataclass
class CommentaryBlueprint:
    category: str
    family: str
    lead: str
    support: str = ""
    verdict: str = ""
    keywords: List[str] = field(default_factory=list)
    draft: str = ""
    points: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def blueprint_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[CommentaryBlueprint]:
    if not isinstance(payload, dict):
        return None
    return CommentaryBlueprint(
        category=str(payload.get("category", "")),
        family=str(payload.get("family", "")),
        lead=str(payload.get("lead", "")),
        support=str(payload.get("support", "")),
        verdict=str(payload.get("verdict", "")),
        keywords=[str(item) for item in payload.get("keywords", []) if item],
        draft=str(payload.get("draft", "")),
        points=[item for item in payload.get("points", []) if isinstance(item, dict)],
    )


def build_commentary_blueprint(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint] = None,
) -> CommentaryBlueprint:
    points = _build_ranked_points(prev_board, curr_board, move, context, previous_blueprint)
    if not points:
        points = [_general_point(prev_board, curr_board, move, context, previous_blueprint)]

    points.sort(key=lambda point: point.priority)
    lead_point = points[0]

    # When the lead is a terminal or decisive event (checkmate, decisive blunder),
    # do NOT pull pawn-structure or strategic noise into the support sentence.
    # Commenting on a passed pawn while the opponent just blundered into checkmate
    # is the single most jarring commentary mistake.
    _TERMINAL_CATEGORIES = {
        "checkmate",
        "blunder_hanging_piece",
        "blunder_missed_tactic",
        "blunder_generic",
    }
    _NOISY_FAMILIES_FOR_TACTICAL_LEAD = {"strategy", "opening", "endgame"}

    support = lead_point.support
    if not support:
        for candidate in points[1:]:
            # Suppress structural commentary when lead is checkmate or decisive blunder
            if lead_point.category in _TERMINAL_CATEGORIES:
                break
            # Suppress structural noise when lead is a strong tactic
            if lead_point.family in {"tactic", "blunder"} and candidate.family in _NOISY_FAMILIES_FOR_TACTICAL_LEAD:
                continue
            if candidate.family != lead_point.family or candidate.category != lead_point.category:
                support = candidate.lead
                break

    verdict = _build_verdict(context, lead_point)
    keywords = _unique_keywords(lead_point.keywords)
    if len(keywords) < 2 and support:
        keywords.extend(keyword for keyword in _keywords_from_text(support) if keyword not in keywords)
    if len(keywords) < 2 and verdict:
        keywords.extend(keyword for keyword in _keywords_from_text(verdict) if keyword not in keywords)

    # Inject move squares so commentary always references specific board squares
    verified = context.get("verified_facts", {}) or {}
    move_to = str(verified.get("move_to", "") or "")
    move_from = str(verified.get("move_from", "") or "")
    if move and not move_to:
        move_to = chess.square_name(move.to_square)
    if move and not move_from:
        move_from = chess.square_name(move.from_square)
    if move_to and move_to not in keywords:
        keywords.append(move_to)
    if move_from and move_from not in keywords:
        keywords.append(move_from)

    # Inject tactic type if present
    tactical = context.get("tactical", {}) or {}
    primary_tactic = tactical.get("primary_tactic") or {}
    tactic_type = str(primary_tactic.get("type", "") or "").replace("_", " ")
    if tactic_type and tactic_type not in keywords:
        keywords.append(tactic_type)

    draft = render_blueprint_draft(
        CommentaryBlueprint(
            category=lead_point.category,
            family=lead_point.family,
            lead=lead_point.lead,
            support=support,
            verdict=verdict,
            keywords=keywords,
            points=[asdict(point) for point in points[:5]],
        )
    )

    return CommentaryBlueprint(
        category=lead_point.category,
        family=lead_point.family,
        lead=lead_point.lead,
        support=support,
        verdict=verdict,
        keywords=keywords,
        draft=draft,
        points=[asdict(point) for point in points[:5]],
    )


def render_blueprint_draft(blueprint: CommentaryBlueprint) -> str:
    sentences = [_ensure_sentence(blueprint.lead)]
    if blueprint.support:
        sentences.append(_ensure_sentence(blueprint.support))
    if blueprint.verdict:
        sentences.append(_ensure_sentence(blueprint.verdict))
    return " ".join(sentence for sentence in sentences if sentence).strip()


def _build_ranked_points(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> List[CommentaryPoint]:
    points: List[CommentaryPoint] = []

    for builder in (
        _blunder_point,
        _checkmate_point,
        _tactic_point,
        _capture_point,
        _attack_or_defense_point,
        _castling_point,
        _pawn_structure_point,
        _piece_coordination_point,
        _endgame_technique_point,
        _blockade_point,
        _prophylaxis_point,
        _opening_plan_point,
        _strategic_point,
    ):
        point = builder(prev_board, curr_board, move, context, previous_blueprint)
        if point:
            points.append(point)

    return points


def _blunder_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    quality = str(context.get("quality", ""))
    if quality not in {"mistake", "blunder"}:
        return None

    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    tactical = context.get("tactical", {})
    hung = tactical.get("hung_piece") or {}
    missed = context.get("missed_tactics") or {}
    opponent = _opponent_name(mover)

    if hung.get("piece"):
        piece = str(hung["piece"])
        lead = f"{mover} goes badly wrong with {san} and leaves the {piece} hanging."
        if missed.get("missed_best_move"):
            support = f"{opponent} can punish it immediately with {missed['missed_best_move']}."
            keywords = [piece, str(missed["missed_best_move"]), "hanging"]
        else:
            support = f"That kind of loose piece gives {opponent} a tactical target straight away."
            keywords = [piece, "hanging", "tactical"]
        return CommentaryPoint("blunder_hanging_piece", "blunder", 0, lead, support, keywords)

    if missed.get("missed_best_move"):
        best_move = str(missed["missed_best_move"])
        lead = f"{mover}'s {san} misses the tactical point of the position."
        support = f"{best_move} was the critical move, and {opponent} should take over after that."
        return CommentaryPoint("blunder_missed_tactic", "blunder", 0, lead, support, [best_move, "tactic", "initiative"])

    lead = f"{mover}'s {san} is inaccurate and hands the initiative away."
    support = f"The position was manageable before this, but {opponent} now gets the more active game."
    return CommentaryPoint("blunder_generic", "blunder", 0, lead, support, ["initiative", "active"])


def _checkmate_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    verified = context.get("verified_facts", {})
    if not verified.get("gives_checkmate"):
        return None

    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    lead = f"{mover} ends it at once with {san}, and the king has no escape."
    support = "Everything in the previous sequence was building toward that mating net."
    return CommentaryPoint("checkmate", "tactic", 1, lead, support, ["checkmate", "king", "mating net"])


def _tactic_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    tactical = context.get("tactical", {})
    primary = tactical.get("primary_tactic") or {}
    if not primary:
        return None

    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    pattern_type = str(primary.get("type", ""))
    attack_info = tactical.get("attack_info") or {}
    verified = context.get("verified_facts", {}) or {}
    quality = str(context.get("quality", "") or "")
    hung = tactical.get("hung_piece") or {}
    repeated = _same_family(previous_blueprint, "tactic")

    if quality in {"inaccuracy", "mistake", "blunder"} and hung:
        return None

    # ---- RELEVANCE GATE ----
    # Only report tactical patterns when the move is tactically meaningful:
    # - The move gives check, captures, creates an attack, or is brilliant
    # - OR the move directly involves the tactical squares (moved piece creates/enables the pattern)
    move_is_tactical = (
        tactical.get("is_brilliant")
        or attack_info.get("is_attacking")
        or move_info.get("is_check")
        or verified.get("captures")
    )
    
    # Check if the moved piece is involved in the tactical pattern
    tactic_squares = primary.get("squares", [])
    move_dest = chess.square_name(move.to_square) if move else ""
    move_involves_tactic = move_dest in tactic_squares

    # For most non-fork patterns, require either tactical context or direct involvement
    passive_patterns = {"pin", "xray", "skewer", "trapped_piece", "overloading",
                       "interference", "clearance", "deflection", "decoy"}
    if verified.get("is_recapture") and pattern_type in passive_patterns:
        return None
    if verified.get("captures") and pattern_type in passive_patterns and not move_info.get("is_check") and not tactical.get("is_brilliant"):
        return None
    if pattern_type in passive_patterns and not (move_is_tactical or move_involves_tactic):
        return None

    # ---- FORK ----
    if pattern_type == "fork":
        targets = _describe_tactic_targets(curr_board, primary.get("squares", [])[1:])
        targets_text = " and ".join(targets[:2]) if len(targets) >= 2 else "multiple targets"
        if repeated:
            lead = f"{san} creates an immediate fork on {targets_text}."
        else:
            lead = f"{mover}'s {san} is a tactical fork, hitting {targets_text} at the same time."
        if tactical.get("is_brilliant"):
            support = "Even if the piece can be challenged, the double attack keeps the initiative and makes the tactic work."
            keywords = ["fork", *targets[:2], "initiative"]
        else:
            support = "That kind of double attack is hard to meet cleanly, because one of those targets usually has to give way."
            keywords = ["fork", *targets[:2]]
        return CommentaryPoint("tactic_fork", "tactic", 2, lead, support, keywords)

    # ---- PIN ----
    if pattern_type == "pin":
        pieces = [str(piece) for piece in primary.get("pieces", []) if piece]
        if len(pieces) >= 2 and pieces[1] == "pawn":
            return None
        pinned_square = ""
        squares = primary.get("squares", [])
        if len(squares) >= 2:
            pinned_square = str(squares[1])
        pinned_piece = ""
        if pinned_square:
            piece = curr_board.piece_at(chess.parse_square(pinned_square))
            if piece:
                pinned_piece = chess.piece_name(piece.piece_type)
        lead_piece = pinned_piece or str(attack_info.get("target_piece", "piece"))
        lead = f"{mover}'s {san} pins the {lead_piece} on {pinned_square} and makes that piece awkward to move.".replace(" on .", ".")
        support = "The point is not just the pin itself, but the way it increases the pressure on the squares behind it."
        return CommentaryPoint("tactic_pin", "tactic", 2, lead, support, ["pin", lead_piece, pinned_square])

    # ---- TRAPPED PIECE ----
    if pattern_type == "trapped_piece":
        trapped_desc = str(primary.get("description", "Trapped piece"))
        narrative = str(primary.get("narrative", "")).strip()
        lead = narrative if narrative else f"{trapped_desc}."
        support = "This changes the character of the position immediately and asks a very concrete question."
        return CommentaryPoint("tactic_trapped", "tactic", 2, lead, support, ["trapped", "material"])

    # ---- REMOVAL OF GUARD ----
    if pattern_type == "removal_of_guard":
        narrative = str(primary.get("narrative", "")).strip()
        lead = narrative if narrative else f"{mover} removes a key defender with {san}."
        support = "With the guard eliminated, the previously defended piece becomes a tactical target."
        return CommentaryPoint("tactic_removal", "tactic", 3, lead, support, ["removal of guard", "tactical"])

    # ---- X-RAY ----
    if pattern_type == "xray":
        if not (
            verified.get("captures")
            or move_info.get("is_check")
            or tactical.get("is_brilliant")
            or str(attack_info.get("target_piece", "")) in {"queen", "rook", "king"}
        ):
            return None
        narrative = str(primary.get("narrative", "")).strip()
        lead = narrative if narrative else f"{mover} creates an x-ray attack with {san}."
        support = "This hidden pressure along the line can become decisive as the position evolves."
        return CommentaryPoint("tactic_xray", "tactic", 4, lead, support, ["x-ray", "pressure"])

    # ---- DECOY / BAIT ----
    # Only label as intentional bait when the engine confirms it is the best move.
    # This distinguishes a genuine sacrifice/lure from an accidental hanging piece.
    if pattern_type == "decoy":
        narrative = str(primary.get("narrative", "")).strip()
        if quality in {"best", "excellent"} or tactical.get("is_brilliant"):
            lead = narrative if narrative else (
                f"{mover} lays a deliberate trap with {san}, offering material "
                f"to lure the opponent into a losing reply."
            )
            support = (
                "The sacrifice is calculated, not accidental — taking the bait "
                "leads straight into a worse position."
            )
            return CommentaryPoint(
                "tactic_decoy_bait", "tactic", 2, lead, support,
                ["decoy", "bait", "sacrifice"]
            )
        else:
            lead = narrative if narrative else (
                f"{mover} plays {san}, creating a complex tactical situation."
            )
            support = (
                "Whether this is an intentional lure or a tactical oversight, "
                "the position requires careful calculation from both sides."
            )
            return CommentaryPoint(
                "tactic_decoy_unclear", "tactic", 3, lead, support,
                ["tactical", "complex", "sacrifice"]
            )

    # ---- ALL OTHER TACTICS ----
    narrative = str(primary.get("narrative", "")).strip()
    if narrative:
        lead = narrative.rstrip("!").rstrip(".") + "."
    else:
        lead = f"{mover} finds a tactical shot with {san}."
    if tactical.get("is_brilliant"):
        support = "It works because the tactical justification is stronger than the material on offer."
        keywords = [pattern_type.replace("_", " "), "tactical", "material"]
    else:
        support = "This changes the character of the position immediately and asks a very concrete question."
        keywords = [pattern_type.replace("_", " "), "tactical", "concrete"]
    return CommentaryPoint("tactic_generic", "tactic", 3, lead, support, keywords)


def _capture_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    capture_info = (context.get("tactical", {}) or {}).get("capture_analysis") or {}
    verified = context.get("verified_facts", {})
    if not capture_info and not verified.get("captures"):
        return None

    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    if verified.get("is_recapture"):
        lead = f"{mover} recaptures with {san}, restoring material balance before the position drifts away."
        support = "That kind of immediate recapture matters because it keeps the opponent from consolidating the extra material."
        return CommentaryPoint("capture_recapture", "material", 4, lead, support, ["recapture", "material"])

    capture_type = str(capture_info.get("type", ""))
    target = str(capture_info.get("target", verified.get("captures", "piece")))
    recapture_details = _expected_recapture_details(curr_board, move.to_square) if move else {}
    if capture_type == "free_capture":
        lead = f"{mover} simply wins material with {san}, picking up the {target} without a clean recapture."
        support = "When a piece can be taken for free like that, the rest of the position almost becomes secondary."
        return CommentaryPoint("capture_free", "material", 4, lead, support, [target, "material", "free"])
    if capture_type == "winning_capture":
        if target == "queen":
            lead = f"{mover} wins the queen with {san}, and that material swing is usually decisive."
            support = "Even if the capturing piece is traded back later, queen-for-piece trades almost never leave the other side enough compensation."
            return CommentaryPoint("capture_wins_queen", "material", 3, lead, support, ["queen", "material", "decisive"])
        lead = f"{mover} wins the {target} with {san} for far less material."
        support = "That kind of favorable exchange usually decides the game unless there is immediate tactical compensation."
        return CommentaryPoint("capture_major_gain", "material", 4, lead, support, [target, "material", "gain"])
    if capture_type == "favorable_trade":
        lead = f"{mover} uses {san} to force a favorable trade against the {target}."
        support = "Exchanges like this are useful because the less valuable piece is doing the work."
        return CommentaryPoint("capture_favorable", "material", 4, lead, support, [target, "trade", "favorable"])
    if capture_type == "equal_trade":
        attacker = str(capture_info.get("attacker", move_info.get("piece", "piece")))
        # Detect Knight-Bishop exchange: both are equal material but different
        # piece types, which the glossary calls "Creating Imbalance" -- changing
        # the technical character of the position (colour-specific weaknesses,
        # open-diagonal play vs. outposts, etc.).
        _minor_set = {"knight", "bishop"}
        if attacker in _minor_set and target in _minor_set and attacker != target:
            lead = f"{mover} trades the {attacker} for the {target} with {san}, creating a strategic imbalance."
            support = (
                "Knights thrive in closed positions with fixed pawn chains, while bishops dominate open diagonals. "
                "This exchange shifts which plans are realistic for both sides over the next few moves."
            )
            return CommentaryPoint("capture_imbalance", "strategy", 4, lead, support,
                                   [attacker, target, "imbalance"])
        if recapture_details.get("doubled_file"):
            file_name = recapture_details["doubled_file"]
            reply = recapture_details.get("san", "")
            lead = f"{mover} uses {san} to trade {attacker} for {target} and damage the pawn structure."
            if reply:
                support = f"After {reply}, the recapturing side is left with doubled pawns on the {file_name}-file."
            else:
                support = f"The resulting structure leaves doubled pawns on the {file_name}-file, which can become a long-term weakness."
            return CommentaryPoint("capture_equal_structure", "material", 4, lead, support, [attacker, target, "doubled pawns"])
        lead = f"{mover} is happy to simplify with {san}, trading {attacker} for {target}."
        support = "That exchange does not win material outright, but it can make the position much easier to handle."
        return CommentaryPoint("capture_equal", "material", 5, lead, support, [attacker, target, "trade"])

    mover_piece = str(move_info.get("piece", "piece"))
    lead = f"{mover} invests material with {san}, offering the {mover_piece} to keep the initiative alive."
    support = "The move is only justified if the follow-up is energetic, so the practical burden shifts to the defender."
    return CommentaryPoint("capture_sacrifice", "material", 4, lead, support, [mover_piece, "sacrifice", "initiative"])


def _attack_or_defense_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    tactical = context.get("tactical", {}) or {}
    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    quality = str(context.get("quality", "") or "")
    if quality in {"inaccuracy", "mistake", "blunder"} and tactical.get("hung_piece"):
        return None
    resolved = tactical.get("resolved_threats") or []
    attack = tactical.get("attack_info") or {}
    pressure = tactical.get("pressure_info") or {}
    repeated = _same_family(previous_blueprint, "attack")

    if resolved:
        saved = str(resolved[0])
        lead = f"{mover} solves an immediate problem with {san}, rescuing the {saved} from danger."
        if attack.get("is_attacking"):
            support = f"It is active defence too, because the move also turns around and asks questions of the {attack.get('target_piece', 'position')}."
            keywords = [saved, str(attack.get("target_piece", "")), "defence"]
        else:
            support = "Good defensive moves do more than survive the moment; they also keep the position coordinated."
            keywords = [saved, "defence", "coordination"]
        return CommentaryPoint("defence_active", "defence", 6, lead, support, keywords)

    # King safety commentary when an attack is building
    analysis = context.get("analysis", {}) or {}
    king_safety = analysis.get("king_safety", {}) or {}
    opponent_color = "black" if mover == "White" else "white"
    mover_color_key = "white" if mover == "White" else "black"
    opp_ks = king_safety.get(opponent_color, {}) or {}
    own_ks = king_safety.get(mover_color_key, {}) or {}

    if attack.get("is_attacking"):
        target = str(attack.get("target_piece", "piece"))
        attacker = str(attack.get("attacker_piece", move_info.get("piece", "piece")))
        attack_type = str(attack.get("attack_type", ""))

        # If the opponent's king is vulnerable/exposed, add king safety context
        opp_safety_level = str(opp_ks.get("level", ""))
        if opp_safety_level in ("vulnerable", "exposed") and target in ("king", "queen", "rook"):
            lead = f"{mover} intensifies the attack with {san}, exploiting the exposed king position."
            support = f"With the king lacking adequate shelter, every additional attacker compounds the danger."
            keywords = ["attack", "king safety", target]
            return CommentaryPoint("attack_king_safety", "attack", 5, lead, support, keywords)

        if attack_type == "favorable_trade":
            if repeated:
                lead = f"{mover} again asks an awkward question with {san}, since the {attacker} is pressuring a more valuable {target}."
                support = "Even without a direct knockout, that imbalance can force uncomfortable defensive choices."
            else:
                lead = f"{mover} uses {san} to attack the {target} with a less valuable {attacker}, so the trade would favor them."
                support = "That is the kind of small tactical detail that can force concessions even without an immediate win."
            keywords = [target, attacker, "trade"]
        elif attack_type == "trade_offer":
            if repeated:
                lead = f"{mover} keeps the pressure practical with {san}, challenging the {target} to clarify the position."
                support = "Whether the trade happens or not, the move makes the strategic choice arrive a little sooner."
            else:
                lead = f"{mover} uses {san} to challenge the {target} directly and invite a simplifying trade."
                support = "Whether the opponent accepts or not, that question helps define the next phase of the position."
            keywords = [target, "trade", "challenge"]
        else:
            if repeated:
                lead = f"{mover} keeps the initiative with {san}, forcing the {target} to stay under watch."
                support = "The value of the move is practical: it limits the opponent's freedom before they can organize properly."
            else:
                lead = f"{mover} creates an immediate threat with {san}, putting the {target} under pressure."
                support = "Moves like this matter because they force the opponent to respond instead of improving freely."
            keywords = [target, "threat", "pressure"]

        # Enrich the support sentence with what the opponent can realistically do:
        # retreat, counter-attack, or add a defender
        if move and attack.get("target_square"):
            try:
                t_sq = chess.parse_square(str(attack["target_square"]))
                t_piece = curr_board.piece_at(t_sq)
                mover_color = prev_board.turn
                if t_piece:
                    # Does the target have a valuable counter-attack available?
                    counter_pieces = [
                        curr_board.piece_at(s)
                        for s in curr_board.attacks(t_sq)
                        if curr_board.piece_at(s) and curr_board.piece_at(s).color == mover_color
                    ]
                    if counter_pieces:
                        best_counter = max(
                            counter_pieces,
                            key=lambda p: _PIECE_VALUE_MAP.get(p.piece_type, 0)
                        )
                        if _PIECE_VALUE_MAP.get(best_counter.piece_type, 0) >= _PIECE_VALUE_MAP.get(t_piece.piece_type, 0):
                            support = (
                                f"The {target} cannot simply retreat — it can fire back at the "
                                f"{chess.piece_name(best_counter.piece_type)}, "
                                f"so {mover.lower()} must calculate carefully."
                            )
                    # Count safe retreat squares
                    safe_retreats = sum(
                        1 for s in curr_board.attacks(t_sq)
                        if not curr_board.piece_at(s)
                        and not curr_board.is_attacked_by(mover_color, s)
                    )
                    if safe_retreats == 0 and not counter_pieces:
                        support = (
                            f"The {target} has no safe retreat squares and no counter-threat — "
                            f"it must find a defender or it falls."
                        )
            except (ValueError, AttributeError):
                pass

        return CommentaryPoint("attack_direct", "attack", 6, lead, support, keywords)

    # If own king is under pressure, note the defensive urgency
    own_safety_level = str(own_ks.get("level", ""))
    if own_safety_level in ("vulnerable", "exposed") and not attack.get("is_attacking"):
        moving_piece = str(move_info.get("piece", "piece"))
        lead = f"{mover} plays {san}, shoring up defences around the king."
        support = "With the king's safety compromised, every defensive move must address the most pressing threat first."
        keywords = ["king safety", "defence", moving_piece]
        return CommentaryPoint("defence_king", "defence", 7, lead, support, keywords)

    if pressure.get("is_pressure"):
        target = str(pressure.get("target_piece", "piece"))
        target_sq_str = str(pressure.get("target_square", "") or "")

        # Describe the actual numerical imbalance — this is the core of pressure commentary
        if move and target_sq_str:
            try:
                t_sq = chess.parse_square(target_sq_str)
                mover_color = prev_board.turn
                curr_att = len(list(curr_board.attackers(mover_color, t_sq)))
                curr_def = len(list(curr_board.attackers(not mover_color, t_sq)))
                if curr_att > curr_def + 1:
                    lead = (
                        f"{mover} piles on the {target} with {san} — "
                        f"now {curr_att} attackers face only {curr_def} defender{'s' if curr_def != 1 else ''}, "
                        f"and something has to give."
                    )
                    support = (
                        f"{_opponent_name(mover)} must either add another defender "
                        f"or the {target} will simply fall."
                    )
                    return CommentaryPoint(
                        "attack_pressure_overwhelming", "attack", 6,
                        lead, support, [target, "pressure", "outnumbered"]
                    )
                elif curr_att > curr_def:
                    lead = (
                        f"{mover} adds another attacker on the {target} with {san}, "
                        f"now outnumbering the defenders {curr_att} to {curr_def}."
                    )
                    support = (
                        f"{_opponent_name(mover)} needs to find another defender — "
                        f"otherwise the {target} is going to fall."
                    )
                    return CommentaryPoint(
                        "attack_pressure_escalate", "attack", 6,
                        lead, support, [target, "pressure", "escalate"]
                    )
            except (ValueError, AttributeError):
                pass

        if repeated:
            lead = f"{mover} develops with a point by playing {san}, keeping the {target} in the firing line."
            support = "Repeated pressure often dictates the pace and forces the opponent into passive defence."
        else:
            lead = f"{mover} increases the pressure with {san}, making the {target} harder to defend comfortably."
            support = "Nothing falls immediately, but the move increases the practical burden on the other side."
        return CommentaryPoint("attack_pressure", "attack", 7, lead, support, [target, "pressure"])

    return None


def _castling_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    move_info = context.get("move", {})
    if not move_info.get("is_castling"):
        return None

    mover = str(move_info.get("mover", "Side"))
    side = str((context.get("verified_facts", {}) or {}).get("castling_side", "kingside"))
    repeated = _same_family(previous_blueprint, "king_safety")
    if repeated:
        lead = f"With {move_info.get('san', 'castling')}, {mover} gets the king out of the center and brings the rook into play."
    else:
        lead = f"{mover} castles {side}, tucking the king away and finally activating the rook."
    support = "That is often the last big opening chore, and it makes the next central decision much easier to handle."
    return CommentaryPoint("castling", "king_safety", 8, lead, support, ["castle", "king", "rook"])


def _opening_plan_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    if not move:
        return None

    move_info = context.get("move", {})
    verified = context.get("verified_facts", {}) or {}
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    piece_name = str(move_info.get("piece", "piece"))
    opening_name = _normalize_opening_name(str(context.get("opening_name", "")))
    repeated = _same_family(previous_blueprint, "opening")
    ply = int(context.get("ply", 0) or 0)
    mover_color = prev_board.turn

    if piece_name == "pawn":
        destination = move.to_square
        if destination in {chess.E4, chess.D4, chess.E5, chess.D5}:
            if repeated:
                lead = f"With {san}, {mover} plants a pawn in the center and claims more space."
            else:
                lead = f"{mover} stakes out the center with {san}, grabbing space straight away."
            if destination in {chess.E4, chess.E5}:
                support = "The move also opens a path for the king's bishop and queen, so the development story writes itself."
                keywords = ["center", "bishop", "queen"]
            else:
                support = "It is the classical way to challenge the middle and give the dark-squared bishop a natural route out."
                keywords = ["center", "bishop", "space"]
            return CommentaryPoint("opening_center_claim", "opening", 9, lead, support, keywords)

        if destination in {chess.C6, chess.C3, chess.E6, chess.E3}:
            break_move = "d5" if destination in {chess.C6, chess.E6} and mover_color == chess.BLACK else "d4"
            if destination in {chess.E3, chess.E6}:
                support = "It keeps the structure solid and prepares the bishop without rushing into contact."
                keywords = [break_move, "solid", "bishop"]
            else:
                support = f"The point is to support {break_move} from a solid base instead of clarifying the center too early."
                keywords = [break_move, "center", "solid"]
            if opening_name and ply <= 12:
                lead = f"{mover} signals a {opening_name} setup with {san}, keeping the structure compact."
                keywords.insert(0, opening_name)
            else:
                lead = f"{mover} plays {san} to support {break_move} and keep the center under control."
            return CommentaryPoint("opening_support_break", "opening", 10, lead, support, keywords)

        if verified.get("is_fianchetto_prep"):
            side = "kingside" if chess.square_file(move.to_square) == 6 else "queenside"
            lead = f"{mover} prepares a {side} fianchetto with {san}, aiming to develop the bishop on the long diagonal."
            support = "That keeps the setup flexible while still improving king safety and central influence."
            return CommentaryPoint("opening_fianchetto", "opening", 10, lead, support, ["fianchetto", "bishop", "diagonal"])

    if verified.get("develops_minor_piece"):
        to_square = chess.square_name(move.to_square)
        central_targets = _join_square_names(verified.get("central_control", []))
        defended = _preferred_supported_piece(verified.get("defends_friendly_pieces", []))
        if piece_name == "knight" and move.to_square in NATURAL_KNIGHT_SQUARES.get(mover_color, set()):
            if repeated:
                lead = f"{mover} brings another knight into the game with {san}, adding more control around the center."
            else:
                lead = f"{mover} develops the knight to {to_square}, and that naturally strengthens the central fight."
        else:
            lead = f"{mover} develops the {piece_name} with {san}, putting it on a more useful square."

        if defended:
            support = f"It also reinforces the {defended}, which makes the whole setup easier to maintain."
            keywords = ["develop", defended, "center"]
        elif central_targets:
            support = f"From there the piece helps control {central_targets}, so the move carries real strategic weight."
            keywords = ["develop", "center", central_targets.split(",")[0]]
        else:
            support = "That kind of quiet development matters because it improves coordination without creating new weaknesses."
            keywords = ["develop", "coordination", piece_name]
        return CommentaryPoint("opening_development", "opening", 10, lead, support, keywords)

    return None


def _pawn_structure_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    """Comment on meaningful pawn structure changes."""
    if not move:
        return None
    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    piece_name = str(move_info.get("piece", "piece"))
    analysis = context.get("analysis", {}) or {}
    pawn_structure = analysis.get("pawn_structure", {}) or {}
    verified = context.get("verified_facts", {}) or {}
    opponent = _opponent_name(mover)
    mover_key = "white" if mover == "White" else "black"
    opp_key = "black" if mover == "White" else "white"
    mover_ps = pawn_structure.get(mover_key, {}) or {}
    opp_ps = pawn_structure.get(opp_key, {}) or {}

    # Only for pawn moves or captures that alter structure
    if piece_name != "pawn" and not verified.get("captures"):
        return None

    dest_sq = move.to_square
    dest_name = chess.square_name(dest_sq)

    # Check if this creates an isolated pawn for the opponent (after a capture)
    if verified.get("captures") and piece_name == "pawn":
        opp_isolated = opp_ps.get("isolated", []) or []
        if opp_isolated:
            lead = f"{mover}'s {san} saddles {opponent} with an isolated pawn on {opp_isolated[0]}."
            support = "An isolated pawn cannot be defended by other pawns and becomes a permanent target for pieces."
            return CommentaryPoint("pawn_creates_weakness", "strategy", 8, lead, support, ["isolated", opp_isolated[0], "weakness"])

    # Check for pawn chain establishment
    if piece_name == "pawn":
        chains = mover_ps.get("chains", []) or []
        for chain in chains:
            if isinstance(chain, list) and len(chain) >= 1:
                chain_str = chain[0] if isinstance(chain[0], str) else str(chain[0])
                if dest_name in chain_str:
                    lead = f"{mover} reinforces the pawn chain with {san}, creating a solid structural foundation."
                    support = "The chain controls key squares and restricts the opponent's piece activity."
                    return CommentaryPoint("pawn_chain", "strategy", 9, lead, support, ["pawn chain", "structure", "control"])

    # Check for pawn break execution
    if piece_name == "pawn" and verified.get("captures"):
        levers = pawn_structure.get("levers", []) or []
        if levers:
            lead = f"{mover} executes the pawn break with {san}, opening the position and creating dynamic chances."
            support = "Pawn breaks are the key moments in most chess games — they determine whether the position becomes open or closed."
            return CommentaryPoint("pawn_break", "strategy", 8, lead, support, ["pawn break", "dynamic", "open"])

    # Check for doubled pawns creation
    if verified.get("captures") and piece_name == "pawn":
        mover_doubled = mover_ps.get("doubled", 0) or 0
        if mover_doubled > 0:
            lead = f"{mover} recaptures with {san}, accepting doubled pawns in exchange for the open file."
            support = "Doubled pawns are not always weak — the open file and piece activity can more than compensate."
            return CommentaryPoint("pawn_doubled_tradeoff", "strategy", 9, lead, support, ["doubled pawns", "open file", "compensation"])

    return None


def _piece_coordination_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    """Comment on piece coordination patterns."""
    if not move:
        return None
    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    piece_name = str(move_info.get("piece", "piece"))
    mover_color = prev_board.turn

    # Rook doubling on a file
    if piece_name == "rook":
        dest_file = chess.square_file(move.to_square)
        rooks = list(curr_board.pieces(chess.ROOK, mover_color))
        if len(rooks) >= 2:
            other_rook = [r for r in rooks if r != move.to_square]
            if other_rook and chess.square_file(other_rook[0]) == dest_file:
                file_letter = chr(ord('a') + dest_file)
                lead = f"{mover} doubles the rooks on the {file_letter}-file with {san} — a powerful configuration."
                support = "Doubled rooks create irresistible pressure along the file and often lead to decisive penetration."
                return CommentaryPoint("coord_doubled_rooks", "strategy", 8, lead, support, ["doubled rooks", f"{file_letter}-file", "pressure"])

    # Rook reaching the 7th rank
    if piece_name == "rook":
        seventh = 6 if mover_color == chess.WHITE else 1
        if chess.square_rank(move.to_square) == seventh:
            lead = f"{mover} invades the seventh rank with {san}, reaching the ideal position for a rook."
            support = "A rook on the 7th rank cuts off the enemy king, attacks pawns from behind, and often decides the game."
            return CommentaryPoint("coord_rook_seventh", "strategy", 8, lead, support, ["rook", "7th rank", "invasion"])

    # Queen + Bishop battery formation
    if piece_name in ("queen", "bishop"):
        queens = list(curr_board.pieces(chess.QUEEN, mover_color))
        bishops = list(curr_board.pieces(chess.BISHOP, mover_color))
        if queens and bishops:
            q_sq = queens[0]
            for b_sq in bishops:
                sq1 = move.to_square
                other_sq = b_sq if piece_name == "queen" else q_sq
                if abs(chess.square_file(sq1) - chess.square_file(other_sq)) == abs(chess.square_rank(sq1) - chess.square_rank(other_sq)):
                    enemy_king = curr_board.king(not mover_color)
                    if enemy_king:
                        if abs(chess.square_file(sq1) - chess.square_file(enemy_king)) == abs(chess.square_rank(sq1) - chess.square_rank(enemy_king)):
                            lead = f"{mover} creates a deadly queen-bishop battery with {san}, aimed straight at the king."
                            support = "This diagonal formation is one of the most dangerous attacking configurations in chess."
                            return CommentaryPoint("coord_battery", "attack", 7, lead, support, ["battery", "diagonal", "king"])

    return None


def _endgame_technique_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    """Comment on endgame-specific ideas."""
    if not move:
        return None
    analysis = context.get("analysis", {}) or {}
    game_phase = str(analysis.get("game_phase", ""))
    if game_phase != "endgame":
        return None

    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    piece_name = str(move_info.get("piece", "piece"))
    mover_color = prev_board.turn
    opponent = _opponent_name(mover)
    pawn_structure = analysis.get("pawn_structure", {}) or {}
    mover_key = "white" if mover == "White" else "black"
    mover_ps = pawn_structure.get(mover_key, {}) or {}

    # King centralization
    if piece_name == "king":
        dest_rank = chess.square_rank(move.to_square)
        dest_file = chess.square_file(move.to_square)
        # Moving toward center
        center_dist = abs(dest_file - 3.5) + abs(dest_rank - 3.5)
        prev_rank = chess.square_rank(move.from_square)
        prev_file = chess.square_file(move.from_square)
        prev_center_dist = abs(prev_file - 3.5) + abs(prev_rank - 3.5)
        if center_dist < prev_center_dist:
            lead = f"{mover} activates the king with {san}, marching it toward the center."
            support = "In the endgame, the king transforms from a liability into a powerful fighting piece — centralization is paramount."
            return CommentaryPoint("endgame_king_centralize", "endgame", 9, lead, support, ["king", "centralization", "endgame"])

    # Passed pawn advancement
    if piece_name == "pawn":
        dest_name = chess.square_name(move.to_square)
        passed = mover_ps.get("passed", []) or []
        if dest_name in passed:
            rank = chess.square_rank(move.to_square)
            adv_rank = rank if mover_color == chess.WHITE else (7 - rank)
            if adv_rank >= 5:
                lead = f"{mover} pushes the passed pawn to {dest_name} with {san} — it is becoming very dangerous."
                support = f"{opponent} must commit material to stop it, which creates tactical opportunities elsewhere."
                return CommentaryPoint("endgame_passed_push", "endgame", 7, lead, support, ["passed pawn", dest_name, "promotion"])
            else:
                lead = f"{mover} advances the passed pawn with {san}, increasing the pressure."
                support = "Every square the passed pawn advances makes it exponentially harder to stop."
                return CommentaryPoint("endgame_passed_advance", "endgame", 9, lead, support, ["passed pawn", "advance", dest_name])

    # Piece activity in endgame
    if piece_name == "rook":
        # Rook behind a passed pawn
        dest_file = chess.square_file(move.to_square)
        for pp_name in (mover_ps.get("passed") or []):
            try:
                pp_sq = chess.parse_square(pp_name)
                pp_file = chess.square_file(pp_sq)
                if pp_file == dest_file:
                    rank_diff = chess.square_rank(move.to_square) - chess.square_rank(pp_sq)
                    behind = (rank_diff < 0 and mover_color == chess.WHITE) or (rank_diff > 0 and mover_color == chess.BLACK)
                    if behind:
                        lead = f"{mover} places the rook behind the passed pawn with {san} — the ideal configuration."
                        support = "Tarrasch's rule: rooks belong behind passed pawns, where they gain power as the pawn advances."
                        return CommentaryPoint("endgame_rook_behind_pp", "endgame", 8, lead, support, ["rook", "passed pawn", "Tarrasch"])
            except ValueError:
                continue

    return None


def _prophylaxis_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    """Identify prophylactic moves — moves that prevent the opponent's plan."""
    if not move:
        return None
    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    piece_name = str(move_info.get("piece", "piece"))
    tactical = context.get("tactical", {}) or {}
    attack = tactical.get("attack_info") or {}
    verified = context.get("verified_facts", {}) or {}
    opponent = _opponent_name(mover)
    mover_color = prev_board.turn
    quality = str(context.get("quality", ""))

    # Only consider prophylaxis for good/excellent/best moves without direct tactical purpose
    if quality in ("blunder", "mistake", "inaccuracy"):
        return None
    if attack.get("is_attacking") or verified.get("captures") or verified.get("gives_check"):
        return None

    # Prophylactic pawn move — blocking an opponent's pawn break or advance
    if piece_name == "pawn":
        dest_sq = move.to_square
        dest_file = chess.square_file(dest_sq)
        dest_rank = chess.square_rank(dest_sq)
        # Check if this pawn blocks a key square from the opponent
        opponent_color = not mover_color
        # Does this pawn now prevent an opponent pawn advance?
        block_rank = dest_rank + 1 if opponent_color == chess.WHITE else dest_rank - 1
        if 0 <= block_rank < 8:
            for df in [-1, 0, 1]:
                check_file = dest_file + df
                if 0 <= check_file < 8:
                    check_sq = chess.square(check_file, block_rank)
                    p = prev_board.piece_at(check_sq)
                    if p and p.piece_type == chess.PAWN and p.color == opponent_color:
                        # Our pawn advance restricts the opponent's pawn
                        lead = f"{mover} plays the prophylactic {san}, restricting {opponent}'s pawn expansion."
                        support = "The best moves often prevent the opponent's ideal plan rather than advancing your own."
                        return CommentaryPoint("prophylaxis_pawn", "strategy", 10, lead, support, ["prophylaxis", "restrict", "pawn"])

    # Prophylactic piece move — covering a key square the opponent wants
    if piece_name in ("knight", "bishop"):
        # Check if the piece now controls an important square that was previously weak
        dest_name = chess.square_name(move.to_square)
        analysis = context.get("analysis", {}) or {}
        mover_key = "white" if mover == "White" else "black"
        own_weak = (analysis.get("weak_squares", {}) or {}).get(mover_key, []) or []
        squares_now_defended = [chess.square_name(sq) for sq in curr_board.attacks(move.to_square)]
        covered_weakness = [sq for sq in own_weak if sq in squares_now_defended]
        if covered_weakness:
            lead = f"{mover} plays the prophylactic {san}, covering the weak {covered_weakness[0]} square."
            support = "Preventing the opponent from exploiting structural weaknesses is a hallmark of strong positional play."
            return CommentaryPoint("prophylaxis_cover", "strategy", 10, lead, support, ["prophylaxis", covered_weakness[0], "defence"])

    return None


def _blockade_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    """Detect when a piece is placed directly in front of an enemy passed pawn.

    A blockader neutralises the pawn's advance.  Knights are the ideal blockaders
    (they remain active and cannot be pushed away by the pawn itself), so they get
    specific praise.  Any other piece in front of a passed pawn is also noteworthy.
    """
    if not move:
        return None

    mover_color = prev_board.turn
    enemy_color = not mover_color
    move_info = context.get("move", {})
    piece_name = str(move_info.get("piece", "piece"))

    # Kings and pawns are not blockaders in the strategic sense
    if piece_name in {"king", "pawn"}:
        return None

    dest_file = chess.square_file(move.to_square)
    dest_rank = chess.square_rank(move.to_square)

    # The passed pawn must be directly behind the moved piece (one rank back
    # in the direction the mover advances).
    pawn_rank = dest_rank - 1 if mover_color == chess.WHITE else dest_rank + 1
    if not (0 <= pawn_rank < 8):
        return None

    pawn_sq = chess.square(dest_file, pawn_rank)
    pawn_piece = curr_board.piece_at(pawn_sq)
    if not (pawn_piece and pawn_piece.piece_type == chess.PAWN and pawn_piece.color == enemy_color):
        return None

    # Confirm the pawn is actually passed via the pawn-structure analysis
    analysis = context.get("analysis", {}) or {}
    pawn_structure = analysis.get("pawn_structure", {}) or {}
    opp_key = "black" if mover_color == chess.WHITE else "white"
    opp_ps = pawn_structure.get(opp_key, {}) or {}
    passed = opp_ps.get("passed", []) or []
    if chess.square_name(pawn_sq) not in passed:
        return None

    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    opponent = _opponent_name(mover)
    pawn_sq_name = chess.square_name(pawn_sq)
    dest_sq_name = chess.square_name(move.to_square)

    if piece_name == "knight":
        lead = (
            f"{mover} blockades {opponent}'s passed pawn with {san}, "
            f"planting the knight on {dest_sq_name} directly in front of the pawn on {pawn_sq_name}."
        )
        support = (
            "A knight is the ideal blockader — it neutralises the pawn's advance while "
            "remaining active, and the pawn itself cannot drive it away."
        )
    else:
        lead = (
            f"{mover} blockades {opponent}'s passed pawn with {san}, "
            f"placing the {piece_name} on {dest_sq_name} to stop it in its tracks."
        )
        support = (
            f"The passed pawn on {pawn_sq_name} is frozen until the blockader is removed — "
            f"{opponent} must invest material to dislodge it."
        )

    return CommentaryPoint(
        "blockade", "strategy", 9,
        lead, support,
        ["blockade", piece_name, pawn_sq_name]
    )


def _strategic_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> Optional[CommentaryPoint]:
    if not move:
        return None

    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    piece_name = str(move_info.get("piece", "piece"))
    analysis = context.get("analysis", {}) or {}
    themes = analysis.get("strategic_themes") or []
    verified = context.get("verified_facts", {}) or {}
    quality = str(context.get("quality", "") or "")
    if quality in {"inaccuracy", "mistake", "blunder"} and (context.get("tactical", {}) or {}).get("hung_piece"):
        return None
    mover_color = prev_board.turn
    pawn_structure = analysis.get("pawn_structure", {}) or {}
    king_safety = analysis.get("king_safety", {}) or {}
    weak_squares = analysis.get("weak_squares", {}) or {}
    piece_activity = analysis.get("piece_activity", {}) or {}

    # Rook placement on open/semi-open files
    if piece_name == "rook":
        file_state = _describe_file_state(curr_board, move.to_square, mover_color)
        file_name = chess.square_name(move.to_square)[0]
        if file_state:
            lead = f"{mover} improves the rook with {san}, lining it up on the {file_state} {file_name}-file."
            support = "Quiet rook moves like that rarely win on the spot, but they make later central operations much easier."
            return CommentaryPoint("strategy_rook_file", "strategy", 11, lead, support, ["rook", file_state, f"{file_name}-file"])
        if file_name in {"d", "e"}:
            lead = f"{mover} centralizes the rook with {san}, improving the coordination of the heavy pieces."
            support = "That is the kind of small improvement that strengthens future pawn breaks and tactical pressure."
            return CommentaryPoint("strategy_rook_central", "strategy", 11, lead, support, ["rook", "central", "coordination"])

    # Piece activity improvement
    mobility_delta = _mobility_delta(prev_board, curr_board, move)
    if piece_name in {"knight", "bishop", "queen", "rook"} and mobility_delta >= 2:
        square_name = chess.square_name(move.to_square)
        central_targets = _join_square_names(verified.get("central_control", []))
        lead = f"{mover} improves the {piece_name} with {san}, giving it a much more active post on {square_name}."
        if central_targets:
            support = f"From there it influences {central_targets}, so the move is more purposeful than it first appears."
            keywords = [piece_name, square_name, central_targets.split(",")[0]]
        else:
            support = "There is no immediate tactic attached to it, but the piece simply does more useful work from that square."
            keywords = [piece_name, square_name, "active"]
        return CommentaryPoint("strategy_piece_improvement", "strategy", 12, lead, support, keywords)

    # Weak squares exploitation
    opponent_color_key = "black" if mover == "White" else "white"
    opp_weak = weak_squares.get(opponent_color_key, []) or []
    if opp_weak and piece_name in {"knight", "bishop", "queen"}:
        dest_name = chess.square_name(move.to_square)
        if dest_name in opp_weak:
            lead = f"{mover}'s {san} plants the {piece_name} on the weak square {dest_name}, where it cannot be easily dislodged."
            support = "Occupying the opponent's structural weaknesses is one of the most reliable ways to build a lasting advantage."
            return CommentaryPoint("strategy_weak_square", "strategy", 11, lead, support, [piece_name, dest_name, "outpost"])

    # Pawn structure context from the analysis
    mover_color_key = "white" if mover == "White" else "black"
    mover_ps = pawn_structure.get(mover_color_key, {}) or {}
    opp_ps = pawn_structure.get(opponent_color_key, {}) or {}
    if piece_name == "pawn":
        # Check if this pawn move creates a passed pawn
        dest = move.to_square
        dest_name = chess.square_name(dest)
        if dest_name in (mover_ps.get("passed") or []):
            lead = f"{mover} creates a passed pawn with {san} — a major strategic achievement."
            support = "Passed pawns are inherently dangerous because they tie down enemy pieces to blockade duties."
            return CommentaryPoint("strategy_passed_pawn", "strategy", 10, lead, support, ["passed pawn", dest_name, "blockade"])

    # King safety as strategic theme 
    opp_ks = king_safety.get(opponent_color_key, {}) or {}
    if str(opp_ks.get("level", "")) in ("vulnerable", "exposed"):
        if piece_name in {"queen", "rook", "bishop", "knight"}:
            lead = f"{mover} repositions the {piece_name} with {san}, building pressure against the exposed king."
            support = "With the opposing king's shelter compromised, every piece aimed at the king zone multiplies the danger."
            return CommentaryPoint("strategy_king_attack", "strategy", 11, lead, support, [piece_name, "king safety", "attack"])

    # Strategic theme from analysis
    if themes:
        theme = str(themes[0]).strip().rstrip(".")
        lead = theme + "."
        support = "That strategic idea matters because it shapes which plans are realistic over the next few moves."
        return CommentaryPoint("strategy_theme", "strategy", 13, lead, support, _keywords_from_text(theme))

    return None


def _general_point(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: Optional[chess.Move],
    context: Dict[str, Any],
    previous_blueprint: Optional[CommentaryBlueprint],
) -> CommentaryPoint:
    move_info = context.get("move", {})
    mover = str(move_info.get("mover", "Side"))
    san = str(move_info.get("san", "the move"))
    piece_name = str(move_info.get("piece", "piece"))
    verified = context.get("verified_facts", {}) or {}
    analysis = context.get("analysis", {}) or {}
    quality = str(context.get("quality", "") or "")
    central_targets = _join_square_names(verified.get("central_control", []))
    themes = analysis.get("strategic_themes") or []
    pawn_structure = analysis.get("pawn_structure", {}) or {}

    if quality == "inaccuracy":
        lead = f"{mover} chooses {san}, but it is not the most accurate continuation."
        support = "The move is playable, yet it lets the better plan slip away."
        return CommentaryPoint("general_inaccuracy", "general", 99, lead, support, ["inaccuracy", piece_name, "accuracy"])

    # Try to extract something meaningful from strategic themes
    if themes:
        theme = str(themes[0]).strip().rstrip(".")
        lead = f"{mover} plays {san}. {theme}."
        support = "Understanding the strategic landscape is key to finding the right plan."
        keywords = _keywords_from_text(theme)
        if not keywords:
            keywords = [piece_name, "strategy"]
        return CommentaryPoint("general_theme", "general", 99, lead, support, keywords)

    if central_targets:
        lead = f"{mover} uses {san} to improve the {piece_name} and increase control over {central_targets}."
        support = "Even without a direct tactic, that kind of central influence usually makes the next decision easier."
        keywords = [piece_name, "center", central_targets.split(",")[0]]
    elif move and piece_name == "pawn":
        # Pawn moves — comment on structure implications
        dest_file = chr(ord('a') + chess.square_file(move.to_square))
        lead = f"{mover} advances the {dest_file}-pawn with {san}, altering the pawn structure."
        support = "Every pawn move is permanent and changes the character of the position — for better or worse."
        keywords = ["pawn", "structure", f"{dest_file}-file"]
    elif move and piece_name == "king":
        lead = f"{mover} moves the king with {san}, finding a safer or more active location."
        support = "King placement is always a strategic consideration and shapes the plans available to both sides."
        keywords = ["king", "safety", "active"]
    else:
        lead = f"{mover} plays {san} to improve the coordination of the pieces."
        if piece_name == "knight":
            support = "The knight is looking for a better outpost — no immediate threat, but it improves the piece structure."
            keywords = [piece_name, "outpost", "coordination"]
        elif piece_name == "bishop":
            support = "The bishop shift prepares to exploit a diagonal if lines open; bishops improve dramatically when pawns get out of their way."
            keywords = [piece_name, "diagonal", "coordination"]
        elif piece_name == "rook":
            support = "Quiet rook moves accumulate value — the rook waits on a good file for the right moment to become decisive."
            keywords = [piece_name, "file", "coordination"]
        elif piece_name == "queen":
            support = "Queen centralisation increases scope without committing to a concrete plan yet."
            keywords = [piece_name, "central", "scope"]
        else:
            support = "Small improvements like this accumulate — there is no single decisive moment, but the position becomes healthier."
            keywords = [piece_name, "coordination", "improvement"]
    return CommentaryPoint("general_improvement", "general", 99, lead, support, keywords)


def _build_verdict(context: Dict[str, Any], lead_point: CommentaryPoint) -> str:
    eval_info = context.get("eval", {}) or {}
    move_info = context.get("move", {}) or {}
    mover = str(move_info.get("mover", "Side"))
    opponent = _opponent_name(mover)
    quality = str(context.get("quality", ""))

    if quality == "blunder":
        return f"{opponent} should be clearly better after that."
    if quality == "mistake":
        return f"{opponent} comes out of the sequence with the easier game."
    if eval_info.get("is_turning_point"):
        swing = int(abs(eval_info.get("swing", 0)))
        if swing >= 180:
            return f"The evaluation swings sharply here, so the move has real practical consequences."
    cp_after = int(eval_info.get("cp_after", 0) or 0)
    if lead_point.family in {"tactic", "attack", "defence"} and abs(cp_after) >= 260:
        better = "White" if cp_after > 0 else "Black"
        return f"{better} should be the one pressing after that sequence."
    return ""


def _describe_tactic_targets(board: chess.Board, square_names: Sequence[str]) -> List[str]:
    targets = []
    for name in square_names:
        try:
            square = chess.parse_square(str(name))
        except ValueError:
            continue
        piece = board.piece_at(square)
        if piece:
            targets.append(f"the {chess.piece_name(piece.piece_type)} on {name}")
    return targets


def _describe_file_state(board: chess.Board, square: int, color: chess.Color) -> str:
    file_index = chess.square_file(square)
    friendly_pawns = 0
    enemy_pawns = 0
    for rank in range(8):
        piece = board.piece_at(chess.square(file_index, rank))
        if not piece or piece.piece_type != chess.PAWN:
            continue
        if piece.color == color:
            friendly_pawns += 1
        else:
            enemy_pawns += 1
    if friendly_pawns == 0 and enemy_pawns == 0:
        return "open"
    if friendly_pawns == 0:
        return "semi-open"
    return ""


def _mobility_delta(prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> int:
    before_piece = prev_board.piece_at(move.from_square)
    after_piece = curr_board.piece_at(move.to_square)
    if not before_piece or not after_piece:
        return 0
    before = len(prev_board.attacks(move.from_square))
    after = len(curr_board.attacks(move.to_square))
    return after - before


def _normalize_opening_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned or cleaned.lower() in {"unknown", "starting position"}:
        return ""
    return cleaned.split(":", 1)[0].strip()


def _same_family(previous_blueprint: Optional[CommentaryBlueprint], family: str) -> bool:
    return bool(previous_blueprint and previous_blueprint.family == family)


def _opponent_name(mover: str) -> str:
    return "Black" if mover == "White" else "White"


def _ensure_sentence(text: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _join_square_names(values: Iterable[str]) -> str:
    squares = [str(value) for value in values if value]
    if not squares:
        return ""
    if len(squares) == 1:
        return squares[0]
    if len(squares) == 2:
        return f"{squares[0]} and {squares[1]}"
    return f"{', '.join(squares[:-1])}, and {squares[-1]}"


def _preferred_supported_piece(entries: Iterable[str]) -> str:
    ordered = [str(entry).lower() for entry in entries if entry]
    for priority in ("pawn on e4", "pawn on d4", "pawn on e5", "pawn on d5"):
        for entry in ordered:
            if priority in entry:
                return entry
    for entry in ordered:
        if any(name in entry for name in {"pawn", "bishop", "knight"}):
            return entry
    return ""


def _expected_recapture_details(board: chess.Board, square: int) -> Dict[str, str]:
    color = board.turn
    chosen = _preferred_recapture_move(board, square, color)
    if not chosen:
        return {}

    san = board.san(chosen)
    before = set(_doubled_files(board, color))
    probe = board.copy(stack=False)
    probe.push(chosen)
    after = set(_doubled_files(probe, color))
    new_files = sorted(after - before)
    details = {"san": san}
    if new_files:
        details["doubled_file"] = new_files[0]
    return details


def _doubled_files(board: chess.Board, color: chess.Color) -> List[str]:
    files = [0] * 8
    for square in board.pieces(chess.PAWN, color):
        files[chess.square_file(square)] += 1
    return [chess.FILE_NAMES[index] for index, count in enumerate(files) if count > 1]


def _preferred_recapture_move(board: chess.Board, square: int, color: chess.Color) -> Optional[chess.Move]:
    recaptures = [move for move in board.legal_moves if move.to_square == square and board.is_capture(move)]
    if not recaptures:
        return None
    return max(recaptures, key=lambda move: _recapture_score(board, move, color))


def _recapture_score(board: chess.Board, move: chess.Move, color: chess.Color) -> float:
    piece = board.piece_at(move.from_square)
    if not piece:
        return float("-inf")

    before_files = set(_doubled_files(board, color))
    probe = board.copy(stack=False)
    probe.push(move)
    after_files = set(_doubled_files(probe, color))
    new_doubled = len(after_files - before_files)

    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.2,
        chess.BISHOP: 3.3,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 100.0,
    }
    score = 100.0
    score -= new_doubled * 100.0
    score -= piece_values.get(piece.piece_type, 0.0)
    return score


def _first_matching_piece(entries: Iterable[str], piece_names: set[str]) -> str:
    for entry in entries:
        lower_entry = str(entry).lower()
        for name in piece_names:
            if name in lower_entry:
                return lower_entry
    return ""


def _unique_keywords(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result[:6]


def _keywords_from_text(text: str) -> List[str]:
    candidates = []
    for token in str(text or "").replace(".", " ").replace(",", " ").split():
        stripped = token.strip().lower()
        if not stripped:
            continue
        if stripped in KEYWORD_STOPWORDS:
            continue
        if len(stripped) <= 2 and stripped not in {"d4", "d5", "e4", "e5", "c6", "c3"}:
            continue
        candidates.append(stripped)
    return _unique_keywords(candidates)[:3]