from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _prioritize_site_packages() -> None:
    cwd = os.getcwd()
    site_packages = next((path for path in sys.path if "site-packages" in path), "")
    if not site_packages:
        return
    sys.path = [site_packages] + [
        path for path in sys.path if path != site_packages and path not in {"", cwd}
    ] + [cwd]


_prioritize_site_packages()
os.environ.setdefault("DISABLE_KITTENTTS", "1")

import chess  # noqa: E402
import chess.pgn  # noqa: E402
import numpy as np  # noqa: E402

from app import (  # noqa: E402
    OPENING_BOOK,
    _build_analysis_after,
    classify_move_quality,
    detect_missed_tactics,
    generate_fallback_commentary,
    build_rich_context,
    get_commentary_eval,
)

try:  # noqa: E402
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None
    st_util = None

try:  # noqa: E402
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover - optional dependency
    rouge_scorer = None

try:  # noqa: E402
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:  # pragma: no cover - optional dependency
    SmoothingFunction = None
    sentence_bleu = None

try:  # noqa: E402
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None
    cosine_similarity = None


TOKEN_RE = re.compile(r"[a-z0-9+#=:-]+")
SQUARE_RE = re.compile(r"\b[a-h][1-8]\b")
SAN_TOKEN_RE = re.compile(r"\b(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b", re.IGNORECASE)
COMMENT_TAG_RE = re.compile(r"\[%[^\]]+\]")

PIECE_WORDS = {"king", "queen", "rook", "bishop", "knight", "pawn"}
TACTICAL_WORDS = {
    "attack", "attacks", "attacking", "pressure", "pressures", "pressing", "threat",
    "threatens", "defend", "defends", "defense", "defence", "defensive", "protect",
    "pin", "pins", "pinned", "fork", "forks", "skewer", "skewers", "x-ray", "xray",
    "recapture", "trade", "trades", "trading", "exchange", "simplify", "simplifies",
    "retreat", "retreats", "retreating", "develop", "develops", "development", "center",
    "central", "castle", "castles", "castling", "check", "checkmate", "mate", "blunder",
    "mistake", "inaccuracy", "queenside", "kingside", "doubled", "isolated", "passed",
    "space", "initiative",
}

CLAIM_PATTERNS = {
    "checkmate": [r"\bcheckmate\b", r"\bmate\b"],
    "check": [r"\bcheck\b", r"\bchecks\b"],
    "castling": [r"\bcastle", r"\bcastling\b"],
    "capture": [r"\bcaptur", r"\btakes?\b", r"\bpicks? up\b", r"\bwins? the\b"],
    "recapture": [r"\brecaptur", r"\btake[s]? back\b"],
    "trade": [r"\btrade", r"\bexchange", r"\bsimplif"],
    "retreat": [r"\bretreat", r"\bbacks? up\b", r"\bpulls? back\b", r"\bsteps? back\b"],
    "development": [r"\bdevelop"],
    "center": [r"\bcenter\b", r"\bcentral\b"],
    "support": [r"\bsupport", r"\breinforc"],
    "attack": [r"\battack", r"\bpressure", r"\bthreat", r"\basks? questions"],
    "defense": [r"\bdefen[sc]e\b", r"\bdefend", r"\bprotect", r"\brescu"],
    "pin": [r"\bpin", r"\bpinned\b"],
    "fork": [r"\bfork"],
    "skewer": [r"\bskewer"],
    "xray": [r"\bx-?ray"],
    "doubled_pawns": [r"\bdoubled pawns?\b", r"\bdoubled pawn\b"],
    "isolated_pawn": [r"\bisolated pawns?\b", r"\bisolated pawn\b"],
    "passed_pawn": [r"\bpassed pawns?\b", r"\bpassed pawn\b"],
    "queenside": [r"\bqueenside\b"],
    "kingside": [r"\bkingside\b"],
    "bad_move": [r"\bblunder\b", r"\bmistake\b", r"\binaccuracy\b"],
    "queen_loss": [r"\bhanging the queen\b", r"\bwins the queen\b", r"\bqueen\b"],
}

ROUGE_L_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if rouge_scorer is not None else None
BLEU_SMOOTHER = SmoothingFunction().method1 if SmoothingFunction is not None else None


@dataclass
class MoveSample:
    corpus_label: str
    source_label: str
    game_id: str
    game_signature: str
    white: str
    black: str
    ply: int
    san: str
    opening_name: str
    phase: str
    commentary: str
    quality: str
    truth_tags: List[str]
    claim_tags: List[str]
    grounded_precision: float
    grounded_recall: float
    grounded_f1: float


@dataclass
class PairSample:
    corpus_label: str
    game_id: str
    game_signature: str
    white: str
    black: str
    ply: int
    san: str
    opening_name: str
    phase: str
    quality: str
    reference_label: str
    candidate_label: str
    reference_text: str
    candidate_text: str
    bleu4: float
    semantic_similarity: float
    rouge_l_f1: float
    token_f1: float
    entity_f1: float
    claim_tag_jaccard: float
    reference_grounded_f1: float
    candidate_grounded_f1: float
    overall_alignment: float


class TextSimilarityBackend:
    def __init__(self, backend: str = "auto", model_name: str = "all-mpnet-base-v2") -> None:
        self.mode = "token-fallback"
        self.model = None
        self.cache: Dict[str, Any] = {}
        self.backend = backend
        self.model_name = model_name

        if backend == "token":
            self.mode = "token-fallback"
            return

        if backend == "tfidf":
            self.mode = "tfidf"
            return

        if SentenceTransformer is not None and backend in {"auto", "sentence-transformers"}:
            candidate_models = [model_name]
            if backend == "auto":
                candidate_models.extend(
                    fallback
                    for fallback in ("all-MiniLM-L6-v2", "all-mpnet-base-v2")
                    if fallback not in candidate_models
                )
            for candidate in candidate_models:
                try:
                    self.model = SentenceTransformer(candidate, local_files_only=True)
                    self.mode = f"sentence-transformers:{candidate}"
                    break
                except Exception:
                    self.model = None

        if self.model is None and backend in {"auto", "sentence-transformers"}:
            self.mode = "tfidf" if TfidfVectorizer is not None and cosine_similarity is not None else "token-fallback"

    def cosine(self, left: str, right: str) -> float:
        left = normalize_commentary(left)
        right = normalize_commentary(right)
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0

        if self.model is not None and st_util is not None:
            left_emb = self._encode(left)
            right_emb = self._encode(right)
            return float(max(0.0, min(1.0, st_util.cos_sim(left_emb, right_emb).item())))

        if self.mode == "tfidf" and TfidfVectorizer is not None and cosine_similarity is not None:
            vect = TfidfVectorizer(ngram_range=(1, 2))
            matrix = vect.fit_transform([left, right])
            return float(max(0.0, min(1.0, cosine_similarity(matrix[0:1], matrix[1:2])[0][0])))

        return token_f1_score(left, right)

    def precompute(self, texts: Iterable[str]) -> None:
        if self.model is None or st_util is None:
            return
        unique_texts = []
        seen = set()
        for text in texts:
            normalized = normalize_commentary(text)
            if not normalized or normalized in self.cache or normalized in seen:
                continue
            seen.add(normalized)
            unique_texts.append(normalized)
        if not unique_texts:
            return
        embeddings = self.model.encode(
            unique_texts,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False,
        )
        for text, embedding in zip(unique_texts, embeddings):
            self.cache[text] = embedding

    def _encode(self, text: str) -> Any:
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        return self.cache[text]


class CommentaryEvaluator:
    def __init__(
        self,
        use_gemini: bool = False,
        semantic_backend: str = "auto",
        semantic_model: str = "all-mpnet-base-v2",
        include_appendix: bool = False,
    ) -> None:
        self.use_gemini = use_gemini
        self.similarity = TextSimilarityBackend(backend=semantic_backend, model_name=semantic_model)
        self.system_sid = "evaluation-script"
        self.system_source_label = "system"
        self.include_appendix = include_appendix

    def evaluate(
        self,
        grandmaster_path: Path,
        chessdotcom_path: Path,
        output_dir: Path,
        max_games: Optional[int] = None,
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        gm_corpus = self._load_corpus(grandmaster_path, "grandmaster", max_games=max_games)
        chessdotcom_corpus = self._load_corpus(chessdotcom_path, "chessdotcom", max_games=max_games)

        gm_reference_samples: List[MoveSample] = []
        system_samples: List[MoveSample] = []
        baseline_samples: List[MoveSample] = []
        system_vs_gm_pairs: List[PairSample] = []
        baseline_vs_gm_pairs: List[PairSample] = []
        generated_by_signature: Dict[str, Dict[int, Dict[str, Any]]] = {}
        gm_games_by_signature: Dict[str, Dict[str, Any]] = {}

        for game in gm_corpus["games"]:
            generated = self._generate_system_commentary_for_game(game)
            generated_by_signature[game["signature"]] = generated
            gm_games_by_signature[game["signature"]] = game
            reference_samples, candidate_samples, pairs = self._score_reference_against_system(
                "grandmaster",
                game,
                generated,
            )
            gm_reference_samples.extend(reference_samples)
            system_samples.extend(candidate_samples)
            system_vs_gm_pairs.extend(pairs)

        self._finalize_pair_metrics(system_vs_gm_pairs)

        chess_games_by_signature = {game["signature"]: game for game in chessdotcom_corpus["games"]}
        overlap_signatures = sorted(set(gm_games_by_signature).intersection(chess_games_by_signature))

        for signature in overlap_signatures:
            gm_game = gm_games_by_signature[signature]
            chessdotcom_game = chess_games_by_signature[signature]
            candidate_samples, pairs = self._score_candidate_against_reference(
                reference_game=gm_game,
                candidate_game=chessdotcom_game,
                generated=generated_by_signature[signature],
                candidate_label="chessdotcom",
            )
            baseline_samples.extend(candidate_samples)
            baseline_vs_gm_pairs.extend(pairs)

        self._finalize_pair_metrics(baseline_vs_gm_pairs)

        overlap_keys = {(pair.game_signature, pair.ply) for pair in baseline_vs_gm_pairs}
        system_overlap_pairs = [
            pair for pair in system_vs_gm_pairs if (pair.game_signature, pair.ply) in overlap_keys
        ]

        summary = self._build_paper_summary(
            gm_corpus=gm_corpus,
            chessdotcom_corpus=chessdotcom_corpus,
            gm_reference_samples=gm_reference_samples,
            system_samples=system_samples,
            baseline_samples=baseline_samples,
            system_vs_gm_pairs=system_vs_gm_pairs,
            system_overlap_pairs=system_overlap_pairs,
            baseline_vs_gm_pairs=baseline_vs_gm_pairs,
        )

        self._write_paper_outputs(
            output_dir=output_dir,
            summary=summary,
            system_vs_gm_pairs=system_vs_gm_pairs,
            system_overlap_pairs=system_overlap_pairs,
            baseline_vs_gm_pairs=baseline_vs_gm_pairs,
        )
        return summary

    def _load_corpus(self, path: Path, label: str, max_games: Optional[int]) -> Dict[str, Any]:
        games = []
        with path.open("r", encoding="utf-8") as handle:
            count = 0
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                count += 1
                if max_games and count > max_games:
                    break
                games.append(self._parse_game(game, label))
        return {"label": label, "path": str(path), "games": games}

    def _parse_game(self, game: chess.pgn.Game, label: str) -> Dict[str, Any]:
        board = game.board()
        moves: List[Dict[str, Any]] = []
        comments_by_ply: Dict[int, str] = {}
        pgn_prefix = ""
        opening_by_ply: Dict[int, str] = {}
        current_opening = normalize_opening_name(game.headers.get("Opening", "")) or "Unknown"

        for node in game.mainline():
            san = board.san(node.move)
            pgn_prefix += f"{board.fullmove_number}. {san} " if board.turn == chess.WHITE else f"{san} "
            if pgn_prefix.strip() in OPENING_BOOK:
                current_opening = OPENING_BOOK[pgn_prefix.strip()]
            opening_by_ply[node.ply()] = current_opening
            comment = normalize_commentary(node.comment)
            if comment:
                comments_by_ply[node.ply()] = comment
            moves.append({"ply": node.ply(), "san": san, "move": node.move, "fen_before": board.fen()})
            board.push(node.move)

        signature_text = " ".join(item["san"] for item in moves)
        signature = hashlib.sha1(signature_text.encode("utf-8")).hexdigest()[:12]
        game_id = f"{label}:{signature}"
        return {
            "game_id": game_id,
            "signature": signature,
            "white": game.headers.get("White", "?"),
            "black": game.headers.get("Black", "?"),
            "headers": dict(game.headers),
            "moves": moves,
            "comments_by_ply": comments_by_ply,
            "opening_by_ply": opening_by_ply,
        }

    def _generate_system_commentary_for_game(self, game: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        generated: Dict[int, Dict[str, Any]] = {}
        previous_move: Optional[chess.Move] = None
        previous_blueprint: Optional[Dict[str, Any]] = None

        for move_row in game["moves"]:
            prev_board = chess.Board(move_row["fen_before"])
            move = move_row["move"]
            curr_board = prev_board.copy(stack=False)
            curr_board.push(move)
            cp_before = get_commentary_eval(self.system_sid, prev_board.fen())
            cp_after = get_commentary_eval(self.system_sid, curr_board.fen())
            quality = classify_move_quality(cp_before, cp_after, prev_board.turn == chess.WHITE)
            missed_tactics = {}
            if quality in {"mistake", "blunder"}:
                try:
                    missed_tactics = detect_missed_tactics(
                        self.system_sid,
                        prev_board,
                        move,
                        cp_before,
                        cp_after,
                        quality,
                        tactics_enabled=True,
                    )
                except Exception:
                    missed_tactics = {}

            analysis_after = _build_analysis_after(self.system_sid, prev_board, curr_board, move, True)
            context = build_rich_context(
                prev_board=prev_board,
                curr_board=curr_board,
                move=move,
                move_san=move_row["san"],
                analysis_after=analysis_after,
                cp_before=cp_before,
                cp_after=cp_after,
                opening_name=game["opening_by_ply"].get(move_row["ply"], "Unknown"),
                missed_tactics=missed_tactics,
                tactics_enabled=True,
                previous_move=previous_move,
                previous_blueprint=previous_blueprint,
                sid="",
            )
            commentary = generate_fallback_commentary(context)
            truth_tags = derive_truth_tags(prev_board, curr_board, move, context)
            claim_tags = sorted(extract_claim_tags(commentary))
            precision, recall, f1 = claim_score(claim_tags, truth_tags)
            generated[move_row["ply"]] = {
                "commentary": commentary,
                "context": context,
                "truth_tags": truth_tags,
                "claim_tags": claim_tags,
                "grounded_precision": precision,
                "grounded_recall": recall,
                "grounded_f1": f1,
                "quality": quality,
            }
            previous_move = move
            previous_blueprint = context.get("blueprint")

        return generated

    def _score_reference_against_system(
        self,
        corpus_label: str,
        game: Dict[str, Any],
        generated: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[MoveSample], List[MoveSample], List[PairSample]]:
        reference_samples: List[MoveSample] = []
        system_samples: List[MoveSample] = []
        pairs: List[PairSample] = []

        for move_row in game["moves"]:
            ply = move_row["ply"]
            reference_text = game["comments_by_ply"].get(ply, "")
            if not reference_text:
                continue
            system_row = generated.get(ply)
            if not system_row:
                continue

            prev_board = chess.Board(move_row["fen_before"])
            move = move_row["move"]
            curr_board = prev_board.copy(stack=False)
            curr_board.push(move)
            truth_tags = system_row["truth_tags"]
            reference_claims = sorted(extract_claim_tags(reference_text))
            ref_precision, ref_recall, ref_f1 = claim_score(reference_claims, truth_tags)

            reference_samples.append(
                MoveSample(
                    corpus_label=corpus_label,
                    source_label=corpus_label,
                    game_id=game["game_id"],
                    game_signature=game["signature"],
                    white=game["white"],
                    black=game["black"],
                    ply=ply,
                    san=move_row["san"],
                    opening_name=game["opening_by_ply"].get(ply, "Unknown"),
                    phase=phase_from_ply(ply),
                    commentary=reference_text,
                    quality=system_row["quality"],
                    truth_tags=truth_tags,
                    claim_tags=reference_claims,
                    grounded_precision=ref_precision,
                    grounded_recall=ref_recall,
                    grounded_f1=ref_f1,
                )
            )

            system_samples.append(
                MoveSample(
                    corpus_label=corpus_label,
                    source_label=self.system_source_label,
                    game_id=game["game_id"],
                    game_signature=game["signature"],
                    white=game["white"],
                    black=game["black"],
                    ply=ply,
                    san=move_row["san"],
                    opening_name=game["opening_by_ply"].get(ply, "Unknown"),
                    phase=phase_from_ply(ply),
                    commentary=system_row["commentary"],
                    quality=system_row["quality"],
                    truth_tags=truth_tags,
                    claim_tags=system_row["claim_tags"],
                    grounded_precision=system_row["grounded_precision"],
                    grounded_recall=system_row["grounded_recall"],
                    grounded_f1=system_row["grounded_f1"],
                )
            )

            pairs.append(
                self._pair_stub(
                    corpus_label=corpus_label,
                    game=game,
                    move_row=move_row,
                    quality=system_row["quality"],
                    reference_label=corpus_label,
                    candidate_label=self.system_source_label,
                    reference_text=reference_text,
                    candidate_text=system_row["commentary"],
                    reference_grounded_f1=ref_f1,
                    candidate_grounded_f1=system_row["grounded_f1"],
                )
            )

        return reference_samples, system_samples, pairs

    def _reference_only_samples(self, corpora: Sequence[Dict[str, Any]]) -> List[MoveSample]:
        seen = set()
        samples: List[MoveSample] = []
        for corpus in corpora:
            for game in corpus["games"]:
                for move_row in game["moves"]:
                    ply = move_row["ply"]
                    text = game["comments_by_ply"].get(ply, "")
                    if not text:
                        continue
                    key = (corpus["label"], game["game_id"], ply)
                    if key in seen:
                        continue
                    seen.add(key)
        return samples

    def _score_reference_overlap(self, corpora: Sequence[Dict[str, Any]]) -> List[PairSample]:
        if len(corpora) < 2:
            return []
        by_sig: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for corpus in corpora:
            for game in corpus["games"]:
                by_sig[game["signature"]][corpus["label"]] = game

        overlap_pairs: List[PairSample] = []
        labels = [corpus["label"] for corpus in corpora[:2]]
        left_label, right_label = labels[0], labels[1]

        for signature, mapping in by_sig.items():
            if left_label not in mapping or right_label not in mapping:
                continue
            left_game = mapping[left_label]
            right_game = mapping[right_label]
            common_plies = sorted(set(left_game["comments_by_ply"]).intersection(right_game["comments_by_ply"]))
            for ply in common_plies:
                left_row = next(item for item in left_game["moves"] if item["ply"] == ply)
                right_row = next(item for item in right_game["moves"] if item["ply"] == ply)
                overlap_pairs.append(
                    self._pair_stub(
                        corpus_label="reference_overlap",
                        game=left_game,
                        move_row=left_row,
                        quality="reference",
                        reference_label=left_label,
                        candidate_label=right_label,
                        reference_text=left_game["comments_by_ply"][ply],
                        candidate_text=right_game["comments_by_ply"][ply],
                        reference_grounded_f1=0.0,
                        candidate_grounded_f1=0.0,
                    )
                )
        return overlap_pairs

    def _pair_stub(
        self,
        corpus_label: str,
        game: Dict[str, Any],
        move_row: Dict[str, Any],
        quality: str,
        reference_label: str,
        candidate_label: str,
        reference_text: str,
        candidate_text: str,
        reference_grounded_f1: float,
        candidate_grounded_f1: float,
    ) -> PairSample:
        return PairSample(
            corpus_label=corpus_label,
            game_id=game["game_id"],
            game_signature=game["signature"],
            white=game["white"],
            black=game["black"],
            ply=move_row["ply"],
            san=move_row["san"],
            opening_name=game["opening_by_ply"].get(move_row["ply"], "Unknown"),
            phase=phase_from_ply(move_row["ply"]),
            quality=quality,
            reference_label=reference_label,
            candidate_label=candidate_label,
            reference_text=reference_text,
            candidate_text=candidate_text,
            bleu4=0.0,
            semantic_similarity=0.0,
            rouge_l_f1=0.0,
            token_f1=0.0,
            entity_f1=0.0,
            claim_tag_jaccard=0.0,
            reference_grounded_f1=reference_grounded_f1,
            candidate_grounded_f1=candidate_grounded_f1,
            overall_alignment=0.0,
        )

    def _finalize_pair_metrics(self, pairs: Sequence[PairSample]) -> None:
        if not pairs:
            return
        self.similarity.precompute(
            [text for pair in pairs for text in (pair.reference_text, pair.candidate_text)]
        )
        for pair in pairs:
            bleu4 = bleu4_score(pair.reference_text, pair.candidate_text)
            semantic = self.similarity.cosine(pair.reference_text, pair.candidate_text)
            rouge = rouge_l_f1(pair.reference_text, pair.candidate_text)
            token_f1 = token_f1_score(pair.reference_text, pair.candidate_text)
            entity_f1 = chess_entity_f1(pair.reference_text, pair.candidate_text)
            claim_jaccard = jaccard_score(
                extract_claim_tags(pair.reference_text),
                extract_claim_tags(pair.candidate_text),
            )
            overall = (
                0.30 * pair.candidate_grounded_f1
                + 0.25 * semantic
                + 0.15 * rouge
                + 0.10 * bleu4
                + 0.10 * entity_f1
                + 0.10 * claim_jaccard
            )
            pair.bleu4 = bleu4
            pair.semantic_similarity = semantic
            pair.rouge_l_f1 = rouge
            pair.token_f1 = token_f1
            pair.entity_f1 = entity_f1
            pair.claim_tag_jaccard = claim_jaccard
            pair.overall_alignment = overall

    def _score_candidate_against_reference(
        self,
        reference_game: Dict[str, Any],
        candidate_game: Dict[str, Any],
        generated: Dict[int, Dict[str, Any]],
        candidate_label: str,
    ) -> Tuple[List[MoveSample], List[PairSample]]:
        candidate_samples: List[MoveSample] = []
        pairs: List[PairSample] = []
        reference_moves_by_ply = {row["ply"]: row for row in reference_game["moves"]}
        common_plies = sorted(
            set(reference_game["comments_by_ply"]).intersection(candidate_game["comments_by_ply"])
        )

        for ply in common_plies:
            move_row = reference_moves_by_ply.get(ply)
            generated_row = generated.get(ply)
            if move_row is None or generated_row is None:
                continue

            truth_tags = generated_row["truth_tags"]
            candidate_text = candidate_game["comments_by_ply"][ply]
            candidate_claims = sorted(extract_claim_tags(candidate_text))
            cand_precision, cand_recall, cand_f1 = claim_score(candidate_claims, truth_tags)
            gm_text = reference_game["comments_by_ply"][ply]
            gm_claims = sorted(extract_claim_tags(gm_text))
            _, _, gm_f1 = claim_score(gm_claims, truth_tags)

            candidate_samples.append(
                MoveSample(
                    corpus_label="grandmaster_overlap",
                    source_label=candidate_label,
                    game_id=reference_game["game_id"],
                    game_signature=reference_game["signature"],
                    white=reference_game["white"],
                    black=reference_game["black"],
                    ply=ply,
                    san=move_row["san"],
                    opening_name=reference_game["opening_by_ply"].get(ply, "Unknown"),
                    phase=phase_from_ply(ply),
                    commentary=candidate_text,
                    quality=generated_row["quality"],
                    truth_tags=truth_tags,
                    claim_tags=candidate_claims,
                    grounded_precision=cand_precision,
                    grounded_recall=cand_recall,
                    grounded_f1=cand_f1,
                )
            )

            pairs.append(
                self._pair_stub(
                    corpus_label="grandmaster_overlap",
                    game=reference_game,
                    move_row=move_row,
                    quality=generated_row["quality"],
                    reference_label="grandmaster",
                    candidate_label=candidate_label,
                    reference_text=gm_text,
                    candidate_text=candidate_text,
                    reference_grounded_f1=gm_f1,
                    candidate_grounded_f1=cand_f1,
                )
            )

        return candidate_samples, pairs

    def _build_paper_summary(
        self,
        gm_corpus: Dict[str, Any],
        chessdotcom_corpus: Dict[str, Any],
        gm_reference_samples: Sequence[MoveSample],
        system_samples: Sequence[MoveSample],
        baseline_samples: Sequence[MoveSample],
        system_vs_gm_pairs: Sequence[PairSample],
        system_overlap_pairs: Sequence[PairSample],
        baseline_vs_gm_pairs: Sequence[PairSample],
    ) -> Dict[str, Any]:
        benchmark_rows = []
        fair_baseline_available = bool(baseline_vs_gm_pairs)

        if fair_baseline_available:
            benchmark_rows.append(
                {
                    "model": "our_system",
                    "subset": "shared GM positions",
                    **aggregate_pair_metrics(system_overlap_pairs),
                }
            )
            benchmark_rows.append(
                {
                    "model": "chessdotcom",
                    "subset": "shared GM positions",
                    **aggregate_pair_metrics(baseline_vs_gm_pairs),
                }
            )
        else:
            benchmark_rows.append(
                {
                    "model": "our_system",
                    "subset": "all GM positions",
                    **aggregate_pair_metrics(system_vs_gm_pairs),
                }
            )
            benchmark_rows.append(
                {
                    "model": "chessdotcom",
                    "subset": "shared GM positions",
                    "moves": 0,
                    "bleu4": None,
                    "rouge_l_f1": None,
                    "semantic_similarity": None,
                    "entity_f1": None,
                    "claim_tag_jaccard": None,
                    "grounded_f1": None,
                    "cgas": None,
                    "status": "No shared Grandmaster reference games in the current files.",
                }
            )

        source_profiles = []
        for source_label, rows in (
            ("grandmaster", gm_reference_samples),
            ("our_system", system_samples),
            ("chessdotcom", baseline_samples),
        ):
            if not rows:
                continue
            source_profiles.append(
                {
                    "source_label": source_label,
                    "moves": len(rows),
                    "avg_tokens": safe_mean(len(normalize_tokens(row.commentary)) for row in rows),
                    "avg_entities": safe_mean(len(extract_chess_entities(row.commentary)) for row in rows),
                    "avg_claim_tags": safe_mean(len(row.claim_tags) for row in rows),
                    "grounded_f1": safe_mean(row.grounded_f1 for row in rows),
                }
            )

        summary = {
            "embedding_backend": self.similarity.mode,
            "reference": {
                "label": gm_corpus["label"],
                "path": gm_corpus["path"],
                "games": len(gm_corpus["games"]),
                "annotated_moves": sum(len(game["comments_by_ply"]) for game in gm_corpus["games"]),
            },
            "baseline": {
                "label": chessdotcom_corpus["label"],
                "path": chessdotcom_corpus["path"],
                "games": len(chessdotcom_corpus["games"]),
                "annotated_moves": sum(len(game["comments_by_ply"]) for game in chessdotcom_corpus["games"]),
                "shared_games_with_reference": len({pair.game_signature for pair in baseline_vs_gm_pairs}),
                "shared_moves_with_reference": len(baseline_vs_gm_pairs),
            },
            "primary_metric": {
                "name": "CGAS",
                "definition": "0.30*grounded_f1 + 0.25*semantic + 0.15*rouge_l + 0.10*bleu4 + 0.10*entity_f1 + 0.10*claim_jaccard",
                "purpose": "Chess Grounded Alignment Score emphasizes factual chess understanding over surface overlap.",
            },
            "fair_baseline_available": fair_baseline_available,
            "fairness_note": (
                "Fair head-to-head comparison is only valid on shared Grandmaster reference games."
                if fair_baseline_available
                else "Current files do not contain shared Grandmaster reference games for Chess.com, so no fair beat-the-baseline claim can be made yet."
            ),
            "benchmark": benchmark_rows,
            "system_full_vs_grandmaster": aggregate_pair_metrics(system_vs_gm_pairs),
            "system_by_phase": aggregate_pairs_by_key(system_vs_gm_pairs, key_name="phase"),
            "system_by_quality": aggregate_pairs_by_key(system_vs_gm_pairs, key_name="quality"),
            "source_profiles": source_profiles,
        }
        return summary

    def _write_paper_outputs(
        self,
        output_dir: Path,
        summary: Dict[str, Any],
        system_vs_gm_pairs: Sequence[PairSample],
        system_overlap_pairs: Sequence[PairSample],
        baseline_vs_gm_pairs: Sequence[PairSample],
    ) -> None:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        (output_dir / "README.md").write_text(self._render_readme(summary), encoding="utf-8")
        (output_dir / "report.md").write_text(self._render_paper_report(summary), encoding="utf-8")

        self._write_csv(output_dir / "benchmark_summary.csv", summary["benchmark"])
        (output_dir / "paper_tables.md").write_text(self._render_paper_tables_markdown(summary), encoding="utf-8")
        (output_dir / "paper_tables.tex").write_text(self._render_paper_tables_latex(summary), encoding="utf-8")

        self._write_paper_figures(
            figures_dir=figures_dir,
            summary=summary,
            system_vs_gm_pairs=system_vs_gm_pairs,
            system_overlap_pairs=system_overlap_pairs,
            baseline_vs_gm_pairs=baseline_vs_gm_pairs,
        )

        if self.include_appendix:
            appendix_rows = [
                pair.__dict__
                for pair in list(system_vs_gm_pairs) + list(baseline_vs_gm_pairs)
            ]
            self._write_csv(output_dir / "appendix_move_scores.csv", appendix_rows)

    def _render_readme(self, summary: Dict[str, Any]) -> str:
        lines = [
            "# Commentary Benchmark",
            "",
            "Start with `report.md`.",
            "",
            "Files:",
            "- `report.md`: human-readable benchmark summary for the paper",
            "- `benchmark_summary.csv`: main result table",
            "- `paper_tables.md`: copy-friendly Markdown tables",
            "- `paper_tables.tex`: LaTeX tables for the paper",
            "- `figures/`: publication-ready SVG charts",
        ]
        if self.include_appendix:
            lines.append("- `appendix_move_scores.csv`: move-level appendix data")
        lines.extend(
            [
                "",
                f"Primary metric: `{summary['primary_metric']['name']}`",
                f"Fairness rule: {summary['fairness_note']}",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    def _render_paper_report(self, summary: Dict[str, Any]) -> str:
        benchmark_rows = summary["benchmark"]
        system_row = benchmark_rows[0] if benchmark_rows else {}
        baseline_row = benchmark_rows[1] if len(benchmark_rows) > 1 else {}
        lines = [
            "# Commentary Evaluation Report",
            "",
            "## Benchmark Goal",
            "",
            "Grandmaster commentary is the only reference. Our system is the model under test. Chess.com is treated as an external baseline only when it comments on the same Grandmaster reference games.",
            "",
            "## Setup",
            "",
            f"- Reference corpus: `{summary['reference']['games']}` games / `{summary['reference']['annotated_moves']}` moves from `{summary['reference']['path']}`",
            f"- Baseline corpus: `{summary['baseline']['games']}` games / `{summary['baseline']['annotated_moves']}` moves from `{summary['baseline']['path']}`",
            f"- Semantic backend: `{summary['embedding_backend']}`",
            f"- Primary metric: `{summary['primary_metric']['name']}` = `{summary['primary_metric']['definition']}`",
            "- Secondary metrics: BLEU-4, ROUGE-L, semantic similarity, chess-entity F1, claim-tag Jaccard, grounded F1",
            "",
            "## Fairness Note",
            "",
            f"- {summary['fairness_note']}",
            "",
            "## Main Result",
            "",
        ]

        if summary["fair_baseline_available"] and baseline_row.get("cgas") is not None:
            delta = system_row["cgas"] - baseline_row["cgas"]
            verdict = "beats" if delta > 0 else "does not beat"
            lines.append(
                f"- On the shared Grandmaster subset, our system `{verdict}` Chess.com by `{fmt(abs(delta))}` CGAS."
            )
            lines.append(
                f"- Our system CGAS: `{fmt(system_row['cgas'])}` vs Chess.com CGAS: `{fmt(baseline_row['cgas'])}`."
            )
        else:
            lines.append(
                "- A fair beat-the-baseline claim is not available from the current files because Chess.com does not overlap with the Grandmaster reference games."
            )
            lines.append(
                f"- Our system still scores `{fmt(summary['system_full_vs_grandmaster']['cgas'])}` CGAS against the full Grandmaster corpus."
            )

        lines.extend(
            [
                "",
                "## Read This First",
                "",
                "- `benchmark_summary.csv` is the main table to cite.",
                "- `paper_tables.tex` is the paper-ready table export.",
                "- `figures/main_benchmark.svg` is the headline chart.",
                "- `figures/system_quality_breakdown.svg` and `figures/system_phase_breakdown.svg` support analysis sections.",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    def _render_paper_tables_markdown(self, summary: Dict[str, Any]) -> str:
        benchmark_rows = summary["benchmark"]
        phase_rows = summary["system_by_phase"]
        quality_rows = summary["system_by_quality"]
        profile_rows = summary["source_profiles"]
        sections = [
            "# Paper Tables",
            "",
            "## Main Benchmark",
            "",
            render_markdown_table(
                headers=["Model", "Subset", "Moves", "BLEU-4", "ROUGE-L", "Semantic", "Entity F1", "Claim Jaccard", "Grounded F1", "CGAS"],
                rows=[
                    [
                        row["model"],
                        row["subset"],
                        row.get("moves", 0),
                        fmt_or_na(row.get("bleu4")),
                        fmt_or_na(row.get("rouge_l_f1")),
                        fmt_or_na(row.get("semantic_similarity")),
                        fmt_or_na(row.get("entity_f1")),
                        fmt_or_na(row.get("claim_tag_jaccard")),
                        fmt_or_na(row.get("grounded_f1")),
                        fmt_or_na(row.get("cgas")),
                    ]
                    for row in benchmark_rows
                ],
            ),
            "",
            "## System Breakdown By Phase",
            "",
            render_markdown_table(
                headers=["Phase", "Moves", "BLEU-4", "ROUGE-L", "Semantic", "Grounded F1", "CGAS"],
                rows=[
                    [
                        row["phase"],
                        row["moves"],
                        fmt(row["bleu4"]),
                        fmt(row["rouge_l_f1"]),
                        fmt(row["semantic_similarity"]),
                        fmt(row["grounded_f1"]),
                        fmt(row["cgas"]),
                    ]
                    for row in phase_rows
                ],
            ),
            "",
            "## System Breakdown By Move Quality",
            "",
            render_markdown_table(
                headers=["Quality", "Moves", "BLEU-4", "ROUGE-L", "Semantic", "Grounded F1", "CGAS"],
                rows=[
                    [
                        row["quality"],
                        row["moves"],
                        fmt(row["bleu4"]),
                        fmt(row["rouge_l_f1"]),
                        fmt(row["semantic_similarity"]),
                        fmt(row["grounded_f1"]),
                        fmt(row["cgas"]),
                    ]
                    for row in quality_rows
                ],
            ),
        ]
        if profile_rows:
            sections.extend(
                [
                    "",
                    "## Source Profiles",
                    "",
                    render_markdown_table(
                        headers=["Source", "Moves", "Avg Tokens", "Avg Entities", "Avg Claim Tags", "Grounded F1"],
                        rows=[
                            [
                                row["source_label"],
                                row["moves"],
                                fmt(row["avg_tokens"]),
                                fmt(row["avg_entities"]),
                                fmt(row["avg_claim_tags"]),
                                fmt(row["grounded_f1"]),
                            ]
                            for row in profile_rows
                        ],
                    ),
                ]
            )
        return "\n".join(sections) + "\n"

    def _render_paper_tables_latex(self, summary: Dict[str, Any]) -> str:
        benchmark_rows = summary["benchmark"]
        phase_rows = summary["system_by_phase"]
        quality_rows = summary["system_by_quality"]
        sections = [
            render_latex_table(
                headers=["Model", "Subset", "Moves", "BLEU-4", "ROUGE-L", "Semantic", "Entity F1", "Claim Jac.", "Grounded F1", "CGAS"],
                rows=[
                    [
                        row["model"],
                        row["subset"],
                        str(row.get("moves", 0)),
                        fmt_or_na(row.get("bleu4")),
                        fmt_or_na(row.get("rouge_l_f1")),
                        fmt_or_na(row.get("semantic_similarity")),
                        fmt_or_na(row.get("entity_f1")),
                        fmt_or_na(row.get("claim_tag_jaccard")),
                        fmt_or_na(row.get("grounded_f1")),
                        fmt_or_na(row.get("cgas")),
                    ]
                    for row in benchmark_rows
                ],
                caption="Main benchmark against Grandmaster commentary.",
                label="tab:main-benchmark",
            ),
            render_latex_table(
                headers=["Phase", "Moves", "BLEU-4", "ROUGE-L", "Semantic", "Grounded F1", "CGAS"],
                rows=[
                    [
                        row["phase"],
                        str(row["moves"]),
                        fmt(row["bleu4"]),
                        fmt(row["rouge_l_f1"]),
                        fmt(row["semantic_similarity"]),
                        fmt(row["grounded_f1"]),
                        fmt(row["cgas"]),
                    ]
                    for row in phase_rows
                ],
                caption="System performance against Grandmaster commentary by game phase.",
                label="tab:system-phase-breakdown",
            ),
            render_latex_table(
                headers=["Quality", "Moves", "BLEU-4", "ROUGE-L", "Semantic", "Grounded F1", "CGAS"],
                rows=[
                    [
                        row["quality"],
                        str(row["moves"]),
                        fmt(row["bleu4"]),
                        fmt(row["rouge_l_f1"]),
                        fmt(row["semantic_similarity"]),
                        fmt(row["grounded_f1"]),
                        fmt(row["cgas"]),
                    ]
                    for row in quality_rows
                ],
                caption="System performance against Grandmaster commentary by move quality.",
                label="tab:system-quality-breakdown",
            ),
        ]
        return "\n\n".join(sections) + "\n"

    def _write_paper_figures(
        self,
        figures_dir: Path,
        summary: Dict[str, Any],
        system_vs_gm_pairs: Sequence[PairSample],
        system_overlap_pairs: Sequence[PairSample],
        baseline_vs_gm_pairs: Sequence[PairSample],
    ) -> None:
        benchmark_rows = [row for row in summary["benchmark"] if row.get("cgas") is not None]
        phase_rows = summary["system_by_phase"]
        quality_rows = summary["system_by_quality"]

        grouped_bar_chart_svg(
            title="Main Benchmark Against Grandmaster Commentary",
            categories=[row["model"] for row in benchmark_rows],
            series={
                "CGAS": [zero_if_none(row.get("cgas")) for row in benchmark_rows],
                "Grounded F1": [zero_if_none(row.get("grounded_f1")) for row in benchmark_rows],
                "Semantic": [zero_if_none(row.get("semantic_similarity")) for row in benchmark_rows],
            },
            path=figures_dir / "main_benchmark.svg",
            y_label="score",
        )

        grouped_bar_chart_svg(
            title="System Performance By Game Phase",
            categories=[row["phase"] for row in phase_rows],
            series={
                "CGAS": [row["cgas"] for row in phase_rows],
                "Grounded F1": [row["grounded_f1"] for row in phase_rows],
                "ROUGE-L": [row["rouge_l_f1"] for row in phase_rows],
            },
            path=figures_dir / "system_phase_breakdown.svg",
            y_label="score",
        )

        grouped_bar_chart_svg(
            title="System Performance By Move Quality",
            categories=[row["quality"] for row in quality_rows],
            series={
                "CGAS": [row["cgas"] for row in quality_rows],
                "Grounded F1": [row["grounded_f1"] for row in quality_rows],
                "Semantic": [row["semantic_similarity"] for row in quality_rows],
            },
            path=figures_dir / "system_quality_breakdown.svg",
            y_label="score",
        )

        heatmap_svg(
            title="Main Benchmark Metric Heatmap",
            rows=[row["model"] for row in benchmark_rows],
            cols=["BLEU-4", "ROUGE-L", "Semantic", "Grounded F1", "CGAS"],
            values=[
                [
                    zero_if_none(row.get("bleu4")),
                    zero_if_none(row.get("rouge_l_f1")),
                    zero_if_none(row.get("semantic_similarity")),
                    zero_if_none(row.get("grounded_f1")),
                    zero_if_none(row.get("cgas")),
                ]
                for row in benchmark_rows
            ],
            path=figures_dir / "metric_heatmap.svg",
        )

    def _build_summary(
        self,
        corpora: Sequence[Dict[str, Any]],
        source_move_samples: Sequence[MoveSample],
        pair_samples: Sequence[PairSample],
        overlap_samples: Sequence[PairSample],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "embedding_backend": self.similarity.mode,
            "corpora": [],
            "groundedness_by_source": [],
            "alignment_by_corpus": [],
            "alignment_by_phase": [],
            "alignment_by_quality": [],
            "source_profiles": [],
            "reference_overlap": {},
        }

        for corpus in corpora:
            annotated_moves = sum(len(game["comments_by_ply"]) for game in corpus["games"])
            summary["corpora"].append(
                {
                    "label": corpus["label"],
                    "path": corpus["path"],
                    "games": len(corpus["games"]),
                    "annotated_moves": annotated_moves,
                }
            )

        grouped_sources: Dict[Tuple[str, str], List[MoveSample]] = defaultdict(list)
        for row in source_move_samples:
            grouped_sources[(row.corpus_label, row.source_label)].append(row)

        for (corpus_label, source_label), rows in sorted(grouped_sources.items()):
            summary["groundedness_by_source"].append(
                {
                    "corpus_label": corpus_label,
                    "source_label": source_label,
                    "moves": len(rows),
                    "grounded_precision": safe_mean(item.grounded_precision for item in rows),
                    "grounded_recall": safe_mean(item.grounded_recall for item in rows),
                    "grounded_f1": safe_mean(item.grounded_f1 for item in rows),
                }
            )

        grouped_pairs: Dict[str, List[PairSample]] = defaultdict(list)
        for row in pair_samples:
            grouped_pairs[row.corpus_label].append(row)

        for corpus_label, rows in sorted(grouped_pairs.items()):
            summary["alignment_by_corpus"].append(
                {
                    "corpus_label": corpus_label,
                    "moves": len(rows),
                    "semantic_similarity": safe_mean(item.semantic_similarity for item in rows),
                    "rouge_l_f1": safe_mean(item.rouge_l_f1 for item in rows),
                    "token_f1": safe_mean(item.token_f1 for item in rows),
                    "entity_f1": safe_mean(item.entity_f1 for item in rows),
                    "claim_tag_jaccard": safe_mean(item.claim_tag_jaccard for item in rows),
                    "system_grounded_f1": safe_mean(item.candidate_grounded_f1 for item in rows),
                    "reference_grounded_f1": safe_mean(item.reference_grounded_f1 for item in rows),
                    "overall_alignment": safe_mean(item.overall_alignment for item in rows),
                }
            )

        grouped_phase: Dict[Tuple[str, str], List[PairSample]] = defaultdict(list)
        for row in pair_samples:
            grouped_phase[(row.corpus_label, row.phase)].append(row)
        for (corpus_label, phase), rows in sorted(grouped_phase.items()):
            summary["alignment_by_phase"].append(
                {
                    "corpus_label": corpus_label,
                    "phase": phase,
                    "moves": len(rows),
                    "semantic_similarity": safe_mean(item.semantic_similarity for item in rows),
                    "entity_f1": safe_mean(item.entity_f1 for item in rows),
                    "overall_alignment": safe_mean(item.overall_alignment for item in rows),
                    "system_grounded_f1": safe_mean(item.candidate_grounded_f1 for item in rows),
                    "reference_grounded_f1": safe_mean(item.reference_grounded_f1 for item in rows),
                }
            )

        grouped_quality: Dict[Tuple[str, str], List[PairSample]] = defaultdict(list)
        for row in pair_samples:
            grouped_quality[(row.corpus_label, row.quality)].append(row)
        for (corpus_label, quality), rows in sorted(grouped_quality.items()):
            summary["alignment_by_quality"].append(
                {
                    "corpus_label": corpus_label,
                    "quality": quality,
                    "moves": len(rows),
                    "semantic_similarity": safe_mean(item.semantic_similarity for item in rows),
                    "entity_f1": safe_mean(item.entity_f1 for item in rows),
                    "overall_alignment": safe_mean(item.overall_alignment for item in rows),
                    "system_grounded_f1": safe_mean(item.candidate_grounded_f1 for item in rows),
                    "reference_grounded_f1": safe_mean(item.reference_grounded_f1 for item in rows),
                }
            )

        grouped_profiles: Dict[str, List[MoveSample]] = defaultdict(list)
        for row in source_move_samples:
            grouped_profiles[row.source_label].append(row)
        for source_label, rows in sorted(grouped_profiles.items()):
            bad_move_rows = [row for row in rows if "bad_move" in row.truth_tags]
            summary["source_profiles"].append(
                {
                    "source_label": source_label,
                    "moves": len(rows),
                    "avg_tokens": safe_mean(len(normalize_tokens(row.commentary)) for row in rows),
                    "avg_entities": safe_mean(len(extract_chess_entities(row.commentary)) for row in rows),
                    "avg_claim_tags": safe_mean(len(row.claim_tags) for row in rows),
                    "grounded_f1": safe_mean(row.grounded_f1 for row in rows),
                    "bad_move_mention_rate": safe_mean(
                        1.0 if "bad_move" in row.claim_tags else 0.0 for row in bad_move_rows
                    ) if bad_move_rows else 0.0,
                }
            )

        summary["reference_overlap"] = {
            "moves": len(overlap_samples),
            "semantic_similarity": safe_mean(item.semantic_similarity for item in overlap_samples),
            "rouge_l_f1": safe_mean(item.rouge_l_f1 for item in overlap_samples),
            "entity_f1": safe_mean(item.entity_f1 for item in overlap_samples),
            "overall_alignment": safe_mean(item.overall_alignment for item in overlap_samples),
        }
        return summary

    def _write_outputs(
        self,
        output_dir: Path,
        corpora: Sequence[Dict[str, Any]],
        source_move_samples: Sequence[MoveSample],
        pair_samples: Sequence[PairSample],
        overlap_samples: Sequence[PairSample],
        summary: Dict[str, Any],
    ) -> None:
        tables_dir = output_dir / "tables"
        charts_dir = output_dir / "charts"
        tables_dir.mkdir(exist_ok=True)
        charts_dir.mkdir(exist_ok=True)

        self._write_json(output_dir / "summary.json", summary)
        self._write_csv(output_dir / "source_groundedness_per_move.csv", source_move_samples)
        self._write_csv(output_dir / "pairwise_alignment_per_move.csv", pair_samples)
        self._write_csv(output_dir / "reference_overlap_per_move.csv", overlap_samples)
        self._write_summary_tables(tables_dir, summary, source_move_samples, pair_samples)
        self._write_report(output_dir / "report.md", corpora, summary)
        self._write_charts(charts_dir, summary, pair_samples, source_move_samples)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_csv(self, path: Path, rows: Sequence[Any]) -> None:
        if not rows:
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write("")
            return
        if hasattr(rows[0], "__dataclass_fields__"):
            payload = [row.__dict__ for row in rows]
        else:
            payload = list(rows)
        fieldnames = sorted({key for row in payload for key in row.keys()})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(payload)

    def _write_summary_tables(
        self,
        tables_dir: Path,
        summary: Dict[str, Any],
        source_move_samples: Sequence[MoveSample],
        pair_samples: Sequence[PairSample],
    ) -> None:
        alignment_rows = summary["alignment_by_corpus"]
        grounded_rows = summary["groundedness_by_source"]
        phase_rows = summary["alignment_by_phase"]
        quality_rows = summary["alignment_by_quality"]
        profile_rows = summary["source_profiles"]

        alignment_md = render_markdown_table(
            headers=[
                "Corpus",
                "Moves",
                "Semantic",
                "ROUGE-L",
                "Entity F1",
                "Claim Jaccard",
                "System Grounded F1",
                "Reference Grounded F1",
                "Overall",
            ],
            rows=[
                [
                    row["corpus_label"],
                    row["moves"],
                    fmt(row["semantic_similarity"]),
                    fmt(row["rouge_l_f1"]),
                    fmt(row["entity_f1"]),
                    fmt(row["claim_tag_jaccard"]),
                    fmt(row["system_grounded_f1"]),
                    fmt(row["reference_grounded_f1"]),
                    fmt(row["overall_alignment"]),
                ]
                for row in alignment_rows
            ],
        )
        (tables_dir / "alignment_summary.md").write_text(alignment_md, encoding="utf-8")

        grounded_md = render_markdown_table(
            headers=["Corpus", "Source", "Moves", "Precision", "Recall", "Grounded F1"],
            rows=[
                [
                    row["corpus_label"],
                    row["source_label"],
                    row["moves"],
                    fmt(row["grounded_precision"]),
                    fmt(row["grounded_recall"]),
                    fmt(row["grounded_f1"]),
                ]
                for row in grounded_rows
            ],
        )
        (tables_dir / "groundedness_summary.md").write_text(grounded_md, encoding="utf-8")

        (tables_dir / "alignment_summary.tex").write_text(
            render_latex_table(
                headers=["Corpus", "Moves", "Semantic", "ROUGE-L", "Entity F1", "Claim Jac.", "System G-F1", "Ref G-F1", "Overall"],
                rows=[
                    [
                        row["corpus_label"],
                        str(row["moves"]),
                        fmt(row["semantic_similarity"]),
                        fmt(row["rouge_l_f1"]),
                        fmt(row["entity_f1"]),
                        fmt(row["claim_tag_jaccard"]),
                        fmt(row["system_grounded_f1"]),
                        fmt(row["reference_grounded_f1"]),
                        fmt(row["overall_alignment"]),
                    ]
                    for row in alignment_rows
                ],
                caption="System-to-reference alignment across commentary corpora.",
                label="tab:commentary-alignment",
            ),
            encoding="utf-8",
        )

        (tables_dir / "groundedness_summary.tex").write_text(
            render_latex_table(
                headers=["Corpus", "Source", "Moves", "Precision", "Recall", "Grounded F1"],
                rows=[
                    [
                        row["corpus_label"],
                        row["source_label"],
                        str(row["moves"]),
                        fmt(row["grounded_precision"]),
                        fmt(row["grounded_recall"]),
                        fmt(row["grounded_f1"]),
                    ]
                    for row in grounded_rows
                ],
                caption="Board-grounded factuality of each commentary source.",
                label="tab:commentary-groundedness",
            ),
            encoding="utf-8",
        )

        phase_md = render_markdown_table(
            headers=["Corpus", "Phase", "Moves", "Semantic", "Entity F1", "System G-F1", "Reference G-F1", "Overall"],
            rows=[
                [
                    row["corpus_label"],
                    row["phase"],
                    row["moves"],
                    fmt(row["semantic_similarity"]),
                    fmt(row["entity_f1"]),
                    fmt(row["system_grounded_f1"]),
                    fmt(row["reference_grounded_f1"]),
                    fmt(row["overall_alignment"]),
                ]
                for row in phase_rows
            ],
        )
        (tables_dir / "alignment_by_phase.md").write_text(phase_md, encoding="utf-8")

        quality_md = render_markdown_table(
            headers=["Corpus", "Quality", "Moves", "Semantic", "Entity F1", "System G-F1", "Reference G-F1", "Overall"],
            rows=[
                [
                    row["corpus_label"],
                    row["quality"],
                    row["moves"],
                    fmt(row["semantic_similarity"]),
                    fmt(row["entity_f1"]),
                    fmt(row["system_grounded_f1"]),
                    fmt(row["reference_grounded_f1"]),
                    fmt(row["overall_alignment"]),
                ]
                for row in quality_rows
            ],
        )
        (tables_dir / "alignment_by_quality.md").write_text(quality_md, encoding="utf-8")

        profile_md = render_markdown_table(
            headers=["Source", "Moves", "Avg Tokens", "Avg Entities", "Avg Claim Tags", "Grounded F1", "Bad Move Mention Rate"],
            rows=[
                [
                    row["source_label"],
                    row["moves"],
                    fmt(row["avg_tokens"]),
                    fmt(row["avg_entities"]),
                    fmt(row["avg_claim_tags"]),
                    fmt(row["grounded_f1"]),
                    fmt(row["bad_move_mention_rate"]),
                ]
                for row in profile_rows
            ],
        )
        (tables_dir / "source_profiles.md").write_text(profile_md, encoding="utf-8")

        (tables_dir / "alignment_by_phase.tex").write_text(
            render_latex_table(
                headers=["Corpus", "Phase", "Moves", "Semantic", "Entity F1", "System G-F1", "Ref G-F1", "Overall"],
                rows=[
                    [
                        row["corpus_label"],
                        row["phase"],
                        str(row["moves"]),
                        fmt(row["semantic_similarity"]),
                        fmt(row["entity_f1"]),
                        fmt(row["system_grounded_f1"]),
                        fmt(row["reference_grounded_f1"]),
                        fmt(row["overall_alignment"]),
                    ]
                    for row in phase_rows
                ],
                caption="System-to-reference alignment broken down by game phase.",
                label="tab:alignment-by-phase",
            ),
            encoding="utf-8",
        )

        (tables_dir / "alignment_by_quality.tex").write_text(
            render_latex_table(
                headers=["Corpus", "Quality", "Moves", "Semantic", "Entity F1", "System G-F1", "Ref G-F1", "Overall"],
                rows=[
                    [
                        row["corpus_label"],
                        row["quality"],
                        str(row["moves"]),
                        fmt(row["semantic_similarity"]),
                        fmt(row["entity_f1"]),
                        fmt(row["system_grounded_f1"]),
                        fmt(row["reference_grounded_f1"]),
                        fmt(row["overall_alignment"]),
                    ]
                    for row in quality_rows
                ],
                caption="System-to-reference alignment broken down by move quality.",
                label="tab:alignment-by-quality",
            ),
            encoding="utf-8",
        )

        (tables_dir / "source_profiles.tex").write_text(
            render_latex_table(
                headers=["Source", "Moves", "Avg Tokens", "Avg Entities", "Avg Claims", "Grounded F1", "Bad Move Rate"],
                rows=[
                    [
                        row["source_label"],
                        str(row["moves"]),
                        fmt(row["avg_tokens"]),
                        fmt(row["avg_entities"]),
                        fmt(row["avg_claim_tags"]),
                        fmt(row["grounded_f1"]),
                        fmt(row["bad_move_mention_rate"]),
                    ]
                    for row in profile_rows
                ],
                caption="High-level commentary profile comparison across sources.",
                label="tab:source-profiles",
            ),
            encoding="utf-8",
        )

        per_game_rows = aggregate_per_game(pair_samples)
        per_game_md = render_markdown_table(
            headers=["Corpus", "Game", "Moves", "Semantic", "Entity F1", "System G-F1", "Reference G-F1", "Overall"],
            rows=[
                [
                    row["corpus_label"],
                    row["game_label"],
                    row["moves"],
                    fmt(row["semantic_similarity"]),
                    fmt(row["entity_f1"]),
                    fmt(row["system_grounded_f1"]),
                    fmt(row["reference_grounded_f1"]),
                    fmt(row["overall_alignment"]),
                ]
                for row in per_game_rows
            ],
        )
        (tables_dir / "per_game_alignment.md").write_text(per_game_md, encoding="utf-8")

    def _write_report(self, path: Path, corpora: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        lines = [
            "# Commentary Evaluation Report",
            "",
            "## Setup",
            "",
            f"- Embedding backend: `{summary['embedding_backend']}`",
            "- Pairwise alignment metrics: semantic cosine, ROUGE-L F1, token F1, chess entity F1, claim-tag Jaccard",
            "- Board-grounded metric: claim precision / recall / F1 against move truth derived from the board, engine context, and your commentary pipeline",
            "- Candidate commentary source: your local commentary system via `build_rich_context(...)` + `generate_fallback_commentary(...)`",
            "- Composite score: `0.45 * semantic + 0.20 * ROUGE-L + 0.20 * entity F1 + 0.15 * claim-tag Jaccard`",
            "",
            "## Corpora",
            "",
        ]
        for corpus in summary["corpora"]:
            lines.append(
                f"- `{corpus['label']}`: {corpus['games']} games, {corpus['annotated_moves']} annotated moves from `{corpus['path']}`"
            )

        lines.extend(
            [
                "",
                "## Key Findings",
                "",
            ]
        )

        best_alignment = max(summary["alignment_by_corpus"], key=lambda row: row["overall_alignment"], default=None)
        weakest_alignment = min(summary["alignment_by_corpus"], key=lambda row: row["overall_alignment"], default=None)
        if best_alignment:
            lines.append(
                f"- Strongest system-reference alignment: `{best_alignment['corpus_label']}` with overall score `{fmt(best_alignment['overall_alignment'])}`."
            )
        if weakest_alignment:
            lines.append(
                f"- Weakest system-reference alignment: `{weakest_alignment['corpus_label']}` with overall score `{fmt(weakest_alignment['overall_alignment'])}`."
            )

        grounded = summary["groundedness_by_source"]
        by_source = defaultdict(list)
        for row in grounded:
            by_source[row["source_label"]].append(row["grounded_f1"])
        for source_label, values in sorted(by_source.items()):
            lines.append(f"- Mean grounded F1 for `{source_label}`: `{fmt(safe_mean(values))}`.")

        best_phase = max(summary["alignment_by_phase"], key=lambda row: row["overall_alignment"], default=None)
        if best_phase:
            lines.append(
                f"- Best phase-level agreement: `{best_phase['corpus_label']}` / `{best_phase['phase']}` at `{fmt(best_phase['overall_alignment'])}`."
            )

        best_quality = max(summary["alignment_by_quality"], key=lambda row: row["overall_alignment"], default=None)
        if best_quality:
            lines.append(
                f"- Best quality bucket agreement: `{best_quality['corpus_label']}` / `{best_quality['quality']}` at `{fmt(best_quality['overall_alignment'])}`."
            )

        if summary["reference_overlap"]["moves"] == 0:
            lines.append(
                "- The two reference corpora do not share any game signatures in the current files, so direct Grandmaster-vs-Chess.com move-by-move comparison is unavailable."
            )
        else:
            lines.append(
                f"- Reference overlap: `{summary['reference_overlap']['moves']}` common annotated moves with overall overlap score `{fmt(summary['reference_overlap']['overall_alignment'])}`."
            )

        lines.extend(
            [
                "",
                "## Outputs",
                "",
                "- `summary.json`: machine-readable aggregate results",
                "- `source_groundedness_per_move.csv`: per-move board-grounded factuality for every source",
                "- `pairwise_alignment_per_move.csv`: per-move system-vs-reference comparison",
                "- `tables/*.md` and `tables/*.tex`: publication-ready tables",
                "- `charts/*.svg`: publication-ready vector charts",
                "",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_charts(
        self,
        charts_dir: Path,
        summary: Dict[str, Any],
        pair_samples: Sequence[PairSample],
        source_move_samples: Sequence[MoveSample],
    ) -> None:
        alignment_rows = summary["alignment_by_corpus"]
        grounded_rows = summary["groundedness_by_source"]
        per_game_rows = aggregate_per_game(pair_samples)
        phase_rows = summary["alignment_by_phase"]
        quality_rows = summary["alignment_by_quality"]
        profile_rows = summary["source_profiles"]

        grouped_bar_chart_svg(
            title="System vs Reference Alignment by Corpus",
            categories=[row["corpus_label"] for row in alignment_rows],
            series={
                "semantic": [row["semantic_similarity"] for row in alignment_rows],
                "entity_f1": [row["entity_f1"] for row in alignment_rows],
                "overall": [row["overall_alignment"] for row in alignment_rows],
            },
            path=charts_dir / "alignment_by_corpus.svg",
            y_label="score",
        )

        grouped_bar_chart_svg(
            title="Board-Grounded Factuality by Source",
            categories=[f"{row['corpus_label']}:{row['source_label']}" for row in grounded_rows],
            series={
                "precision": [row["grounded_precision"] for row in grounded_rows],
                "recall": [row["grounded_recall"] for row in grounded_rows],
                "grounded_f1": [row["grounded_f1"] for row in grounded_rows],
            },
            path=charts_dir / "groundedness_by_source.svg",
            y_label="score",
        )

        heatmap_svg(
            title="Corpus Metric Heatmap",
            rows=[row["corpus_label"] for row in alignment_rows],
            cols=["semantic", "rouge_l", "entity_f1", "overall"],
            values=[
                [
                    row["semantic_similarity"],
                    row["rouge_l_f1"],
                    row["entity_f1"],
                    row["overall_alignment"],
                ]
                for row in alignment_rows
            ],
            path=charts_dir / "corpus_metric_heatmap.svg",
        )

        grouped_bar_chart_svg(
            title="Per-Game System vs Reference Overall Alignment",
            categories=[row["game_label"] for row in per_game_rows],
            series={"overall": [row["overall_alignment"] for row in per_game_rows]},
            path=charts_dir / "per_game_overall_alignment.svg",
            y_label="score",
        )

        grouped_bar_chart_svg(
            title="Alignment by Game Phase",
            categories=[f"{row['corpus_label']}:{row['phase']}" for row in phase_rows],
            series={
                "overall": [row["overall_alignment"] for row in phase_rows],
                "semantic": [row["semantic_similarity"] for row in phase_rows],
            },
            path=charts_dir / "alignment_by_phase.svg",
            y_label="score",
        )

        grouped_bar_chart_svg(
            title="Alignment by Move Quality",
            categories=[f"{row['corpus_label']}:{row['quality']}" for row in quality_rows],
            series={
                "overall": [row["overall_alignment"] for row in quality_rows],
                "system_grounded_f1": [row["system_grounded_f1"] for row in quality_rows],
                "reference_grounded_f1": [row["reference_grounded_f1"] for row in quality_rows],
            },
            path=charts_dir / "alignment_by_quality.svg",
            y_label="score",
        )

        grouped_bar_chart_svg(
            title="Commentary Source Profiles",
            categories=[row["source_label"] for row in profile_rows],
            series={
                "grounded_f1": [row["grounded_f1"] for row in profile_rows],
                "bad_move_rate": [row["bad_move_mention_rate"] for row in profile_rows],
                "avg_claim_tags": [min(1.0, row["avg_claim_tags"] / 6.0) for row in profile_rows],
            },
            path=charts_dir / "source_profiles.svg",
            y_label="normalized score",
        )


def normalize_opening_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned or cleaned.lower() in {"unknown", "starting position"}:
        return ""
    return cleaned.split(":", 1)[0].strip()


def normalize_commentary(text: str) -> str:
    text = COMMENT_TAG_RE.sub(" ", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize_commentary(text).lower())


def token_f1_score(left: str, right: str) -> float:
    left_tokens = Counter(normalize_tokens(left))
    right_tokens = Counter(normalize_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((left_tokens & right_tokens).values())
    precision = overlap / sum(right_tokens.values())
    recall = overlap / sum(left_tokens.values())
    return f1_score(precision, recall)


def bleu4_score(left: str, right: str) -> float:
    left_tokens = normalize_tokens(left)
    right_tokens = normalize_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens or sentence_bleu is None:
        return 0.0
    return float(
        sentence_bleu(
            [left_tokens],
            right_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=BLEU_SMOOTHER,
        )
    )


def rouge_l_f1(left: str, right: str) -> float:
    left = normalize_commentary(left)
    right = normalize_commentary(right)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if ROUGE_L_SCORER is not None:
        return float(ROUGE_L_SCORER.score(left, right)["rougeL"].fmeasure)
    left_tokens = normalize_tokens(left)
    right_tokens = normalize_tokens(right)
    lcs = lcs_length(left_tokens, right_tokens)
    precision = lcs / len(right_tokens)
    recall = lcs / len(left_tokens)
    return f1_score(precision, recall)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    rows = len(left) + 1
    cols = len(right) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if left[i - 1] == right[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def extract_chess_entities(text: str) -> set[str]:
    lowered = normalize_commentary(text).lower()
    entities = set(SQUARE_RE.findall(lowered))
    entities.update(item.lower() for item in SAN_TOKEN_RE.findall(text))
    entities.update(word for word in PIECE_WORDS if word in lowered)
    entities.update(word for word in TACTICAL_WORDS if word in lowered)
    return entities


def chess_entity_f1(left: str, right: str) -> float:
    left_entities = extract_chess_entities(left)
    right_entities = extract_chess_entities(right)
    if not left_entities and not right_entities:
        return 1.0
    if not left_entities or not right_entities:
        return 0.0
    overlap = len(left_entities.intersection(right_entities))
    precision = overlap / len(right_entities)
    recall = overlap / len(left_entities)
    return f1_score(precision, recall)


def extract_claim_tags(text: str) -> set[str]:
    lowered = normalize_commentary(text).lower()
    tags = set()
    for tag, patterns in CLAIM_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                tags.add(tag)
                break
    if "queen" in lowered and ("wins" in lowered or "hang" in lowered):
        tags.add("queen_loss")
    return tags


def derive_truth_tags(
    prev_board: chess.Board,
    curr_board: chess.Board,
    move: chess.Move,
    context: Dict[str, Any],
) -> List[str]:
    tags = set()
    verified = context.get("verified_facts", {}) or {}
    tactical = context.get("tactical", {}) or {}
    analysis = context.get("analysis", {}) or {}
    move_info = context.get("move", {}) or {}
    quality = str(context.get("quality", "") or "")

    if prev_board.is_capture(move):
        tags.add("capture")
    if verified.get("is_recapture"):
        tags.add("recapture")
    if move_info.get("is_castling"):
        tags.add("castling")
    if verified.get("gives_check"):
        tags.add("check")
    if verified.get("gives_checkmate"):
        tags.add("checkmate")
    if verified.get("develops_minor_piece"):
        tags.add("development")
    if verified.get("central_control") or verified.get("central_support") or verified.get("central_pressure"):
        tags.add("center")
    if verified.get("central_support"):
        tags.add("support")

    capture_type = str((tactical.get("capture_analysis") or {}).get("type", ""))
    if capture_type in {"equal_trade", "favorable_trade"}:
        tags.add("trade")
    attack_info = tactical.get("attack_info") or {}
    pressure_info = tactical.get("pressure_info") or {}
    if attack_info.get("is_attacking") or pressure_info.get("is_pressure"):
        tags.add("attack")
    if tactical.get("resolved_threats"):
        tags.add("defense")

    primary = tactical.get("primary_tactic") or {}
    pattern_type = str(primary.get("type", ""))
    if pattern_type == "pin":
        tags.add("pin")
    if pattern_type == "fork":
        tags.add("fork")
    if pattern_type == "skewer":
        tags.add("skewer")
    if pattern_type == "xray":
        tags.add("xray")

    if quality in {"inaccuracy", "mistake", "blunder"}:
        tags.add("bad_move")
    hung = tactical.get("hung_piece") or {}
    if str(hung.get("piece", "")) == "queen":
        tags.add("queen_loss")

    pawn_structure = analysis.get("pawn_structure", {}) or {}
    mover_color = prev_board.turn
    mover_key = "white" if mover_color == chess.WHITE else "black"
    opp_key = "black" if mover_color == chess.WHITE else "white"
    mover_before = pawn_structure.get(mover_key, {}) or {}
    opp_before = pawn_structure.get(opp_key, {}) or {}

    before_snapshot = structure_snapshot(prev_board, mover_color, not mover_color)
    after_snapshot = structure_snapshot(curr_board, mover_color, not mover_color)
    if sorted(set(after_snapshot["opp_doubled"]) - set(before_snapshot["opp_doubled"])):
        tags.add("doubled_pawns")
    if sorted(set(after_snapshot["opp_isolated"]) - set(before_snapshot["opp_isolated"])):
        tags.add("isolated_pawn")
    if sorted(set(after_snapshot["mover_passed"]) - set(before_snapshot["mover_passed"])):
        tags.add("passed_pawn")

    if is_retreat(prev_board, move):
        tags.add("retreat")
    if is_queenside_action(move):
        tags.add("queenside")
    if is_kingside_action(move):
        tags.add("kingside")

    return sorted(tags)


def structure_snapshot(board: chess.Board, mover_color: chess.Color, opp_color: chess.Color) -> Dict[str, List[str]]:
    return {
        "mover_passed": passed_pawns(board, mover_color),
        "opp_doubled": doubled_files(board, opp_color),
        "opp_isolated": isolated_pawns(board, opp_color),
    }


def doubled_files(board: chess.Board, color: chess.Color) -> List[str]:
    files = [0] * 8
    for square in board.pieces(chess.PAWN, color):
        files[chess.square_file(square)] += 1
    return [chess.FILE_NAMES[i] for i, count in enumerate(files) if count > 1]


def isolated_pawns(board: chess.Board, color: chess.Color) -> List[str]:
    pawns = list(board.pieces(chess.PAWN, color))
    files_with_pawns = {chess.square_file(square) for square in pawns}
    isolated: List[str] = []
    for square in pawns:
        file_idx = chess.square_file(square)
        if (file_idx - 1) not in files_with_pawns and (file_idx + 1) not in files_with_pawns:
            isolated.append(chess.square_name(square))
    return sorted(isolated)


def passed_pawns(board: chess.Board, color: chess.Color) -> List[str]:
    enemy = not color
    passed: List[str] = []
    for square in board.pieces(chess.PAWN, color):
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        ok = True
        for enemy_square in board.pieces(chess.PAWN, enemy):
            enemy_file = chess.square_file(enemy_square)
            enemy_rank = chess.square_rank(enemy_square)
            if abs(enemy_file - file_idx) > 1:
                continue
            if color == chess.WHITE and enemy_rank > rank_idx:
                ok = False
                break
            if color == chess.BLACK and enemy_rank < rank_idx:
                ok = False
                break
        if ok:
            passed.append(chess.square_name(square))
    return sorted(passed)


def claim_score(claim_tags: Iterable[str], truth_tags: Iterable[str]) -> Tuple[float, float, float]:
    claims = set(claim_tags)
    truth = set(truth_tags)
    if not claims and not truth:
        return 1.0, 1.0, 1.0
    if not claims:
        return 1.0, 0.0, 0.0
    overlap = len(claims.intersection(truth))
    precision = overlap / len(claims)
    recall = overlap / len(truth) if truth else 1.0
    return precision, recall, f1_score(precision, recall)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def jaccard_score(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set.union(right_set)
    if not union:
        return 1.0
    return len(left_set.intersection(right_set)) / len(union)


def is_retreat(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type == chess.KING:
        return False
    if piece.piece_type == chess.KNIGHT:
        return False
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    return to_rank < from_rank if board.turn == chess.WHITE else to_rank > from_rank


def is_queenside_action(move: chess.Move) -> bool:
    return chess.square_file(move.to_square) <= 2


def is_kingside_action(move: chess.Move) -> bool:
    return chess.square_file(move.to_square) >= 5


def phase_from_ply(ply: int) -> str:
    if ply <= 12:
        return "opening"
    if ply <= 70:
        return "middlegame"
    return "endgame"


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt_or_na(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return fmt(value)


def zero_if_none(value: Optional[float]) -> float:
    return float(value) if value is not None else 0.0


def aggregate_pair_metrics(rows: Sequence[PairSample]) -> Dict[str, Any]:
    if not rows:
        return {
            "moves": 0,
            "bleu4": 0.0,
            "rouge_l_f1": 0.0,
            "semantic_similarity": 0.0,
            "entity_f1": 0.0,
            "claim_tag_jaccard": 0.0,
            "grounded_f1": 0.0,
            "cgas": 0.0,
        }
    return {
        "moves": len(rows),
        "bleu4": safe_mean(row.bleu4 for row in rows),
        "rouge_l_f1": safe_mean(row.rouge_l_f1 for row in rows),
        "semantic_similarity": safe_mean(row.semantic_similarity for row in rows),
        "entity_f1": safe_mean(row.entity_f1 for row in rows),
        "claim_tag_jaccard": safe_mean(row.claim_tag_jaccard for row in rows),
        "grounded_f1": safe_mean(row.candidate_grounded_f1 for row in rows),
        "cgas": safe_mean(row.overall_alignment for row in rows),
    }


def aggregate_pairs_by_key(rows: Sequence[PairSample], key_name: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[PairSample]] = defaultdict(list)
    for row in rows:
        key = getattr(row, key_name)
        grouped[str(key)].append(row)
    results: List[Dict[str, Any]] = []
    for key, grouped_rows in sorted(grouped.items()):
        results.append({key_name: key, **aggregate_pair_metrics(grouped_rows)})
    return results


def render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def render_latex_table(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str, label: str) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def aggregate_per_game(pair_samples: Sequence[PairSample]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[PairSample]] = defaultdict(list)
    for row in pair_samples:
        grouped[(row.corpus_label, row.game_id)].append(row)
    results: List[Dict[str, Any]] = []
    for (corpus_label, game_id), rows in sorted(grouped.items()):
        results.append(
            {
                "corpus_label": corpus_label,
                "game_id": game_id,
                "game_label": f"{rows[0].white[:8]}-{rows[0].black[:8]}-{rows[0].game_signature}",
                "moves": len(rows),
                "semantic_similarity": safe_mean(item.semantic_similarity for item in rows),
                "entity_f1": safe_mean(item.entity_f1 for item in rows),
                "system_grounded_f1": safe_mean(item.candidate_grounded_f1 for item in rows),
                "reference_grounded_f1": safe_mean(item.reference_grounded_f1 for item in rows),
                "overall_alignment": safe_mean(item.overall_alignment for item in rows),
            }
        )
    return results


def grouped_bar_chart_svg(
    title: str,
    categories: Sequence[str],
    series: Dict[str, Sequence[float]],
    path: Path,
    y_label: str,
) -> None:
    width = 1200
    height = 720
    margin = {"top": 90, "right": 40, "bottom": 150, "left": 80}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    if not categories or not series:
        path.write_text("", encoding="utf-8")
        return

    n_series = len(series)
    group_w = plot_w / max(1, len(categories))
    inner_w = group_w * 0.72
    bar_w = inner_w / max(1, n_series)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937} .tick{font-size:12px} .title{font-size:24px;font-weight:700} .label{font-size:14px} .legend{font-size:13px}</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="40" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]

    for i in range(6):
        y = margin["top"] + plot_h - (plot_h * i / 5)
        val = i / 5
        lines.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{width-margin["right"]}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{margin["left"]-10}" y="{y+4}" text-anchor="end" class="tick">{val:.1f}</text>')

    lines.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"]+plot_h}" stroke="#111827" stroke-width="2"/>')
    lines.append(f'<line x1="{margin["left"]}" y1="{margin["top"]+plot_h}" x2="{width-margin["right"]}" y2="{margin["top"]+plot_h}" stroke="#111827" stroke-width="2"/>')
    lines.append(f'<text x="{18}" y="{margin["top"]+plot_h/2}" transform="rotate(-90 18 {margin["top"]+plot_h/2})" class="label">{html.escape(y_label)}</text>')

    series_items = list(series.items())
    for idx, category in enumerate(categories):
        x0 = margin["left"] + idx * group_w + (group_w - inner_w) / 2
        for j, (label, values) in enumerate(series_items):
            value = values[idx] if idx < len(values) else 0.0
            bar_h = plot_h * max(0.0, min(1.0, value))
            x = x0 + j * bar_w
            y = margin["top"] + plot_h - bar_h
            color = colors[j % len(colors)]
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w*0.82:.2f}" height="{bar_h:.2f}" fill="{color}" rx="2"/>')
        tick_x = margin["left"] + idx * group_w + group_w / 2
        lines.append(
            f'<text x="{tick_x:.2f}" y="{height-margin["bottom"]+25}" text-anchor="end" class="tick" transform="rotate(-32 {tick_x:.2f} {height-margin["bottom"]+25})">{html.escape(category)}</text>'
        )

    legend_x = margin["left"]
    legend_y = height - 60
    for i, (label, _) in enumerate(series_items):
        color = colors[i % len(colors)]
        lx = legend_x + i * 180
        lines.append(f'<rect x="{lx}" y="{legend_y}" width="16" height="16" fill="{color}" rx="2"/>')
        lines.append(f'<text x="{lx+24}" y="{legend_y+13}" class="legend">{html.escape(label)}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def heatmap_svg(
    title: str,
    rows: Sequence[str],
    cols: Sequence[str],
    values: Sequence[Sequence[float]],
    path: Path,
) -> None:
    width = 960
    height = 620
    margin = {"top": 100, "right": 50, "bottom": 70, "left": 190}
    cell_w = (width - margin["left"] - margin["right"]) / max(1, len(cols))
    cell_h = (height - margin["top"] - margin["bottom"]) / max(1, len(rows))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937} .tick{font-size:13px} .title{font-size:24px;font-weight:700}</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="40" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]
    for col_idx, col in enumerate(cols):
        x = margin["left"] + col_idx * cell_w + cell_w / 2
        lines.append(f'<text x="{x}" y="{margin["top"]-15}" text-anchor="middle" class="tick">{html.escape(col)}</text>')
    for row_idx, row in enumerate(rows):
        y = margin["top"] + row_idx * cell_h + cell_h / 2 + 5
        lines.append(f'<text x="{margin["left"]-10}" y="{y}" text-anchor="end" class="tick">{html.escape(row)}</text>')
        for col_idx, value in enumerate(values[row_idx]):
            x = margin["left"] + col_idx * cell_w
            y0 = margin["top"] + row_idx * cell_h
            color = heat_color(value)
            lines.append(f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="white" stroke-width="2"/>')
            lines.append(f'<text x="{x+cell_w/2}" y="{y0+cell_h/2+5}" text-anchor="middle" class="tick">{value:.2f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def heat_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    red = int(255 * (1 - value))
    green = int(180 + 75 * value)
    blue = int(220 * (1 - value))
    return f"rgb({red},{green},{blue})"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate your commentary system against annotated PGN corpora.")
    parser.add_argument("--grandmaster", default="Grandmaster Commentry.pgn", help="Path to the grandmaster commentary PGN corpus")
    parser.add_argument("--chessdotcom", default="chessdotcomCommentry.pgn", help="Path to the Chess.com commentary PGN corpus")
    parser.add_argument("--output-dir", default="paper_benchmark", help="Directory for the readable benchmark package")
    parser.add_argument("--max-games", type=int, default=None, help="Optional cap on number of games per corpus")
    parser.add_argument(
        "--semantic-backend",
        choices=["auto", "sentence-transformers", "tfidf", "token"],
        default="auto",
        help="Semantic similarity backend. `auto` prefers a cached sentence-transformer and falls back locally.",
    )
    parser.add_argument(
        "--semantic-model",
        default="all-mpnet-base-v2",
        help="Preferred local sentence-transformer model name when using `auto` or `sentence-transformers`.",
    )
    parser.add_argument(
        "--include-appendix",
        action="store_true",
        help="Write move-level appendix data in addition to the compact paper-facing outputs.",
    )
    args = parser.parse_args(argv)

    evaluator = CommentaryEvaluator(
        semantic_backend=args.semantic_backend,
        semantic_model=args.semantic_model,
        include_appendix=args.include_appendix,
    )
    summary = evaluator.evaluate(
        grandmaster_path=Path(args.grandmaster),
        chessdotcom_path=Path(args.chessdotcom),
        output_dir=Path(args.output_dir),
        max_games=args.max_games,
    )

    print("Evaluation complete.")
    print(f"Embedding backend: {summary['embedding_backend']}")
    for row in summary["benchmark"]:
        print(
            f"{row['model']}: subset={row['subset']} moves={row.get('moves', 0)} "
            f"CGAS={fmt_or_na(row.get('cgas'))} grounded={fmt_or_na(row.get('grounded_f1'))}"
        )
    print(summary["fairness_note"])
    print(f"Outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
