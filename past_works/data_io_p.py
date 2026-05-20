"""
Low-level I/O for the two raw CEINN datasets.

This module is intentionally *pure*: every function here only reads, parses,
and yields records from disk. It performs no filtering, no remapping, no
feature engineering, and no validation. Higher-level orchestration
(Phase 1 validation, Phase 2 preprocessing) lives in `preprocess.py`.

Why a separate module?
    Several phases need the same byte-accurate parsers for the same files.
    Centralising them here eliminates the risk of two slightly-different
    re-implementations silently disagreeing on edge cases (e.g. the
    Python-literal vs. strict-JSON quirk in Amazon's dumps, or the
    ISO-8859-1 fallback for movies.dat). Touch this file rarely.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


# =============================================================================
# Amazon Beauty: JSON-line files
# =============================================================================
#
# Both `reviews_Beauty.json` (and its 5-core subset `Beauty_5.json`) and
# `meta_Beauty.json` follow Julian McAuley's distribution format: one record
# per line. Two distinct text encodings of those records exist in the wild:
#
#   (A) Strict JSON  — newer dumps. Parseable by json.loads.
#                      Example: {"reviewerID": "A1...", "overall": 5.0, ...}
#   (B) Python dict literal — older dumps. Single-quoted, not strict JSON.
#                      Example: {'reviewerID': 'A1...', 'overall': 5.0, ...}
#
# We try (A) first because it's strictly faster and safer; on failure we fall
# back to ast.literal_eval which handles (B) without invoking eval().
# -----------------------------------------------------------------------------

def _parse_record_robust(line: str) -> Dict[str, Any]:
    """Parse a single record line, tolerating both JSON and Python-literal."""
    line = line.strip()
    if not line:
        raise ValueError("empty record line")
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # ast.literal_eval is safe (no arbitrary code execution).
        return ast.literal_eval(line)


def iter_amazon_reviews(path: str | Path) -> Iterator[Dict[str, Any]]:
    """
    Yield one review dict per line.

    Expected fields (subset; not all reviews carry every field):
        reviewerID      : str    -- user identifier
        asin            : str    -- item identifier
        overall         : float  -- rating, 1.0..5.0 in integer steps
        unixReviewTime  : int    -- seconds since epoch (UTC)
        reviewText      : str    -- free text (unused by CEINN)
        ...
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                yield _parse_record_robust(raw)
            except (json.JSONDecodeError, SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Malformed Amazon review at {path}:{line_no} — {exc}"
                ) from exc


def iter_amazon_meta(path: str | Path) -> Iterator[Dict[str, Any]]:
    """
    Yield one metadata dict per line.

    Expected fields (all optional; missing is the dominant case for some):
        asin        : str
        price       : float | None
        salesRank   : dict[str, int] | None
                      e.g. {"Beauty": 12345} or {"Beauty & Personal Care": ...}
        categories  : list[list[str]] | None
                      a list of hierarchical paths, e.g.
                      [["Beauty", "Skin Care", "Lotions & Moisturizers"]]
        brand       : str | None
        title       : str | None
        ...
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                yield _parse_record_robust(raw)
            except (json.JSONDecodeError, SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Malformed Amazon meta record at {path}:{line_no} — {exc}"
                ) from exc


# -----------------------------------------------------------------------------
# Amazon metadata field extractors
# -----------------------------------------------------------------------------

def extract_leaf_categories(categories) -> list[str]:
    if not categories:
        return []
    # 只取最深路徑的葉節點，與 EDA 的 max(cats, key=len) 一致
    deepest = max(categories, key=len)
    if deepest:
        leaf = deepest[-1]
        return [leaf] if isinstance(leaf, str) and leaf.strip() else []
    return []

def extract_salesrank(
    salesrank_field: Any,
    prefer_key: str = "Beauty",
) -> Tuple[int | None, str]:
    """
    Resolve a single salesRank value from the dict-valued `salesRank` field.

    Per the EDA, two coverage figures need to be reproducible:
      - "any-category" coverage 98.2%  → fall back to any positive rank
      - "Beauty"      coverage 88.3%  → only count the preferred key

    Returns (rank_value, source) where source ∈ {"preferred_key", "any_key",
    "missing"}. Callers use `source` to attribute the coverage correctly.
    """
    if salesrank_field is None or not isinstance(salesrank_field, dict):
        return None, "missing"

    # 1) Try the preferred (more specific) key.
    if prefer_key in salesrank_field:
        v = salesrank_field[prefer_key]
        if isinstance(v, (int, float)) and v > 0:
            return int(v), "preferred_key"

    # 2) Fall back to any other category's rank.
    for _key, v in salesrank_field.items():
        if isinstance(v, (int, float)) and v > 0:
            return int(v), "any_key"

    return None, "missing"


def has_valid_price(price_field: Any) -> bool:
    """A price counts toward coverage iff it is a positive finite number."""
    if not isinstance(price_field, (int, float)):
        return False
    # Reject NaN / inf / non-positive prices.
    return (price_field > 0) and (price_field == price_field) and (price_field != float("inf"))


def has_valid_brand(brand_field: Any) -> bool:
    """A brand counts toward coverage iff it is a non-empty string."""
    return isinstance(brand_field, str) and bool(brand_field.strip())


# =============================================================================
# MovieLens 10M: "::"-delimited .dat files
# =============================================================================

def iter_movielens_ratings(path: str | Path) -> Iterator[Tuple[int, int, float, int]]:
    """
    Yield (user_id, movie_id, rating, timestamp) tuples.

    Format per line:   "UserID::MovieID::Rating::Timestamp\\n"
                       rating is 0.5..5.0 in 0.5 steps;
                       timestamp is seconds since Unix epoch.

    File is plain ASCII; UTF-8 is sufficient.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 4:
                raise ValueError(
                    f"Malformed ratings row at {path}:{line_no} — got {len(parts)} fields"
                )
            uid, mid, rating, ts = parts
            yield int(uid), int(mid), float(rating), int(ts)


def iter_movielens_movies(
    path: str | Path,
    encoding: str = "iso-8859-1",
) -> Iterator[Tuple[int, str, List[str]]]:
    """
    Yield (movie_id, title, genres) tuples.

    Format per line:   "MovieID::Title (Year)::Genre1|Genre2|...\\n"

    The title may contain "::" theoretically; in practice it does not in
    MovieLens 10M, so we use a strict 3-way split. The encoding defaults to
    ISO-8859-1 because some titles contain Latin-1 accented characters that
    cannot be decoded as UTF-8 (e.g., "Cité de Dieu", "Amélie").

    Genres are split on '|'. The sentinel "(no genres listed)" is preserved
    in the returned list; callers decide whether to keep or drop it.
    """
    with open(path, "r", encoding=encoding) as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed movies row at {path}:{line_no} — got {len(parts)} fields"
                )
            mid, title, genres_str = parts
            genres = [g.strip() for g in genres_str.split("|") if g.strip()]
            yield int(mid), title, genres


def iter_movielens_tags(
    path: str | Path,
) -> Iterator[Tuple[int, int, str, int]]:
    """
    Yield (user_id, movie_id, tag, timestamp) tuples.

    Format per line:   "UserID::MovieID::Tag::Timestamp\\n"

    Tag text is free-form and may include punctuation and mixed case. We
    preserve case verbatim here so downstream callers can decide on case
    folding (e.g., the EDA `tag_vocab` of 15,241 is case-sensitive).

    If a tag legitimately contains "::" we recover by treating the first two
    and last fields as fixed and rejoining the middle parts. The MovieLens
    10M tags file in fact contains no such cases, but the guard is cheap.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("::")
            if len(parts) == 4:
                uid, mid, tag, ts = parts
            elif len(parts) > 4:
                # Unusual: tag contained "::". Rejoin the middle.
                uid, mid = parts[0], parts[1]
                ts = parts[-1]
                tag = "::".join(parts[2:-1])
            else:
                raise ValueError(
                    f"Malformed tags row at {path}:{line_no} — got {len(parts)} fields"
                )
            yield int(uid), int(mid), tag, int(ts)
