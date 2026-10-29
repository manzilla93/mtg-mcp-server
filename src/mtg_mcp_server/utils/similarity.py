"""Rules-text similarity between Magic cards (TF-IDF features + cosine).

Used by ``similar_cards`` to answer "what else does what this card does?".

How a card is compared:

1. It is turned into a set of *features*:
   - words from its oracle text, normalised so templating differences don't
     matter: its own name and "this creature"-style self-references become
     ``cardname``, reminder text is dropped, mana symbols are generalised
     (``{1}`` and ``{4}`` are both ``{n}``), numbers and filler words are
     dropped, and simple plurals are stemmed ("draws" -> "draw");
   - adjacent word pairs within a sentence ("draw card", "player pay"), which
     say what a card *does* far better than loose words;
   - the words of its type line, prefixed ``type:``.
2. Each feature is weighted by inverse document frequency over the whole card
   pool, so boilerplate thousands of cards share ("whenever", "creature")
   counts for little and distinctive phrasing ("card unless") counts a lot.
3. Similarity is the cosine of two cards' weighted feature sets, 0..1.

Building the index for the full pool (~30k cards) takes a few seconds, so it
is built once per bulk-data snapshot and reused by every query. The index
holds no Card objects (callers pass the pool to ``rank``), so caching it never
keeps an old snapshot alive after the bulk data refreshes.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mtg_mcp_server.types import Card

__all__ = ["SimilarityIndex", "card_features"]

# Reminder text "(It's an artifact with ...)" explains rules, not what the card does.
_REMINDER_RE = re.compile(r"\([^)]*\)")

# Current Oracle templating calls the card "this creature", "this enchantment",
# etc.; older cards use their own name. Both become "cardname".
_SELF_REFERENCE_RE = re.compile(
    r"\bthis (?:artifact|aura|battle|card|creature|enchantment|equipment|land|"
    r"permanent|planeswalker|saga|spell|vehicle)\b"
)

# Word pairs are only formed inside one sentence, line or modal bullet (•).
_SENTENCE_SPLIT_RE = re.compile(r"[.;\n•]")

# A symbol such as "{2}", "{W/U}" or "{T}", or a word ("+1/+1" stays whole).
_TOKEN_RE = re.compile(r"\{[^}]*\}|[a-z0-9+/'\-]+")

# Filler that says nothing about what a card does. Dropping it (and numbers)
# makes "draw a card" and "draw two cards" both read "draw card".
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "that",
        "it",
        "its",
        "is",
        "are",
        "be",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "on",
        "onto",
        "with",
        "then",
        "than",
        "this",
        "these",
        "those",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "x",
    }
)


def _normalize_symbol(symbol: str) -> str:
    """Generalise a braced symbol: generic costs -> ``{n}``, other mana -> ``{m}``."""
    inner = symbol[1:-1]
    if inner in ("t", "q"):  # tap / untap keep their meaning
        return symbol
    if inner.isdigit() or inner in ("x", "y", "z"):
        return "{n}"  # "pays {1}" and "pays {4}" are the same effect
    return "{m}"  # coloured, colourless, hybrid, Phyrexian, snow mana


def _stem(word: str) -> str:
    """Crude plural/verb stemming: "draws" -> "draw", "abilities" -> "ability"."""
    if word.endswith("'s"):
        word = word[:-2]  # possessive: "opponent's" -> "opponent"
    elif word.endswith("s'"):
        word = word[:-1]  # plural possessive: "owners'" -> "owners"
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _sentence_tokens(sentence: str) -> list[str]:
    """Normalised tokens of one sentence, in order (filler words dropped)."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(sentence):
        if raw.startswith("{"):
            tokens.append(_normalize_symbol(raw))
            continue
        word = raw.strip("'-")
        if not word or word.isdigit() or word in _STOP_WORDS:
            continue
        tokens.append(_stem(word))
    return tokens


def _self_names(card: Card) -> list[str]:
    """Lowercase names the card's own text may use for itself, longest first."""
    names: set[str] = set()
    for face in card.name.lower().split(" // "):
        names.add(face)
        short = face.split(",")[0]  # "muldrotha, the gravetide" is "muldrotha" in text
        if short != face and len(short) >= 4:
            names.add(short)
    return sorted(names, key=len, reverse=True)


def card_features(card: Card) -> frozenset[str]:
    """Return the normalised word, word-pair and type features of *card*."""
    text = _REMINDER_RE.sub(" ", (card.oracle_text or "").lower())
    for name in _self_names(card):
        if name and name in text:
            # Letter look-arounds instead of \b: names may start or end with
            # punctuation, and "ice" must not match inside "twice".
            text = re.sub(rf"(?<![a-z]){re.escape(name)}(?![a-z])", "cardname", text)
    text = _SELF_REFERENCE_RE.sub("cardname", text)

    features: set[str] = set()
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        tokens = _sentence_tokens(sentence)
        features.update(tokens)
        features.update(f"{a} {b}" for a, b in itertools.pairwise(tokens))
    features.update("type:" + word for word in re.findall(r"[a-z]+", card.type_line.lower()))
    return frozenset(features)


def _idf_squared(pool_size: int, doc_freq: int) -> float:
    """Squared smoothed IDF; squared because features are binary (dot = sum of idf^2)."""
    return (math.log((1 + pool_size) / (1 + doc_freq)) + 1.0) ** 2


@dataclass(frozen=True)
class SimilarityIndex:
    """Precomputed features and weights for one card pool.

    Holds no Card objects: :meth:`rank` takes the same pool, in the same order,
    that the index was built from.

    Attributes:
        features: Per card (parallel to the pool), its features shared with at
            least one other card. Features unique to one card can never match
            another card, so they are kept only in that card's norm.
        weights: Squared IDF per shared feature.
        norms: Per card, the vector length over *all* its features.
        unique_weight: Weight of a feature seen on only one card (also used for
            features of a source card that is not in the pool).
    """

    features: list[tuple[str, ...]]
    weights: dict[str, float]
    norms: list[float]
    unique_weight: float

    @classmethod
    def build(cls, cards: Sequence[Card]) -> SimilarityIndex:
        """Featurise every card and compute document frequencies (CPU-bound)."""
        vocab: dict[str, str] = {}  # interns feature strings shared across cards
        per_card: list[tuple[str, ...]] = []
        doc_freq: dict[str, int] = {}
        for card in cards:
            feats = tuple(vocab.setdefault(f, f) for f in card_features(card))
            per_card.append(feats)
            for f in feats:
                doc_freq[f] = doc_freq.get(f, 0) + 1

        pool_size = len(cards)
        all_weights = {f: _idf_squared(pool_size, df) for f, df in doc_freq.items()}
        norms = [math.sqrt(sum(all_weights[f] for f in feats)) for feats in per_card]
        shared = {f: w for f, w in all_weights.items() if doc_freq[f] > 1}
        features = [tuple(f for f in feats if f in shared) for feats in per_card]
        return cls(
            features=features,
            weights=shared,
            norms=norms,
            unique_weight=_idf_squared(pool_size, 1),
        )

    def rank(
        self,
        source: Card,
        cards: Sequence[Card],
        *,
        limit: int,
        keep: Callable[[Card], bool] | None = None,
    ) -> list[tuple[float, Card]]:
        """Return up to *limit* ``(similarity, card)`` pairs, most similar first.

        *cards* must be the pool the index was built from; a pool of a different
        size raises ``ValueError``. The source card itself is skipped, as is any
        card for which *keep* returns False. Ties go to the more popular card
        (lower EDHREC rank).
        """
        if len(cards) != len(self.features):
            raise ValueError(
                f"Index was built from {len(self.features)} cards; rank() got {len(cards)}"
            )
        source_weights = {f: self.weights.get(f, self.unique_weight) for f in card_features(source)}
        source_norm = math.sqrt(sum(source_weights.values()))
        if source_norm == 0:
            return []

        source_name = source.name.lower()
        scored: list[tuple[float, Card]] = []
        for card, feats, norm in zip(cards, self.features, self.norms, strict=True):
            if not feats or card.name.lower() == source_name:
                continue
            dot = 0.0
            for f in feats:
                weight = source_weights.get(f)
                if weight is not None:
                    dot += weight
            if dot == 0.0 or (keep is not None and not keep(card)):
                continue
            scored.append((dot / (source_norm * norm), card))

        scored.sort(
            key=lambda pair: (
                -pair[0],
                pair[1].edhrec_rank if pair[1].edhrec_rank is not None else math.inf,
            )
        )
        return scored[:limit]
