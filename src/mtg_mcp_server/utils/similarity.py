"""Rules-text similarity between Magic cards.

Used by ``similar_cards`` to answer "what else does what this card does?".

A card is described three ways, and two cards' similarity is a weighted sum of
three cosine similarities, so it runs from 0 to 1:

- **Text (50%)**: the words of its oracle text and adjacent word pairs,
  weighted by inverse document frequency over the pool, so boilerplate
  ("whenever", "creature") counts for little and distinctive phrasing
  ("card unless") for a lot. Templating is normalised first: the card's own
  name and "this creature" become ``cardname``, reminder text is dropped,
  generic costs become ``{n}`` and coloured mana ``{m}`` (colourless ``{c}``
  stays apart, so Sol Ring reads like other {C}{C} rocks rather than
  Signets), filler words and numbers are dropped, and plurals are stemmed.
- **Roles (25%)**: what each ability does for its controller (draw, ramp,
  removal, tax, tokens and so on), found by text patterns, plus a "+" variant
  when the ability repeats (a trigger or an activated ability rather than a
  one-shot spell). Tax ("unless that player pays") qualifies an effect rather
  than being one, so it weighs half.
- **Type (25%)**: the words of the type line, IDF-weighted.

Results are ordered by similarity x (0.4 + 0.6 x popularity), where
popularity falls from 1 at EDHREC rank 1 to 0 at rank 40,000 or no rank.
Text alone can't tell a card's purpose from its trigger: obscure cards that
share Rhystic Study's "whenever an opponent casts a spell" outranked Phyrexian
Arena and Smothering Tithe, the card-advantage engines players actually
compare it with. The returned score stays the similarity, so identical rules
text is 100%; only the order includes popularity.

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

__all__ = ["SimilarityIndex", "card_features", "card_roles", "type_features"]

# How much each description of a card counts; they sum to 1.
_TEXT_WEIGHT = 0.5
_ROLE_WEIGHT = 0.25
_TYPE_WEIGHT = 0.25

# Ordering: similarity x (1 - _POPULARITY_WEIGHT + _POPULARITY_WEIGHT x popularity).
_POPULARITY_WEIGHT = 0.6
_UNPOPULAR_RANK = 40_000  # EDHREC rank at which popularity reaches 0

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

# --- Roles: what an ability does for its controller ---------------------------

# "draw" only counts when you draw: not "that player draws seven cards".
_NOT_YOU_RE = re.compile(
    r"(?:opponent|that player|its controller|their controller|defending player"
    r"|its owner|that creature's controller)s?\W*(?:\w+\W+){0,1}$"
)
_DRAW_RE = re.compile(
    r"\bdraws? (?:a|an|one|two|three|four|five|six|seven|x|that many|cards?"
    r"|\w+ additional|up to \w+)\b"
)
_ROLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "tohand": re.compile(
        r"put (?:that card|it|them|those cards|one of them|\w+ of them) into your hand"
    ),
    "tax": re.compile(
        r"unless (?:that player|they|its controller|he or she|that spell's controller|"
        r"their controller|the player|its owner|any player) pays?\b"
        r"|may pay \{[^}]*\}[^.]*\.? if (?:the player|they|that player) (?:doesn't|don't|does not)"
    ),
    "ramp": re.compile(
        r"\badd \{|\badd (?:one|two|three|x|that much|an amount of) |create (?:a|an|\w+) treasure"
        r"|search your library for [^.]*\b(?:land|forest|plains|island|swamp|mountain)s? cards?"
        r"[^.]*onto the battlefield|put (?:a|up to \w+) land cards? from your hand onto"
        r"|untap target (?:land|forest|plains|island|swamp|mountain)"
    ),
    "removal": re.compile(
        r"\b(?:destroy|exile) (?:target|up to \w+ (?:other )?target|another target)[^.]*?"
        r"\b(?:creature|permanent|planeswalker)s?\b"
        r"|\bput target [^.]*?\b(?:creature|permanent)s? [^.]*?on (?:the )?(?:top|bottom)"
        r"|target (?:player|opponent) sacrifices"
        r"|put target [^.]*?into its owner's library"
    ),
    "artifact_removal": re.compile(
        r"\b(?:destroy|exile) (?:target|up to \w+ target|another target)"
        r" (?:\w+ )?(?:artifact|enchantment)"
    ),
    "wipe": re.compile(
        r"\b(?:destroy|exile) (?:all|each)\b[^.]*?\b(?:creature|permanent|nonland)s?\b"
        r"|deals? (?:\d+|x) damage to each creature|each player sacrifices"
    ),
    "damage": re.compile(
        r"deals? (?:\d+|x|that much) damage to (?:any target|target|each opponent"
        r"|that player|any one target|each of up to)"
    ),
    "counterspell": re.compile(
        r"\bcounter (?:target|up to \w+ target)\b[^.]*?\b(?:spell|ability)"
        r"|\bcounter (?:it|that spell)\b"
    ),
    "tokens": re.compile(r"\bcreate [^.]*?\bcreature tokens?\b"),
    "counters": re.compile(r"\+1/\+1 counters?"),
    "doubler": re.compile(r"twice that many|three times that many|that many plus one|double the "),
    "drain": re.compile(
        r"\b(?:each opponent|target player|target opponent|that player|"
        r"defending player|each player) loses? (?:\d+|x|that much) life"
    ),
    "lifegain": re.compile(r"\byou gain (?:\d+|x|that much) life"),
    "mill": re.compile(r"\bmills? (?:(?:a|an|\w+) )?cards?"),
    "discard": re.compile(
        r"\b(?:target player|target opponent|each opponent|that player|each player) discards?"
    ),
    "tutor": re.compile(
        r"search your library for (?:a|an|up to \w+) (?!basic land|land|forest|"
        r"plains|island|swamp|mountain)[^.]*?cards?"
    ),
    "recursion": re.compile(r"return [^.]*?from your graveyard to (?:your hand|the battlefield)"),
    "bounce": re.compile(
        r"return (?:target|up to \w+ target|each|all)[^.]*? to (?:its|their) owners?'? hands?"
    ),
    "scry": re.compile(r"\b(?:scry|surveil) (?:\d+|x)\b"),
    "pump": re.compile(r"gets? \+\d+/\+\d+ until end of turn"),
    "anthem": re.compile(r"creatures you control get \+\d+/\+\d+"),
    "protection": re.compile(r"\bgains? (?:hexproof|indestructible|protection|shroud)\b"),
}
# The cost of a sacrifice outlet: "Sacrifice a creature: ...".
_SACRIFICE_COST_RE = re.compile(
    r"\bsacrifice (?:a|an|another|one or more|\w+ other) [^:.]*?(?:creature|permanent|artifact)s?\b"
)
_TRIGGER_START_RE = re.compile(r"^\s*(?:whenever|when|at the beginning|at end|at the end)\b")
_ACTIVATED_RE = re.compile(r"^([^:\"]+):\s")
_ONE_SHOT_COST_RE = re.compile(r"\b(?:sacrifice|exile) cardname\b")
_SPELL_TYPES_RE = re.compile(r"\b(?:instant|sorcery)\b")
_TAX_ROLE_WEIGHT = 0.5


def _normalize_symbol(symbol: str) -> str:
    """Generalise a braced symbol: generic costs -> ``{n}``, coloured mana -> ``{m}``."""
    inner = symbol[1:-1]
    if inner in ("t", "q", "c"):  # tap, untap and colourless mana keep their meaning
        return symbol
    if inner.isdigit() or inner in ("x", "y", "z"):
        return "{n}"  # "pays {1}" and "pays {4}" are the same effect
    return "{m}"  # coloured, hybrid, Phyrexian, snow mana


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


def _normalized_text(card: Card) -> str:
    """Lowercase oracle text without reminder text, self-references as ``cardname``."""
    text = _REMINDER_RE.sub(" ", (card.oracle_text or "").lower())
    for name in _self_names(card):
        if name and name in text:
            # Letter look-arounds instead of \b: names may start or end with
            # punctuation, and "ice" must not match inside "twice".
            text = re.sub(rf"(?<![a-z]){re.escape(name)}(?![a-z])", "cardname", text)
    return _SELF_REFERENCE_RE.sub("cardname", text)


def card_features(card: Card) -> frozenset[str]:
    """Return the normalised words and word pairs of *card*'s oracle text."""
    features: set[str] = set()
    for sentence in _SENTENCE_SPLIT_RE.split(_normalized_text(card)):
        tokens = _sentence_tokens(sentence)
        features.update(tokens)
        features.update(f"{a} {b}" for a, b in itertools.pairwise(tokens))
    return frozenset(features)


def type_features(card: Card) -> frozenset[str]:
    """Return the words of *card*'s type line, prefixed ``type:``."""
    return frozenset("type:" + word for word in re.findall(r"[a-z]+", card.type_line.lower()))


def _line_roles(line: str) -> set[str]:
    """Roles of one ability (one line of normalised oracle text)."""
    roles = {role for role, pattern in _ROLE_PATTERNS.items() if pattern.search(line)}
    for match in _DRAW_RE.finditer(line):
        if not _NOT_YOU_RE.search(line[max(0, match.start() - 40) : match.start()]):
            roles.add("draw")
            break
    if "tohand" in roles:  # putting cards into your hand is card advantage too
        roles.discard("tohand")
        roles.add("draw")
    if (cost := _ACTIVATED_RE.match(line)) and _SACRIFICE_COST_RE.search(cost.group(1)):
        roles.add("sacoutlet")
    return roles


def card_roles(card: Card) -> frozenset[str]:
    """Return *card*'s roles, plus ``role+`` for each one an ability repeats.

    A trigger ("whenever", "at the beginning") or an activated ability repeats;
    an instant or sorcery, a "when" trigger, or an ability that sacrifices or
    exiles the card itself doesn't.
    """
    out: set[str] = set()
    is_spell = bool(_SPELL_TYPES_RE.search(card.type_line.lower().split("//")[0]))
    previous_repeats = not is_spell
    for line in _normalized_text(card).split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("•"):
            repeats = previous_repeats  # a mode inherits its ability's timing
        elif _TRIGGER_START_RE.match(line):
            repeats = not line.startswith("when ")
        elif cost := _ACTIVATED_RE.match(line):
            repeats = not _ONE_SHOT_COST_RE.search(cost.group(1))
        else:
            repeats = not is_spell
        previous_repeats = repeats
        for role in _line_roles(line):
            out.add(role)
            if repeats:
                out.add(role + "+")
    return frozenset(out)


def _role_weight(role: str) -> float:
    """Weight of a role feature: tax is a qualifier, so it counts half."""
    return _TAX_ROLE_WEIGHT if role.rstrip("+") == "tax" else 1.0


def _idf(pool_size: int, doc_freq: int) -> float:
    """Smoothed inverse document frequency."""
    return math.log((1 + pool_size) / (1 + doc_freq)) + 1.0


def _popularity(edhrec_rank: int | None) -> float:
    """1.0 for EDHREC rank 1, falling log-linearly to 0.0 at rank 40,000 or no rank."""
    if edhrec_rank is None or edhrec_rank < 1:
        return 0.0
    return max(0.0, 1.0 - math.log(edhrec_rank) / math.log(_UNPOPULAR_RANK))


@dataclass(frozen=True)
class SimilarityIndex:
    """Precomputed features and weights for one card pool.

    Holds no Card objects: :meth:`rank` takes the same pool, in the same order,
    that the index was built from.

    Attributes:
        features: Per card (parallel to the pool), its text features shared with
            at least one other card. A feature unique to one card can never
            match another, so it only counts in that card's norm.
        roles: Per card, its role features.
        types: Per card, its type-line features.
        text_weights: IDF of each shared text feature.
        type_weights: IDF of each type feature.
        text_norms, role_norms, type_norms: Per card, its vector length in each
            description, over all its features.
        unique_weight: IDF of a feature seen on only one card (also used for a
            source card's features the pool doesn't share).
    """

    features: list[tuple[str, ...]]
    roles: list[tuple[str, ...]]
    types: list[tuple[str, ...]]
    text_weights: dict[str, float]
    type_weights: dict[str, float]
    text_norms: list[float]
    role_norms: list[float]
    type_norms: list[float]
    unique_weight: float

    @classmethod
    def build(cls, cards: Sequence[Card]) -> SimilarityIndex:
        """Describe every card and compute document frequencies (CPU-bound)."""
        vocab: dict[str, str] = {}  # interns feature strings shared across cards
        texts: list[tuple[str, ...]] = []
        roles: list[tuple[str, ...]] = []
        types: list[tuple[str, ...]] = []
        text_df: dict[str, int] = {}
        type_df: dict[str, int] = {}
        for card in cards:
            text = tuple(vocab.setdefault(f, f) for f in card_features(card))
            kind = tuple(vocab.setdefault(f, f) for f in type_features(card))
            texts.append(text)
            types.append(kind)
            roles.append(tuple(vocab.setdefault(f, f) for f in sorted(card_roles(card))))
            for f in text:
                text_df[f] = text_df.get(f, 0) + 1
            for f in kind:
                type_df[f] = type_df.get(f, 0) + 1

        pool_size = len(cards)
        all_text_weights = {f: _idf(pool_size, df) for f, df in text_df.items()}
        type_weights = {f: _idf(pool_size, df) for f, df in type_df.items()}
        shared = {f: w for f, w in all_text_weights.items() if text_df[f] > 1}
        return cls(
            features=[tuple(f for f in text if f in shared) for text in texts],
            roles=roles,
            types=types,
            text_weights=shared,
            type_weights=type_weights,
            text_norms=[math.sqrt(sum(all_text_weights[f] ** 2 for f in t)) for t in texts],
            role_norms=[math.sqrt(sum(_role_weight(f) ** 2 for f in r)) for r in roles],
            type_norms=[math.sqrt(sum(type_weights[f] ** 2 for f in t)) for t in types],
            unique_weight=_idf(pool_size, 1),
        )

    def rank(
        self,
        source: Card,
        cards: Sequence[Card],
        *,
        limit: int,
        keep: Callable[[Card], bool] | None = None,
    ) -> list[tuple[float, Card]]:
        """Return up to *limit* ``(similarity, card)`` pairs.

        Ordered by similarity weighted toward widely played cards (see the
        module docstring); the similarity itself is what's returned. *cards*
        must be the pool the index was built from: a pool of a different size
        raises ``ValueError``. The source card itself is skipped, as is any
        card for which *keep* returns False.
        """
        if len(cards) != len(self.features):
            raise ValueError(
                f"Index was built from {len(self.features)} cards; rank() got {len(cards)}"
            )
        # Source vectors: squared weights, so a shared feature adds w_source * w_card
        text_sq = {
            f: self.text_weights.get(f, self.unique_weight) ** 2 for f in card_features(source)
        }
        role_sq = {f: _role_weight(f) ** 2 for f in card_roles(source)}
        type_sq = {
            f: self.type_weights.get(f, self.unique_weight) ** 2 for f in type_features(source)
        }
        text_norm = math.sqrt(sum(text_sq.values()))
        role_norm = math.sqrt(sum(role_sq.values()))
        type_norm = math.sqrt(sum(type_sq.values()))

        # Weigh only the descriptions the source has (a vanilla creature has no text or roles)
        weights = (
            _TEXT_WEIGHT if text_norm else 0.0,
            _ROLE_WEIGHT if role_norm else 0.0,
            _TYPE_WEIGHT if type_norm else 0.0,
        )
        total = sum(weights)
        if total == 0.0:
            return []
        w_text, w_role, w_type = (w / total for w in weights)
        # A card must share text or a role with a source that has any; type
        # words alone ("creature") only count for a vanilla source.
        needs_overlap = bool(text_norm or role_norm)

        source_name = source.name.lower()
        scored: list[tuple[float, float, Card]] = []
        for i, card in enumerate(cards):
            text_dot = sum(text_sq[f] for f in self.features[i] if f in text_sq) if w_text else 0.0
            role_dot = sum(role_sq[f] for f in self.roles[i] if f in role_sq) if w_role else 0.0
            if needs_overlap and not (text_dot or role_dot):
                continue
            type_dot = sum(type_sq[f] for f in self.types[i] if f in type_sq) if w_type else 0.0
            if not (text_dot or role_dot or type_dot) or card.name.lower() == source_name:
                continue
            if keep is not None and not keep(card):
                continue
            score = 0.0
            if text_dot:
                score += w_text * text_dot / (text_norm * self.text_norms[i])
            if role_dot:
                score += w_role * role_dot / (role_norm * self.role_norms[i])
            if type_dot:
                score += w_type * type_dot / (type_norm * self.type_norms[i])
            order = score * (
                1.0 - _POPULARITY_WEIGHT + _POPULARITY_WEIGHT * _popularity(card.edhrec_rank)
            )
            scored.append((order, score, card))

        scored.sort(
            key=lambda t: (-t[0], t[2].edhrec_rank if t[2].edhrec_rank is not None else math.inf)
        )
        return [(score, card) for _, score, card in scored[:limit]]
