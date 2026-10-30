"""Mana pip counting, land suggestion, and mana rock detection utilities."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mtg_mcp_server.types import Card

# A mana ability with {T} in its cost: "{T}: Add {C}{C}", "{1}, {T}: Add {W}{B}",
# "{T}, Sacrifice this artifact: Add three mana of any one color". The cost
# runs from {T} to the colon on the same line. Requiring {T} leaves out
# sacrifice outlets such as Ashnod's Altar ("Sacrifice a creature: Add {C}{C}").
_MANA_ABILITY_RE = re.compile(r"\{T\}[^:\n]*:\s*add\b", re.IGNORECASE)

# What people call mana rocks when they ask for them as a category or theme.
MANA_ROCK_NAMES = frozenset({"mana rocks", "mana rock", "rocks", "mana artifacts", "mana artifact"})

# Reminder text, e.g. a Treasure's "(It's an artifact with ... Add one mana ...)".
_REMINDER_TEXT_RE = re.compile(r"\([^)]*\)")

# An ability granted to something else, e.g. Paradise Mantle's
# 'Equipped creature has "{T}: Add one mana of any color."'
_QUOTED_TEXT_RE = re.compile(r'"[^"]*"')

# Separators people use in category names: "mana-rocks", "mana_rocks", "mana  rocks".
_NAME_SEPARATORS_RE = re.compile(r"[\s_-]+")

# Process hybrid BEFORE standard to avoid double-counting letters.
# E.g. {G/W} should give G:0.5, W:0.5 — not G:1, W:1 from separate matches.
_HYBRID_RE = re.compile(r"\{([WUBRG])/([WUBRG])\}")
_PHYREXIAN_RE = re.compile(r"\{([WUBRG])/P\}")
_STANDARD_RE = re.compile(r"\{([WUBRG])\}")

_COMMANDER_FORMATS = frozenset({"commander", "brawl", "oathbreaker", "duel"})


def count_pips(mana_cost: str | None) -> dict[str, float]:
    """Count color pips in a mana cost string, returning color -> count.

    Handles:
    - Standard pips: {W}, {U}, {B}, {R}, {G} -> 1.0 each
    - Hybrid pips: {G/W} -> 0.5 to each color
    - Phyrexian pips: {W/P} -> 0.5 (can be paid with life)
    - Generic {1}, {2}, colorless {C}, {X} -> ignored (not color pips)
    """
    if not mana_cost:
        return {}

    counts: dict[str, float] = {}

    # Strip out hybrid pips first so they don't match as standard pips.
    remaining = mana_cost

    for match in _HYBRID_RE.finditer(remaining):
        c1, c2 = match.group(1), match.group(2)
        counts[c1] = counts.get(c1, 0.0) + 0.5
        counts[c2] = counts.get(c2, 0.0) + 0.5
    remaining = _HYBRID_RE.sub("", remaining)

    # Strip phyrexian pips next (before standard, since {W/P} contains {W}).
    for match in _PHYREXIAN_RE.finditer(remaining):
        color = match.group(1)
        counts[color] = counts.get(color, 0.0) + 0.5
    remaining = _PHYREXIAN_RE.sub("", remaining)

    # Now match standard single-color pips on the remaining string.
    for match in _STANDARD_RE.finditer(remaining):
        color = match.group(1)
        counts[color] = counts.get(color, 0.0) + 1.0

    return counts


def suggest_land_count(avg_cmc: float, format_name: str) -> int:
    """Suggest total land count based on average CMC and format.

    - 60-card formats: roughly 19 + avg_cmc * 2, clamped to [20, 26]
    - Commander-style formats: roughly 33 + avg_cmc * 1.5, clamped to [33, 40]
    """
    if format_name.lower() in _COMMANDER_FORMATS:
        return max(33, min(40, round(33 + avg_cmc * 1.5)))
    return max(20, min(26, round(19 + avg_cmc * 2)))


def is_mana_rock(card: Card) -> bool:
    """True for a noncreature, nonland artifact that taps for mana.

    Covers Sol Ring, Signets, Talismans, Arcane Signet and one-shot rocks like
    Lotus Petal. Ignored: reminder text (an artifact that only makes Treasure
    tokens), quoted abilities it grants to other things (Paradise Mantle), and
    mana abilities without {T} (Ashnod's Altar, a sacrifice outlet).
    """
    types = set(re.findall(r"[a-z]+", card.type_line.lower()))
    if "artifact" not in types or types & {"creature", "land"}:
        return False
    rules_text = _REMINDER_TEXT_RE.sub("", card.oracle_text or "")
    rules_text = _QUOTED_TEXT_RE.sub("", rules_text)
    return bool(_MANA_ABILITY_RE.search(rules_text))


def is_mana_rock_request(name: str) -> bool:
    """True when a category or theme name asks for mana rocks.

    Case and separators don't matter: "Mana Rocks", "mana-rocks" and
    "mana_rocks" all count, as do the other names in ``MANA_ROCK_NAMES``.
    """
    return _NAME_SEPARATORS_RE.sub(" ", name.lower()).strip() in MANA_ROCK_NAMES
