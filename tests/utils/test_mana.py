"""Tests for mana pip counting and land suggestion utilities."""

from __future__ import annotations

import pytest

from mtg_mcp_server.types import Card
from mtg_mcp_server.utils.mana import (
    count_pips,
    is_mana_rock,
    is_mana_rock_request,
    suggest_land_count,
)

# ---------------------------------------------------------------------------
# count_pips — standard pips
# ---------------------------------------------------------------------------


class TestCountPipsStandard:
    """Standard single-color pips like {W}, {U}, {B}, {R}, {G}."""

    def test_single_pip(self) -> None:
        assert count_pips("{G}") == {"G": 1.0}

    def test_two_same_color(self) -> None:
        assert count_pips("{U}{U}") == {"U": 2.0}

    def test_multiple_different_colors(self) -> None:
        result = count_pips("{3}{B}{G}{U}")
        assert result == {"B": 1.0, "G": 1.0, "U": 1.0}

    def test_all_five_colors(self) -> None:
        result = count_pips("{W}{U}{B}{R}{G}")
        assert result == {"W": 1.0, "U": 1.0, "B": 1.0, "R": 1.0, "G": 1.0}


# ---------------------------------------------------------------------------
# count_pips — hybrid pips
# ---------------------------------------------------------------------------


class TestCountPipsHybrid:
    """Hybrid mana pips like {G/W} count 0.5 to each color."""

    def test_single_hybrid(self) -> None:
        result = count_pips("{G/W}")
        assert result == {"G": 0.5, "W": 0.5}

    def test_hybrid_plus_standard(self) -> None:
        # {G/W}{G} should give G: 1.5, W: 0.5
        result = count_pips("{G/W}{G}")
        assert result == {"G": 1.5, "W": 0.5}

    def test_multiple_hybrids(self) -> None:
        result = count_pips("{U/B}{U/B}")
        assert result == {"U": 1.0, "B": 1.0}

    def test_hybrid_not_double_counted(self) -> None:
        # {R/G} should not also match as {R} and {G} individually
        result = count_pips("{R/G}")
        assert result == {"R": 0.5, "G": 0.5}


# ---------------------------------------------------------------------------
# count_pips — phyrexian pips
# ---------------------------------------------------------------------------


class TestCountPipsPhyrexian:
    """Phyrexian mana pips like {W/P} count 0.5 (can be paid with life)."""

    def test_single_phyrexian(self) -> None:
        result = count_pips("{W/P}")
        assert result == {"W": 0.5}

    def test_phyrexian_plus_standard(self) -> None:
        result = count_pips("{B/P}{B}")
        assert result == {"B": 1.5}

    def test_multiple_phyrexian(self) -> None:
        result = count_pips("{R/P}{R/P}")
        assert result == {"R": 1.0}


# ---------------------------------------------------------------------------
# count_pips — generic / colorless / X / edge cases
# ---------------------------------------------------------------------------


class TestCountPipsIgnored:
    """Generic, colorless, and X mana are not color pips."""

    def test_generic_ignored(self) -> None:
        result = count_pips("{2}{U}{B}")
        assert result == {"U": 1.0, "B": 1.0}

    def test_colorless_ignored(self) -> None:
        assert count_pips("{C}") == {}

    def test_x_ignored(self) -> None:
        assert count_pips("{X}{R}{R}") == {"R": 2.0}

    def test_none_returns_empty(self) -> None:
        assert count_pips(None) == {}

    def test_empty_string_returns_empty(self) -> None:
        assert count_pips("") == {}

    def test_only_generic(self) -> None:
        assert count_pips("{4}") == {}


# ---------------------------------------------------------------------------
# count_pips — mixed cases
# ---------------------------------------------------------------------------


class TestCountPipsMixed:
    """Combinations of standard, hybrid, and phyrexian pips."""

    def test_hybrid_and_phyrexian(self) -> None:
        # {G/W}{B/P} = G: 0.5, W: 0.5, B: 0.5
        result = count_pips("{G/W}{B/P}")
        assert result == {"G": 0.5, "W": 0.5, "B": 0.5}

    def test_realistic_muldrotha(self) -> None:
        # Muldrotha: {3}{B}{G}{U}
        result = count_pips("{3}{B}{G}{U}")
        assert result == {"B": 1.0, "G": 1.0, "U": 1.0}


# ---------------------------------------------------------------------------
# suggest_land_count — 60-card formats
# ---------------------------------------------------------------------------


class TestSuggestLandCount60Card:
    """60-card format land suggestions."""

    def test_low_cmc_aggro(self) -> None:
        # avg_cmc ~1.5 → round(19 + 1.5 * 2) = round(22) = 22
        assert suggest_land_count(1.5, "modern") == 22

    def test_midrange(self) -> None:
        # avg_cmc ~3.0 → round(19 + 3.0 * 2) = round(25) = 25
        assert suggest_land_count(3.0, "standard") == 25

    def test_very_low_cmc_clamps_to_20(self) -> None:
        # avg_cmc ~0.0 → round(19 + 0) = 19, clamped to 20
        assert suggest_land_count(0.0, "modern") == 20

    def test_very_high_cmc_clamps_to_26(self) -> None:
        # avg_cmc ~5.0 → round(19 + 10) = 29, clamped to 26
        assert suggest_land_count(5.0, "standard") == 26

    def test_unknown_format_uses_60_card(self) -> None:
        assert suggest_land_count(2.0, "invented_format") == 23


# ---------------------------------------------------------------------------
# suggest_land_count — Commander formats
# ---------------------------------------------------------------------------


class TestSuggestLandCountCommander:
    """Commander-style format land suggestions."""

    def test_commander_low_cmc(self) -> None:
        # avg_cmc ~2.0 → round(33 + 2.0 * 1.5) = round(36) = 36
        assert suggest_land_count(2.0, "commander") == 36

    def test_commander_high_cmc(self) -> None:
        # avg_cmc ~4.0 → round(33 + 6.0) = 39
        assert suggest_land_count(4.0, "commander") == 39

    def test_commander_very_low_clamps_to_33(self) -> None:
        # avg_cmc ~0.0 → round(33 + 0) = 33
        assert suggest_land_count(0.0, "commander") == 33

    def test_commander_very_high_clamps_to_40(self) -> None:
        # avg_cmc ~6.0 → round(33 + 9) = 42, clamped to 40
        assert suggest_land_count(6.0, "commander") == 40

    def test_brawl_uses_commander_formula(self) -> None:
        assert suggest_land_count(2.0, "brawl") == 36

    def test_oathbreaker_uses_commander_formula(self) -> None:
        assert suggest_land_count(2.0, "oathbreaker") == 36

    def test_duel_uses_commander_formula(self) -> None:
        assert suggest_land_count(2.0, "duel") == 36

    def test_case_insensitive(self) -> None:
        assert suggest_land_count(2.0, "Commander") == 36


# ---------------------------------------------------------------------------
# is_mana_rock
# ---------------------------------------------------------------------------


def _card(name: str, type_line: str, oracle_text: str) -> Card:
    """Minimal card with just the fields is_mana_rock reads."""
    return Card(id=f"test-{name}", name=name, type_line=type_line, oracle_text=oracle_text)


class TestIsManaRock:
    """Noncreature, nonland artifacts with a mana ability."""

    @pytest.mark.parametrize(
        ("name", "type_line", "oracle_text"),
        [
            ("Sol Ring", "Artifact", "{T}: Add {C}{C}."),
            (
                "Arcane Signet",
                "Artifact",
                "{T}: Add one mana of any color in your commander's color identity.",
            ),
            ("Dimir Signet", "Artifact", "{1}, {T}: Add {U}{B}."),
            (
                "Lotus Petal",
                "Artifact",
                "{T}, Sacrifice this artifact: Add one mana of any color.",
            ),
            (
                "Cultivator's Caravan",
                "Artifact \u2014 Vehicle",
                "{T}: Add one mana of any color.\nCrew 3",
            ),
            (
                "Talisman of Dominance",
                "Artifact",
                "{T}: Add {C}.\n{T}: Add {U} or {B}. This artifact deals 1 damage to you.",
            ),
            # Grants a mana ability to lands and has its own
            (
                "Chromatic Lantern",
                "Artifact",
                'Lands you control have "{T}: Add one mana of any color."\n'
                "{T}: Add one mana of any color.",
            ),
            # The ability says something before it adds the mana (Oracle text, 2026-10-30)
            (
                "Nyx Lotus",
                "Legendary Artifact",
                "Nyx Lotus enters tapped.\n{T}: Choose a color. Add an amount of mana of that "
                "color equal to your devotion to that color. (Your devotion to a color is the "
                "number of mana symbols of that color in the mana costs of permanents you control.)",
            ),
            (
                "Astral Cornucopia",
                "Artifact",
                "This artifact enters with X charge counters on it.\n{T}: Choose a color. "
                "Add one mana of that color for each charge counter on this artifact.",
            ),
            (
                "Charmed Pendant",
                "Artifact",
                "{T}, Mill a card: For each colored mana symbol in the milled card's mana cost, "
                "add one mana of that color. (Activate only as an instant. For example, if the "
                "card's mana cost is {3}{U}{U}{B}, you add {U}{U}{B}.)",
            ),
            (
                "Empowered Autogenerator",
                "Artifact",
                "This artifact enters tapped.\n{T}: Put a charge counter on this artifact. "
                "Add X mana of any one color, where X is the number of charge counters on "
                "this artifact.",
            ),
        ],
    )
    def test_mana_rocks(self, name: str, type_line: str, oracle_text: str) -> None:
        assert is_mana_rock(_card(name, type_line, oracle_text))

    @pytest.mark.parametrize(
        ("name", "type_line", "oracle_text"),
        [
            # Mana creature, not a rock
            ("Palladium Myr", "Artifact Creature \u2014 Myr", "{T}: Add {C}{C}."),
            # Artifact land
            ("Ancient Den", "Artifact Land", "({T}: Add {W}.)"),
            # Only the Treasure reminder text mentions mana
            (
                "Goldvein Pick",
                "Artifact \u2014 Equipment",
                "Equipped creature gets +1/+1.\nWhenever equipped creature deals combat "
                "damage to a player, create a Treasure token. (It's an artifact with "
                '"{T}, Sacrifice this token: Add one mana of any color.")\nEquip {1}',
            ),
            # Artifact without a mana ability
            (
                "Swiftfoot Boots",
                "Artifact \u2014 Equipment",
                "Equipped creature has hexproof and haste.",
            ),
            # Not an artifact
            ("Smothering Tithe", "Enchantment", "Whenever an opponent draws a card..."),
            # Sacrifice outlets: the mana ability doesn't tap the artifact
            ("Ashnod's Altar", "Artifact", "Sacrifice a creature: Add {C}{C}."),
            ("Phyrexian Altar", "Artifact", "Sacrifice a creature: Add one mana of any color."),
            # The mana ability is granted to the equipped creature, in quotes
            (
                "Paradise Mantle",
                "Artifact — Equipment",
                'Equipped creature has "{T}: Add one mana of any color."\nEquip {1}',
            ),
            # A {T} ability that adds something other than mana (a made-up card)
            (
                "Ante Amulet",
                "Artifact",
                "{T}, Sacrifice this artifact: Target opponent may add the top card of their "
                "library to the ante.",
            ),
            # {T} in one ability, "add" in another (a made-up card)
            (
                "Two Abilities",
                "Artifact",
                "{T}: Untap target land.\nSacrifice a creature: Add {C}.",
            ),
        ],
    )
    def test_not_mana_rocks(self, name: str, type_line: str, oracle_text: str) -> None:
        assert not is_mana_rock(_card(name, type_line, oracle_text))


class TestIsManaRockRequest:
    """Category and theme names that ask for mana rocks."""

    @pytest.mark.parametrize(
        "name", ["mana rocks", "Mana Rocks", "mana-rocks", "mana_rocks", "rocks", "mana artifacts"]
    )
    def test_requests(self, name: str) -> None:
        assert is_mana_rock_request(name)

    @pytest.mark.parametrize("name", ["artifacts", "ramp", "mana", "rock lobster"])
    def test_other_names(self, name: str) -> None:
        assert not is_mana_rock_request(name)
