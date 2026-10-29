"""Tests for rules-text similarity (TF-IDF features + cosine)."""

from __future__ import annotations

import gc
import weakref

import pytest

from mtg_mcp_server.types import Card
from mtg_mcp_server.utils.similarity import SimilarityIndex, card_features


def _card(name: str, type_line: str, oracle_text: str, edhrec_rank: int | None = None) -> Card:
    """Minimal card with the fields the similarity module reads."""
    return Card(
        id=f"test-{name}",
        name=name,
        type_line=type_line,
        oracle_text=oracle_text,
        edhrec_rank=edhrec_rank,
    )


# Oracle text as served by Scryfall (Oct 2026), including current templating
# such as "this enchantment" in place of the card's name.
RHYSTIC_STUDY = _card(
    "Rhystic Study",
    "Enchantment",
    "Whenever an opponent casts a spell, you may draw a card unless that player pays {1}.",
    46,
)
MYSTIC_REMORA = _card(
    "Mystic Remora",
    "Enchantment",
    "Cumulative upkeep {1} (At the beginning of your upkeep, put an age counter on this "
    "permanent, then sacrifice it unless you pay its upkeep cost for each age counter on it.)\n"
    "Whenever an opponent casts a noncreature spell, you may draw a card unless that player "
    "pays {4}.",
    101,
)
ESPER_SENTINEL = _card(
    "Esper Sentinel",
    "Artifact Creature — Human Soldier",
    "Whenever an opponent casts their first noncreature spell each turn, draw a card unless "
    "that player pays {X}, where X is this creature's power.",
    79,
)
SMOTHERING_TITHE = _card(
    "Smothering Tithe",
    "Enchantment",
    "Whenever an opponent draws a card, that player may pay {2}. If the player doesn't, you "
    "create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one "
    'mana of any color.")',
    67,
)
SOUL_BARRIER = _card(
    "Soul Barrier",
    "Enchantment",
    "Whenever an opponent casts a creature spell, this enchantment deals 2 damage to that "
    "player unless they pay {2}.",
    18054,
)
YAWGMOTHS_EDICT = _card(
    "Yawgmoth's Edict",
    "Enchantment",
    "Whenever an opponent casts a white spell, that player loses 1 life and you gain 1 life.",
    26121,
)

POOL = [
    RHYSTIC_STUDY,
    MYSTIC_REMORA,
    ESPER_SENTINEL,
    SMOTHERING_TITHE,
    SOUL_BARRIER,
    YAWGMOTHS_EDICT,
    _card(
        "Aether Barrier",
        "Enchantment",
        "Whenever a player casts a creature spell, that player sacrifices a permanent of their "
        "choice unless they pay {1}.",
        14556,
    ),
    _card(
        "Shadow in the Warp",
        "Enchantment",
        "The first creature spell you cast each turn costs {2} less to cast.\nWhenever an "
        "opponent casts their first noncreature spell each turn, this enchantment deals 2 "
        "damage to that player.",
        3620,
    ),
    _card(
        "Nether Void",
        "World Enchantment",
        "Whenever a player casts a spell, counter it unless that player pays {3}.",
        16351,
    ),
    _card(
        "Painful Quandary",
        "Enchantment",
        "Whenever an opponent casts a spell, that player loses 5 life unless they discard a card.",
        1195,
    ),
    _card(
        "Consecrated Sphinx",
        "Creature — Sphinx",
        "Flying\nWhenever an opponent draws a card, you may draw two cards.",
        655,
    ),
    _card(
        "Trouble in Pairs",
        "Enchantment",
        "If an opponent would begin an extra turn, that player skips that turn instead.\n"
        "Whenever an opponent attacks you with two or more creatures, draws their second card "
        "each turn, or casts their second spell each turn, you draw a card.",
        620,
    ),
    # Unrelated cards, so word weights don't come from lookalikes alone
    _card("Lightning Bolt", "Instant", "Lightning Bolt deals 3 damage to any target.", 50),
    _card("Counterspell", "Instant", "Counter target spell.", 25),
    _card("Sol Ring", "Artifact", "{T}: Add {C}{C}.", 1),
    _card(
        "Arcane Signet",
        "Artifact",
        "{T}: Add one mana of any color in your commander's color identity.",
        3,
    ),
    _card("Llanowar Elves", "Creature — Elf Druid", "{T}: Add {G}.", 63),
    _card(
        "Birds of Paradise",
        "Creature — Bird",
        "Flying\n{T}: Add one mana of any color.",
        120,
    ),
    _card(
        "Swords to Plowshares",
        "Instant",
        "Exile target creature. Its controller gains life equal to its power.",
        11,
    ),
    _card("Wrath of God", "Sorcery", "Destroy all creatures. They can't be regenerated.", 603),
    _card(
        "Demonic Tutor",
        "Sorcery",
        "Search your library for a card, put that card into your hand, then shuffle.",
        57,
    ),
    _card(
        "Phyrexian Arena",
        "Enchantment",
        "At the beginning of your upkeep, you draw a card and you lose 1 life.",
        300,
    ),
    _card("Grizzly Bears", "Creature — Bear", "", 20000),
]


# ---------------------------------------------------------------------------
# card_features
# ---------------------------------------------------------------------------


class TestCardFeatures:
    """Templating differences don't change what a card's text says it does."""

    def test_own_name_and_this_object_read_the_same(self) -> None:
        """Old templating (card name) and new ("this enchantment") give equal features."""
        old = _card("Soul Barrier", "Enchantment", "Soul Barrier deals 2 damage to that player.")
        new = _card(
            "Soul Barrier", "Enchantment", "This enchantment deals 2 damage to that player."
        )
        assert card_features(old) == card_features(new)
        assert "cardname deal" in card_features(old)

    def test_short_legendary_name_is_a_self_reference(self) -> None:
        """'Korvold' in Korvold, Fae-Cursed King's text refers to the card itself."""
        korvold = _card(
            "Korvold, Fae-Cursed King",
            "Legendary Creature — Dragon Noble",
            "Whenever Korvold enters or attacks, sacrifice another permanent.",
        )
        assert "cardname enter" in card_features(korvold)

    def test_name_inside_another_word_is_left_alone(self) -> None:
        """A short name like 'Ice' must not rewrite 'twice'."""
        features = card_features(_card("Ice", "Instant", "Ice deals 1 damage twice."))
        assert "cardname" in features
        assert "twice" in features

    def test_reminder_text_is_dropped(self) -> None:
        """Smothering Tithe's Treasure reminder text isn't part of what it does."""
        features = card_features(SMOTHERING_TITHE)
        assert "create treasure" in features
        assert "sacrifice" not in features

    def test_generic_costs_and_mana_are_generalised(self) -> None:
        """'pays {1}' and 'pays {4}' match; mana symbols become {m}."""
        assert "pay {n}" in card_features(RHYSTIC_STUDY)
        assert "pay {n}" in card_features(MYSTIC_REMORA)
        assert {"{t} add", "add {m}"} <= card_features(_card("Rock", "Artifact", "{T}: Add {C}."))

    def test_filler_words_and_numbers_are_dropped(self) -> None:
        """'draw a card' and 'draw two cards' share the pair 'draw card'."""
        # Names must not appear in the text, or they'd read as self-references.
        one = card_features(_card("Alpha", "Sorcery", "Draw a card."))
        two = card_features(_card("Beta", "Sorcery", "Draw two cards."))
        assert "draw card" in one
        assert "draw card" in two

    def test_type_line_words_are_features(self) -> None:
        """Types and subtypes become prefixed features."""
        assert {"type:artifact", "type:creature", "type:human", "type:soldier"} <= card_features(
            ESPER_SENTINEL
        )


# ---------------------------------------------------------------------------
# SimilarityIndex
# ---------------------------------------------------------------------------


class TestSimilarityIndex:
    """Ranking by what cards do, not by shared boilerplate."""

    @pytest.fixture(scope="class")
    def index(self) -> SimilarityIndex:
        return SimilarityIndex.build(POOL)

    @staticmethod
    def _names(results: list[tuple[float, Card]]) -> list[str]:
        return [card.name for _, card in results]

    def test_rhystic_study_finds_mystic_remora_first(self, index: SimilarityIndex) -> None:
        """The closest functional match ranks first (it tied for 3rd before)."""
        assert self._names(index.rank(RHYSTIC_STUDY, POOL, limit=3))[0] == "Mystic Remora"

    def test_same_effect_beats_shared_boilerplate(self, index: SimilarityIndex) -> None:
        """'Draw a card unless that player pays' outranks 'whenever an opponent casts' alone."""
        ranked = self._names(index.rank(RHYSTIC_STUDY, POOL, limit=len(POOL)))
        assert ranked.index("Esper Sentinel") < ranked.index("Soul Barrier")
        assert ranked.index("Esper Sentinel") < ranked.index("Yawgmoth's Edict")

    def test_cards_sharing_nothing_are_not_returned(self, index: SimilarityIndex) -> None:
        """Sol Ring shares no feature with Rhystic Study."""
        assert "Sol Ring" not in self._names(index.rank(RHYSTIC_STUDY, POOL, limit=len(POOL)))

    def test_source_card_is_skipped(self, index: SimilarityIndex) -> None:
        assert "Rhystic Study" not in self._names(index.rank(RHYSTIC_STUDY, POOL, limit=len(POOL)))

    def test_scores_are_between_zero_and_one(self, index: SimilarityIndex) -> None:
        for score, _ in index.rank(RHYSTIC_STUDY, POOL, limit=len(POOL)):
            assert 0.0 < score <= 1.0

    def test_keep_filters_candidates(self, index: SimilarityIndex) -> None:
        results = index.rank(RHYSTIC_STUDY, POOL, limit=5, keep=lambda c: c.name != "Mystic Remora")
        assert "Mystic Remora" not in self._names(results)

    def test_identical_text_scores_one(self) -> None:
        """A functional reprint under another name is a perfect match."""
        twin = _card("Rhystic Twin", "Enchantment", RHYSTIC_STUDY.oracle_text or "")
        pool = [*POOL, twin]
        score, card = SimilarityIndex.build(pool).rank(RHYSTIC_STUDY, pool, limit=1)[0]
        assert card.name == "Rhystic Twin"
        assert score == pytest.approx(1.0)

    def test_ties_go_to_the_more_played_card(self) -> None:
        """Equal scores are ordered by EDHREC rank."""
        obscure = _card("Obscure Copy", "Enchantment", "Draw a card.", 20000)
        popular = _card("Popular Copy", "Enchantment", "Draw a card.", 5)
        source = _card("Source", "Enchantment", "Draw a card.", 100)
        pool = [obscure, popular, source, *POOL]
        results = SimilarityIndex.build(pool).rank(source, pool, limit=2)
        assert self._names(results) == ["Popular Copy", "Obscure Copy"]

    def test_source_outside_the_pool(self, index: SimilarityIndex) -> None:
        """A source card that isn't in the pool can still be compared."""
        outsider = _card(
            "Outsider",
            "Enchantment",
            "Whenever an opponent casts a spell, you may draw a card unless that player pays {2}.",
        )
        assert self._names(index.rank(outsider, POOL, limit=2))[0] == "Rhystic Study"

    def test_rank_needs_the_pool_it_was_built_from(self, index: SimilarityIndex) -> None:
        """A pool of a different size is rejected rather than mis-ranked."""
        with pytest.raises(ValueError, match="built from"):
            index.rank(RHYSTIC_STUDY, POOL[:-1], limit=1)

    def test_index_does_not_keep_cards_alive(self) -> None:
        """A cached index must not pin an old bulk snapshot in memory after a refresh."""
        pool = [_card(f"Card {i}", "Instant", "Destroy target creature.") for i in range(5)]
        first_card = weakref.ref(pool[0])
        index = SimilarityIndex.build(pool)
        del pool
        gc.collect()
        assert first_card() is None
        assert len(index.features) == 5
