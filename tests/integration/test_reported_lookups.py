"""Regression tests for questions a deployed server answered wrongly.

Each test replays the question through the full MCP pipeline (client ->
orchestrator -> provider -> service) with all HTTP mocked by fixture data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastmcp import Client


pytestmark = pytest.mark.integration


class TestRulesLookupByNumber:
    """Asking "what does rule 702.19 say?" returned only the title "Trample"."""

    async def test_keyword_rule_includes_its_subrules(self, mcp_client: Client):
        """702.19 is a title; the rules themselves are its lettered subrules."""
        result = await mcp_client.call_tool("rules_lookup", {"query": "702.19"})
        text = result.content[0].text
        assert "702.19a" in text
        assert "702.19b" in text
        assert "Keyword Abilities" in text

    async def test_subrule_includes_its_parent(self, mcp_client: Client):
        """702.19b shows that it belongs to 702.19 Trample, without leaked headings."""
        result = await mcp_client.call_tool("rules_lookup", {"query": "702.19b"})
        text = result.content[0].text
        assert "*7. Additional Rules > 702. Keyword Abilities > 702.19. Trample*" in text
        assert "State-Based Actions" not in text

    async def test_concise_lookup_keeps_full_subrule_text(self, mcp_client: Client):
        """Concise mode cut 702.19b to its first sentence, losing where excess damage goes."""
        result = await mcp_client.call_tool(
            "rules_lookup", {"query": "702.19", "response_format": "concise"}
        )
        text = result.content[0].text
        assert "any excess damage is assigned as its controller chooses" in text


class TestEsperManaRocks:
    """Asking for the best Esper mana rocks returned spells, not rocks."""

    async def test_staples_with_mana_artifacts_category(self, mcp_client: Client):
        """The category a model actually sent, 'mana artifacts', now means mana rocks."""
        result = await mcp_client.call_tool(
            "color_identity_staples",
            {"color_identity": "esper", "category": "mana artifacts"},
        )
        text = result.content[0].text
        assert "Sol Ring" in text
        assert "Arcane Signet" in text
        assert "Counterspell" not in text  # an instant
        assert "Aetherflux Reservoir" not in text  # an artifact with no mana ability

    async def test_theme_search_mana_rocks(self, mcp_client: Client):
        """theme_search('mana rocks') found nothing; it now returns the rocks, most played first."""
        result = await mcp_client.call_tool(
            "theme_search",
            {"theme": "mana rocks", "color_identity": "esper", "format": "commander"},
        )
        cards = [c["name"] for c in result.structured_content["cards"]]
        assert cards[:2] == ["Sol Ring", "Arcane Signet"]
        assert "Aetherflux Reservoir" not in cards
