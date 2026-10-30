"""Integration tests for Scryfall bulk data through the full MCP pipeline.

Tests exercise bulk_card_lookup, bulk_card_search, and bulk card-data resources
via the orchestrator with fixture-mocked HTTP backends.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastmcp import Client


pytestmark = pytest.mark.integration


class TestBulkCardLookup:
    """Bulk card lookup through the orchestrator."""

    async def test_sol_ring_returns_artifact_not_minigame(self, mcp_client: Client):
        """bulk_card_lookup returns the real Sol Ring (Artifact), not the minigame version."""
        result = await mcp_client.call_tool("bulk_card_lookup", {"name": "Sol Ring"})
        text = result.content[0].text
        assert "Artifact" in text
        assert "acmm" not in text

    async def test_sol_ring_has_prices(self, mcp_client: Client):
        """bulk_card_lookup output includes price data."""
        result = await mcp_client.call_tool("bulk_card_lookup", {"name": "Sol Ring"})
        text = result.content[0].text
        assert "$1.50" in text

    async def test_sol_ring_has_legalities(self, mcp_client: Client):
        """bulk_card_lookup output includes legality information."""
        result = await mcp_client.call_tool("bulk_card_lookup", {"name": "Sol Ring"})
        text = result.content[0].text
        assert "commander" in text.lower()
        assert "legal" in text.lower()

    async def test_sol_ring_has_edhrec_rank(self, mcp_client: Client):
        """bulk_card_lookup output includes EDHREC rank."""
        result = await mcp_client.call_tool("bulk_card_lookup", {"name": "Sol Ring"})
        text = result.content[0].text
        assert "EDHREC Rank" in text

    async def test_dfc_lookup_by_front_face(self, mcp_client: Client):
        """bulk_card_lookup finds a DFC by front-face name and includes Creature type."""
        result = await mcp_client.call_tool("bulk_card_lookup", {"name": "Delver of Secrets"})
        text = result.content[0].text
        assert "Creature" in text

    async def test_not_found_returns_error(self, mcp_client: Client):
        """bulk_card_lookup returns an error for a nonexistent card."""
        result = await mcp_client.call_tool(
            "bulk_card_lookup", {"name": "Nonexistent Card"}, raise_on_error=False
        )
        assert result.is_error


class TestBulkCardSearch:
    """Bulk card search through the orchestrator."""

    async def test_search_by_name(self, mcp_client: Client):
        """bulk_card_search by name finds Sol Ring."""
        result = await mcp_client.call_tool("bulk_card_search", {"query": "Sol"})
        text = result.content[0].text
        assert "Sol Ring" in text

    async def test_search_excludes_nonplayable(self, mcp_client: Client):
        """bulk_card_search for 'Sol Ring' excludes the minigame version."""
        result = await mcp_client.call_tool("bulk_card_search", {"query": "Sol Ring"})
        text = result.content[0].text
        assert "acmm" not in text

    async def test_search_by_type_excludes_tokens(self, mcp_client: Client):
        """bulk_card_search by type for Creature does not return token entries."""
        result = await mcp_client.call_tool(
            "bulk_card_search", {"query": "Creature", "search_field": "type"}
        )
        text = result.content[0].text
        assert "Token" not in text

    async def test_search_by_text(self, mcp_client: Client):
        """bulk_card_search by oracle text for 'Add' finds Sol Ring."""
        result = await mcp_client.call_tool(
            "bulk_card_search", {"query": "Add", "search_field": "text"}
        )
        text = result.content[0].text
        assert "Sol Ring" in text


class TestBulkResource:
    """Bulk card-data resource through the orchestrator."""

    async def test_resource_returns_correct_card(self, mcp_client: Client):
        """mtg://bulk/card-data/Sol Ring returns valid JSON with correct card data."""
        result = await mcp_client.read_resource("mtg://bulk/card-data/Sol Ring")
        data = json.loads(result[0].text)
        assert data["name"] == "Sol Ring"
        assert data["layout"] == "normal"

    async def test_resource_card_not_found(self, mcp_client: Client):
        """mtg://bulk/card-data/nonexistent returns error JSON."""
        result = await mcp_client.read_resource("mtg://bulk/card-data/nonexistent")
        data = json.loads(result[0].text)
        assert "error" in data
        assert "not found" in data["error"].lower()


class TestCheapestPrices:
    """With MTG_MCP_BULK_CHEAPEST_PRICES on, cards take their cheapest paper printing's price."""

    async def test_price_comparison_uses_the_cheapest_printing(
        self, cheapest_prices: None, mcp_client: Client
    ):
        """The listed printings cost $1.50 and $2.50; Default Cards has $0.99 and $0.75 ones."""
        from mtg_mcp_server.workflows import server as workflows

        cards = ["Sol Ring", "Lightning Bolt", "Counterspell"]
        # The first call loads the card data, which starts the price update
        await mcp_client.call_tool("price_comparison", {"cards": cards})
        assert workflows._bulk is not None
        assert workflows._bulk._prices_task is not None
        await workflows._bulk._prices_task

        result = await mcp_client.call_tool("price_comparison", {"cards": cards})
        assert result.structured_content is not None
        usd = {c["name"]: c["usd"] for c in result.structured_content["cards"]}
        # Not the oversized $0.05 Sol Ring or the memorabilia $0.10 Bolt
        assert usd == {"Sol Ring": "0.99", "Lightning Bolt": "0.75", "Counterspell": "1.80"}


class TestOneCardDatabase:
    """The bulk provider and the workflow server load the card data once between them."""

    async def test_provider_and_workflows_share_one_client(self, mcp_client: Client):
        from mtg_mcp_server.providers import scryfall_bulk as provider
        from mtg_mcp_server.workflows import server as workflows

        assert provider._client is not None
        assert provider._client is workflows._bulk

    async def test_one_download_serves_both(self, mcp_client: Client):
        """A bulk tool and a workflow tool: the card data is fetched once."""
        import respx

        await mcp_client.call_tool("bulk_card_lookup", {"name": "Sol Ring"})
        await mcp_client.call_tool("price_comparison", {"cards": ["Sol Ring", "Lightning Bolt"]})

        downloads = [call for call in respx.calls if "oracle-cards" in str(call.request.url)]
        assert len(downloads) == 1
