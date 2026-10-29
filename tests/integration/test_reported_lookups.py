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
        assert "702.19 Trample" in text
        assert "State-Based Actions" not in text
