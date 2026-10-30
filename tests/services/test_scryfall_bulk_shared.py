"""Tests for shared_bulk_client: one card database per process.

The bulk provider and the workflow server each need a ScryfallBulkClient; each
loaded copy of the card data is a few hundred MB, so they share one. The
client is created for the first user and closed when the last one leaves.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mtg_mcp_server.services import scryfall_bulk
from mtg_mcp_server.services.scryfall_bulk import shared_bulk_client

SETTINGS = {"base_url": "https://api.scryfall.com", "refresh_hours": 12, "cheapest_prices": True}


class _Factory:
    """Stands in for the ScryfallBulkClient class: records every client it makes."""

    def __init__(self) -> None:
        self.made: list[tuple[AsyncMock, dict]] = []

    def __call__(self, **kwargs: object) -> AsyncMock:
        client = AsyncMock()
        self.made.append((client, kwargs))
        return client


@pytest.fixture
def factory() -> _Factory:
    return _Factory()


class TestSharedBulkClient:
    async def test_same_settings_share_one_client(self, factory: _Factory):
        async with (
            shared_bulk_client(**SETTINGS, factory=factory) as first,
            shared_bulk_client(**SETTINGS, factory=factory) as second,
        ):
            assert first is second
        assert len(factory.made) == 1
        assert factory.made[0][1] == SETTINGS

    async def test_closed_only_when_the_last_user_leaves(self, factory: _Factory):
        async with shared_bulk_client(**SETTINGS, factory=factory) as client:
            client.__aenter__.assert_awaited_once()
            async with shared_bulk_client(**SETTINGS, factory=factory):
                pass
            client.__aexit__.assert_not_awaited()
        client.__aexit__.assert_awaited_once()
        assert scryfall_bulk._shared_clients == {}

    async def test_different_settings_get_different_clients(self, factory: _Factory):
        async with (
            shared_bulk_client(**SETTINGS, factory=factory) as first,
            shared_bulk_client(**{**SETTINGS, "cheapest_prices": False}, factory=factory) as other,
        ):
            assert first is not other
        assert len(factory.made) == 2

    async def test_a_new_client_after_everyone_left(self, factory: _Factory):
        async with shared_bulk_client(**SETTINGS, factory=factory) as first:
            pass
        async with shared_bulk_client(**SETTINGS, factory=factory) as second:
            assert second is not first

    async def test_an_error_in_a_lifespan_still_releases(self, factory: _Factory):
        with pytest.raises(RuntimeError, match="boom"):
            async with shared_bulk_client(**SETTINGS, factory=factory):
                raise RuntimeError("boom")
        assert scryfall_bulk._shared_clients == {}
        factory.made[0][0].__aexit__.assert_awaited_once()

    async def test_a_client_that_fails_to_start_isnt_kept(self, factory: _Factory):
        def failing(**kwargs: object) -> AsyncMock:
            client = factory(**kwargs)
            client.__aenter__.side_effect = OSError("no network")
            return client

        with pytest.raises(OSError, match="no network"):
            async with shared_bulk_client(**SETTINGS, factory=failing):
                pass
        assert scryfall_bulk._shared_clients == {}

    async def test_real_client_by_default(self):
        async with shared_bulk_client(**SETTINGS) as client:
            assert isinstance(client, scryfall_bulk.ScryfallBulkClient)
            assert client._cheapest_prices is True
