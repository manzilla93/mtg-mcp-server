"""Tests for cheapest-printing prices in the Scryfall bulk data service.

Oracle Cards lists one printing per card; its prices are that printing's, often
none (a preview, a digital printing). ``apply_cheapest_prices`` reads every
printing from Default Cards and gives each card its cheapest paper printing's.
The Oracle and Default Cards data here is built in code: one dict per printing,
the same shape as Scryfall's card objects, trimmed to the fields that matter.
"""

from __future__ import annotations

import gzip
import json

import httpx
import pytest
import respx
from structlog.testing import capture_logs

from mtg_mcp_server.services.scryfall_bulk import (
    ScryfallBulkClient,
    ScryfallBulkDownloadError,
    ScryfallBulkError,
    _jsonl_lines,
)

_BASE_URL = "https://api.scryfall.com"
_ORACLE_URL = "https://data.scryfall.io/oracle-cards/oracle-cards-20261030090000.jsonl.gz"
_DEFAULT_URL = "https://data.scryfall.io/default-cards/default-cards-20261030090000.jsonl.gz"

SOL_RING = "6ad8011d-3471-4369-9d68-b264cc027487"
FELLWAR = "b5b9f8bd-6a5c-4e16-9ac7-7a0b3d3d1c5e"
BRAND_NEW = "0b6b5c2e-7f2c-4b65-9d2e-6a2f5c1d9e01"


def _card(name: str, oracle_id: str | None, set_code: str, **prices: str | None) -> dict:
    """One printing, as Scryfall serves it (only the fields the code reads)."""
    return {
        "object": "card",
        "id": f"{set_code}-{name}",
        "oracle_id": oracle_id,
        "name": name,
        "layout": "normal",
        "type_line": "Artifact",
        "set": set_code,
        "set_type": "expansion",
        "digital": False,
        "oversized": False,
        "border_color": "black",
        "legalities": {"commander": "legal"},
        "prices": {"usd": None, "usd_foil": None, "eur": None, **prices},
    }


def _jsonl_gz(entries: list[dict]) -> bytes:
    """Gzipped JSON Lines, like Scryfall's bulk downloads."""
    return gzip.compress(b"".join(json.dumps(e).encode() + b"\n" for e in entries))


# What Oracle Cards lists: Sol Ring's printing is an unreleased preview with no
# price; Fellwar Stone's is a pricey special printing; the new card has no
# price anywhere yet; the last card has no Oracle ID to match on.
ORACLE_CARDS = [
    _card("Sol Ring", SOL_RING, "trc"),
    _card("Fellwar Stone", FELLWAR, "sld", usd="10.75", usd_foil="20.00", eur="9.00"),
    _card("Brand New Card", BRAND_NEW, "trc"),
    _card("Mystery Card", None, "unk", usd="5.00"),
]

# Every printing, as Default Cards lists them.
DEFAULT_CARDS = [
    _card("Sol Ring", SOL_RING, "trc"),  # the listed preview: no prices
    _card("Sol Ring", SOL_RING, "c21", usd="1.23", usd_foil="2.50", eur="1.00"),
    _card("Sol Ring", SOL_RING, "cmm", usd="3.00", usd_foil="1.99", eur="2.00"),
    # Cheaper, but not a playable paper printing: none of these may set the price
    {**_card("Sol Ring", SOL_RING, "o90p", usd="0.10"), "oversized": True},
    {**_card("Sol Ring", SOL_RING, "30a", usd="0.05"), "set_type": "memorabilia"},
    {**_card("Sol Ring", SOL_RING, "wc97", usd="0.20"), "border_color": "gold"},
    {**_card("Sol Ring", SOL_RING, "prm", usd="0.01"), "digital": True},
    {**_card("Sol Ring", SOL_RING, "tc21", usd="0.02"), "layout": "token"},
    _card("Fellwar Stone", FELLWAR, "sld", usd="10.75", usd_foil="20.00", eur="9.00"),
    _card("Fellwar Stone", FELLWAR, "c18", usd="0.77"),
    _card("Fellwar Stone", FELLWAR, "drk", usd="not a price"),  # ignored
    _card("Brand New Card", BRAND_NEW, "trc"),
]


def _mock_scryfall(router: respx.MockRouter, default_cards: bytes | None = None) -> respx.Route:
    """Mock both bulk files and their metadata; return the Default Cards download route."""
    router.get(f"{_BASE_URL}/bulk-data/oracle_cards").mock(
        return_value=httpx.Response(200, json={"jsonl_download_uri": _ORACLE_URL})
    )
    router.get(_ORACLE_URL).mock(return_value=httpx.Response(200, content=_jsonl_gz(ORACLE_CARDS)))
    router.get(f"{_BASE_URL}/bulk-data/default_cards").mock(
        return_value=httpx.Response(200, json={"jsonl_download_uri": _DEFAULT_URL})
    )
    body = default_cards if default_cards is not None else _jsonl_gz(DEFAULT_CARDS)
    return router.get(_DEFAULT_URL).mock(return_value=httpx.Response(200, content=body))


async def _prices(client: ScryfallBulkClient, name: str) -> tuple[str | None, ...]:
    """(usd, usd_foil, eur) of a loaded card."""
    card = await client.get_card(name)
    assert card is not None
    return (card.prices.usd, card.prices.usd_foil, card.prices.eur)


class TestApplyCheapestPrices:
    """Cards take their cheapest paper printing's prices."""

    async def test_each_price_is_the_lowest_paper_printing(self):
        with respx.mock:
            _mock_scryfall(respx)
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                changed = await client.apply_cheapest_prices()

                # Lowest of each kind, across c21 and cmm; the cheaper oversized,
                # memorabilia, gold-bordered, digital and token printings don't count.
                assert await _prices(client, "Sol Ring") == ("1.23", "1.99", "1.00")
                # The listed special printing loses to a $0.77 reprint
                assert await _prices(client, "Fellwar Stone") == ("0.77", "20.00", "9.00")
                assert changed == 2

    async def test_cards_without_a_priced_printing_keep_their_prices(self):
        with respx.mock:
            _mock_scryfall(respx)
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                await client.apply_cheapest_prices()

                assert await _prices(client, "Brand New Card") == (None, None, None)
                # No Oracle ID to match on: left as listed
                assert await _prices(client, "Mystery Card") == ("5.00", None, None)

    async def test_budget_filters_see_the_cheapest_price(self):
        """The bug this fixes: max_price dropped Sol Ring, whose listed printing has no price."""
        with respx.mock:
            _mock_scryfall(respx)
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                assert await client.filter_cards(max_price=2.0) == []

                await client.apply_cheapest_prices()
                names = {c.name for c in await client.filter_cards(max_price=2.0)}
                assert names == {"Sol Ring", "Fellwar Stone"}

    async def test_one_download_serves_every_client(self):
        """The provider and the workflow server each run a client; prices are fetched once."""
        with respx.mock:
            download = _mock_scryfall(respx)
            async with (
                ScryfallBulkClient(base_url=_BASE_URL) as first,
                ScryfallBulkClient(base_url=_BASE_URL) as second,
            ):
                await first.ensure_loaded()
                await second.ensure_loaded()
                await first.apply_cheapest_prices()
                await second.apply_cheapest_prices()

                assert download.call_count == 1
                assert await _prices(second, "Sol Ring") == ("1.23", "1.99", "1.00")

    async def test_plain_jsonl_is_read_too(self):
        plain = b"".join(json.dumps(e).encode() + b"\n" for e in DEFAULT_CARDS)
        with respx.mock:
            _mock_scryfall(respx, default_cards=plain)
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                await client.apply_cheapest_prices()
                assert await _prices(client, "Sol Ring") == ("1.23", "1.99", "1.00")


class TestBackgroundUpdate:
    """With cheapest_prices on, every load is followed by a background update."""

    async def test_load_starts_the_update(self):
        with respx.mock:
            _mock_scryfall(respx)
            async with ScryfallBulkClient(base_url=_BASE_URL, cheapest_prices=True) as client:
                await client.ensure_loaded()
                assert client._prices_task is not None
                await client._prices_task

                assert await _prices(client, "Sol Ring") == ("1.23", "1.99", "1.00")

    async def test_off_by_default(self):
        with respx.mock:
            download = _mock_scryfall(respx)
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()

                assert client._prices_task is None
                assert not download.called
                assert await _prices(client, "Sol Ring") == (None, None, None)

    async def test_failure_keeps_listed_prices_and_is_logged(self):
        with respx.mock:
            _mock_scryfall(respx)
            respx.get(_DEFAULT_URL).mock(return_value=httpx.Response(503))
            async with ScryfallBulkClient(base_url=_BASE_URL, cheapest_prices=True) as client:
                with capture_logs() as logs:
                    await client.ensure_loaded()
                    assert client._prices_task is not None
                    await client._prices_task  # doesn't raise

                assert await _prices(client, "Fellwar Stone") == ("10.75", "20.00", "9.00")
                assert any(e["event"] == "scryfall_bulk.cheapest_prices_failed" for e in logs)

    async def test_exit_cancels_a_running_update(self):
        with respx.mock:
            _mock_scryfall(respx)
            async with ScryfallBulkClient(base_url=_BASE_URL, cheapest_prices=True) as client:
                await client.ensure_loaded()
                task = client._prices_task
            assert task is not None
            assert task.done()
            assert client._prices_task is None


class TestDownloadErrors:
    """apply_cheapest_prices raises typed errors; the background task logs them."""

    async def test_http_error(self):
        with respx.mock:
            _mock_scryfall(respx)
            respx.get(_DEFAULT_URL).mock(return_value=httpx.Response(500))
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                with pytest.raises(ScryfallBulkDownloadError, match="HTTP 500"):
                    await client.apply_cheapest_prices()

    async def test_network_error(self):
        with respx.mock:
            _mock_scryfall(respx)
            respx.get(_DEFAULT_URL).mock(side_effect=httpx.ConnectError("refused"))
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                with pytest.raises(ScryfallBulkDownloadError, match="Network error"):
                    await client.apply_cheapest_prices()

    async def test_metadata_without_a_download_link(self):
        with respx.mock:
            _mock_scryfall(respx)
            respx.get(f"{_BASE_URL}/bulk-data/default_cards").mock(
                return_value=httpx.Response(200, json={"type": "default_cards"})
            )
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                with pytest.raises(ScryfallBulkError, match="missing both"):
                    await client.apply_cheapest_prices()

    async def test_malformed_line_names_its_number(self):
        body = _jsonl_gz(DEFAULT_CARDS[:2]) + gzip.compress(b"{not json\n")
        with respx.mock:
            _mock_scryfall(respx, default_cards=body)
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                with pytest.raises(ScryfallBulkError, match="line 3"):
                    await client.apply_cheapest_prices()
                # Nothing applied from a file that failed part-way
                assert await _prices(client, "Sol Ring") == (None, None, None)

    async def test_a_file_that_isnt_json_lines(self):
        """One huge line (an old-style JSON array) stops early instead of filling memory."""
        body = b"[" + b'{"name": "x"},' * 200_000
        with respx.mock:
            _mock_scryfall(respx, default_cards=body)
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                with pytest.raises(ScryfallBulkError, match="expected JSON Lines"):
                    await client.apply_cheapest_prices()

    async def test_truncated_gzip(self):
        """A download cut short raises instead of pricing from part of the file."""
        body = _jsonl_gz(DEFAULT_CARDS)[:-20]
        with respx.mock:
            _mock_scryfall(respx, default_cards=body)
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                with pytest.raises(ScryfallBulkError, match="truncated gzip"):
                    await client.apply_cheapest_prices()

    async def test_corrupt_gzip(self):
        body = b"\x1f\x8b" + b"definitely not deflate data" * 10
        with respx.mock:
            _mock_scryfall(respx, default_cards=body)
            async with ScryfallBulkClient(base_url=_BASE_URL) as client:
                await client.ensure_loaded()
                with pytest.raises(ScryfallBulkError, match="corrupt gzip"):
                    await client.apply_cheapest_prices()


class TestJsonlLines:
    """The streaming line reader, fed chunks the way httpx might deliver them."""

    @staticmethod
    async def _read(data: bytes, chunk_size: int) -> list[bytes]:
        async def chunks():
            for start in range(0, len(data), chunk_size):
                yield data[start : start + chunk_size]

        return [line async for line in _jsonl_lines(chunks())]

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 64, 10_000])
    async def test_gzipped_lines_any_chunking(self, chunk_size: int):
        data = gzip.compress(b'{"a": 1}\n\n{"b": 2}\n{"c": 3}')
        assert await self._read(data, chunk_size) == [b'{"a": 1}', b'{"b": 2}', b'{"c": 3}']

    async def test_plain_lines(self):
        assert await self._read(b"one\ntwo\n", 3) == [b"one", b"two"]

    async def test_a_single_byte_file(self):
        assert await self._read(b"x", 1) == [b"x"]

    async def test_empty_file(self):
        assert await self._read(b"", 1) == []
