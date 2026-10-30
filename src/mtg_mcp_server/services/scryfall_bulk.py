"""Scryfall bulk card data service — lazy download and in-memory card cache.

Download Scryfall's Oracle Cards bulk data once and keep it in memory for O(1)
card lookups and fast substring searches. Refresh automatically when the data
becomes stale (default 12h). On refresh failure with existing data, serve stale
data rather than propagating the error.

Unlike :class:`BaseClient`, this is a standalone service managing its own HTTP
downloads. It does not use rate limiting or retries — the bulk-data endpoint is
a single lightweight request and the bulk download is a large file fetch, neither
benefiting from the per-request rate-limit pattern that ``BaseClient`` provides.

Returns :class:`~mtg_mcp_server.types.Card` objects with prices, legalities,
EDHREC rank, and image URIs. Checks the ``/bulk-data/oracle_cards`` metadata
endpoint before download, supports ETag-based conditional downloads, uses
``asyncio.Lock`` to prevent duplicate concurrent downloads, and runs a
background refresh loop via ``asyncio.create_task()``.

Oracle Cards lists one printing per card, and its prices are that printing's:
often a preview with no price yet, or a digital printing with none. With
``cheapest_prices`` on, each load is followed by a background pass over
Scryfall's Default Cards file (every printing) that gives each card its
cheapest paper printing's prices.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import random
import time
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Self

import httpx
import structlog
from cachetools import TTLCache
from pydantic import ValidationError

from mtg_mcp_server.services.base import DEFAULT_USER_AGENT, ServiceError
from mtg_mcp_server.services.cache import async_cached
from mtg_mcp_server.types import Card, CardPrices

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Hashable
    from typing import Any as JSONValue

__all__ = [
    "ScryfallBulkClient",
    "ScryfallBulkDownloadError",
    "ScryfallBulkError",
    "shared_bulk_client",
]

log = structlog.get_logger(service="ScryfallBulkClient")

# Scryfall Oracle Cards includes non-playable entries (minigames, art series,
# tokens, emblems) that can share names with real cards. Filter these out during
# parsing. Uses a deny-list so new playable layouts are included by default.
_EXCLUDED_LAYOUTS: frozenset[str] = frozenset(
    {
        "art_series",
        "double_faced_token",
        "emblem",
        "minigame",
        "placeholder",
        "token",
    }
)

# The prices a card can take from its other printings.
_PRICE_FIELDS = ("usd", "usd_foil", "eur")

# Printings that don't set a card's price: they aren't the playable card
# (oversized), or aren't legal in sanctioned play (memorabilia such as 30th
# Anniversary Edition and gold-bordered World Championship decks). Digital
# printings are skipped too; they have no paper price.
_NON_PLAYABLE_SET_TYPES = frozenset({"memorabilia"})

# One Default Cards line is one printing, a few kilobytes. This much data with
# no line break means the file isn't JSON Lines: stop instead of buffering it.
_MAX_LINE_BYTES = 1_000_000

_GZIP_MAGIC = b"\x1f\x8b"


class CheapestPrices(NamedTuple):
    """The lowest price of each kind across a card's paper printings."""

    usd: str | None
    usd_foil: str | None
    eur: str | None


class ScryfallBulkError(ServiceError):
    """Base exception for Scryfall bulk data service errors.

    Always passes ``status_code=None`` since bulk data operations don't map
    cleanly to a single HTTP status.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None)


class ScryfallBulkDownloadError(ScryfallBulkError):
    """Error downloading Scryfall bulk data (network or HTTP failure)."""


class ScryfallBulkClient:
    """Manages Scryfall Oracle Cards bulk data with lazy download and refresh.

    Downloads the Oracle Cards JSON on first access and caches parsed
    :class:`Card` models in memory. Refreshes automatically when
    ``refresh_hours`` has elapsed, using ETag-based conditional downloads
    to avoid re-parsing unchanged data.

    Use as an async context manager::

        async with ScryfallBulkClient(base_url=...) as client:
            card = await client.get_card("Sol Ring")
    """

    # Cheapest paper prices by Oracle ID, read from Default Cards. Class-level,
    # so every client in the process shares one download. Scryfall updates
    # prices daily.
    _printing_prices_cache: TTLCache = TTLCache(maxsize=1, ttl=6 * 3600)  # 6h

    def __init__(
        self,
        *,
        base_url: str = "https://api.scryfall.com",
        refresh_hours: int = 12,
        cheapest_prices: bool = False,
    ) -> None:
        """Initialize the Scryfall bulk client.

        Args:
            base_url: Scryfall API base URL (for metadata endpoint).
            refresh_hours: Hours before re-downloading data.
            cheapest_prices: After each load, give every card its cheapest
                paper printing's prices (see :meth:`apply_cheapest_prices`).
        """
        self._base_url = base_url
        self._refresh_seconds = refresh_hours * 3600
        self._cheapest_prices = cheapest_prices
        self._prices_task: asyncio.Task[None] | None = None

        # _cards: lowercase-name -> Card for O(1) exact lookup.
        # _unique_cards: deduplicated list for linear substring search.
        # Separation avoids double-counting DFCs which have two keys
        # in _cards (front-face name + full "//" name) but one entry
        # in _unique_cards.
        self._cards: dict[str, Card] = {}
        self._unique_cards: list[Card] = []
        self._loaded_at: float = 0.0  # monotonic timestamp; 0 = never loaded

        # ETag for conditional download (HTTP 304 Not Modified)
        self._etag: str | None = None
        self._etag_url: str | None = None  # URL the ETag applies to

        # asyncio.Lock prevents duplicate concurrent downloads
        self._load_lock = asyncio.Lock()

        # Background refresh task
        self._refresh_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        """Enter async context. Data is loaded lazily on first access, not here."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Cancel background work and release in-memory data."""
        for task in (self._refresh_task, self._prices_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._refresh_task = None
        self._prices_task = None

        self._cards.clear()
        self._unique_cards.clear()
        self._loaded_at = 0.0

    def start_background_refresh(self) -> None:
        """Start the background refresh loop.

        Creates an ``asyncio.Task`` that periodically calls
        :meth:`ensure_loaded` at the refresh interval. The task is
        cancelled in :meth:`__aexit__`.
        """
        if self._refresh_task is not None:
            return  # Already running
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def ensure_loaded(self) -> None:
        """Download and parse Oracle Cards if not loaded or stale.

        On first load failure, the error propagates (server cannot start
        without data). On **refresh** failure (data was previously loaded),
        logs a warning and serves stale data.

        Uses ``asyncio.Lock`` to prevent duplicate concurrent downloads.

        Raises:
            ScryfallBulkDownloadError: On first-load network/HTTP failure.
            ScryfallBulkError: On first-load parse failure.
        """
        if not self._is_stale():
            return

        async with self._load_lock:
            # Double-check after acquiring lock — another coroutine may have
            # already completed the load while we were waiting.
            if not self._is_stale():
                return

            is_refresh = self._loaded_at > 0
            log.info("scryfall_bulk.loading", base_url=self._base_url, is_refresh=is_refresh)

            try:
                # Step 1: Fetch metadata to get the current download URL.
                # Scryfall renamed download_uri → jsonl_download_uri (2026);
                # prefer the new field, fall back to the legacy name.
                metadata = await self._fetch_metadata()
                download_url = self._resolve_download_url(metadata)

                # Step 2: Download the bulk data (with ETag if URL matches)
                result = await self._download(download_url)

                if result is not None:
                    # Got new data — parse it
                    self._parse(result)
                    # The new cards carry their listed printing's prices until
                    # the cheapest printing's arrive.
                    self._start_price_update()

                # Either we parsed new data or got 304 — update timestamp
                self._loaded_at = time.monotonic()
                log.info(
                    "scryfall_bulk.loaded",
                    card_count=len(self._unique_cards),
                )
            except ScryfallBulkError:
                if is_refresh:
                    # Stale data is better than no data — retry in 5 min, not full interval
                    log.warning(
                        "scryfall_bulk.refresh_failed",
                        base_url=self._base_url,
                        exc_info=True,
                    )
                    self._loaded_at = time.monotonic() - self._refresh_seconds + 300
                    return
                raise

    async def get_card(self, name: str) -> Card | None:
        """Look up a card by exact name (case-insensitive).

        Args:
            name: Card name (front-face or full ``//`` name for DFCs).

        Returns:
            Card data, or None if not found.
        """
        await self.ensure_loaded()
        return self._cards.get(name.lower())

    async def search_cards(self, query: str, limit: int = 20) -> list[Card]:
        """Search cards by name substring (case-insensitive).

        Args:
            query: Substring to match against card names.
            limit: Maximum results to return.

        Returns:
            Matching cards, up to ``limit``.
        """
        await self.ensure_loaded()
        query_lower = query.lower()
        results: list[Card] = []
        for card in self._unique_cards:
            if query_lower in card.name.lower():
                results.append(card)
                if len(results) >= limit:
                    break
        return results

    async def search_by_type(self, type_query: str, limit: int = 20) -> list[Card]:
        """Search cards by type line substring (case-insensitive).

        Args:
            type_query: Substring to match against type lines.
            limit: Maximum results to return.

        Returns:
            Matching cards, up to ``limit``.
        """
        await self.ensure_loaded()
        query_lower = type_query.lower()
        results: list[Card] = []
        for card in self._unique_cards:
            if query_lower in card.type_line.lower():
                results.append(card)
                if len(results) >= limit:
                    break
        return results

    async def search_by_text(self, text_query: str, limit: int = 20) -> list[Card]:
        """Search cards by oracle text substring (case-insensitive).

        Args:
            text_query: Substring to match against oracle text.
            limit: Maximum results to return.

        Returns:
            Matching cards, up to ``limit``.
        """
        await self.ensure_loaded()
        query_lower = text_query.lower()
        results: list[Card] = []
        for card in self._unique_cards:
            oracle = card.oracle_text or ""
            if query_lower in oracle.lower():
                results.append(card)
                if len(results) >= limit:
                    break
        return results

    async def filter_cards(
        self,
        *,
        format: str | None = None,
        color_identity: frozenset[str] | None = None,
        type_contains: list[str] | None = None,
        text_contains: list[str] | None = None,
        text_any: list[str] | None = None,
        keywords: list[str] | None = None,
        cmc_eq: float | None = None,
        cmc_lte: float | None = None,
        max_price: float | None = None,
        rarity: str | None = None,
        name_contains: str | None = None,
        limit: int | None = 100,
    ) -> list[Card]:
        """Multi-criteria filter over all cards. Returns up to ``limit`` matches.

        Matches come back in bulk-file order, which says nothing about how good
        or popular a card is. To rank, pass ``limit=None`` and sort the result.

        All filter parameters are optional. Only cards matching **all** specified
        criteria are returned (AND logic). ``text_any`` is the exception: it uses
        OR logic among its terms.

        Args:
            format: Only cards where ``legalities[format] == "legal"``.
            color_identity: Cards whose color identity is a subset of this set.
            type_contains: ALL strings must appear in the type line (case-insensitive).
            text_contains: ALL strings must appear in oracle text (case-insensitive).
            text_any: ANY string must appear in oracle text (case-insensitive).
            keywords: ALL must be in the card's keywords (case-insensitive).
            cmc_eq: Exact CMC match.
            cmc_lte: CMC at or below this value.
            max_price: Maximum USD price (cards without a USD price are excluded).
            rarity: Exact rarity match (e.g. ``"mythic"``).
            name_contains: Substring match on card name (case-insensitive).
            limit: Maximum results to return, or ``None`` for every match.

        Returns:
            Matching cards, up to ``limit``.
        """
        await self.ensure_loaded()

        # Pre-compute lowercase versions of string filters
        type_lower = [t.lower() for t in type_contains] if type_contains else None
        text_lower = [t.lower() for t in text_contains] if text_contains else None
        text_any_lower = [t.lower() for t in text_any] if text_any else None
        kw_lower = [k.lower() for k in keywords] if keywords else None
        name_lower = name_contains.lower() if name_contains else None

        results: list[Card] = []
        for card in self._unique_cards:
            if format is not None and card.legalities.get(format) != "legal":
                continue
            if color_identity is not None and not frozenset(card.color_identity).issubset(
                color_identity
            ):
                continue
            if type_lower is not None:
                tl = card.type_line.lower()
                if not all(t in tl for t in type_lower):
                    continue
            if text_lower is not None:
                oracle = (card.oracle_text or "").lower()
                if not all(t in oracle for t in text_lower):
                    continue
            if text_any_lower is not None:
                oracle = (card.oracle_text or "").lower()
                if not any(t in oracle for t in text_any_lower):
                    continue
            if kw_lower is not None:
                card_kw = {k.lower() for k in card.keywords}
                if not all(k in card_kw for k in kw_lower):
                    continue
            if cmc_eq is not None and card.cmc != cmc_eq:
                continue
            if cmc_lte is not None and card.cmc > cmc_lte:
                continue
            if max_price is not None:
                if card.prices.usd is None:
                    continue
                try:
                    if float(card.prices.usd) > max_price:
                        continue
                except ValueError:
                    continue
            if rarity is not None and card.rarity != rarity:
                continue
            if name_lower is not None and name_lower not in card.name.lower():
                continue

            results.append(card)
            if limit is not None and len(results) >= limit:
                break

        return results

    async def cards_by_legality(self, format: str, status: str) -> list[Card]:
        """All cards with a specific legality status in a format.

        Args:
            format: Format name (e.g. ``"commander"``, ``"modern"``).
            status: Legality status (e.g. ``"banned"``, ``"restricted"``, ``"legal"``).

        Returns:
            All matching cards (no limit).
        """
        await self.ensure_loaded()
        return [card for card in self._unique_cards if card.legalities.get(format) == status]

    @property
    def loaded_at(self) -> float:
        """Monotonic time of the last load or refresh check (0.0 = never loaded).

        It changes whenever the card list may have been replaced, so a cache
        derived from :meth:`all_cards` can be keyed on it instead of holding
        the list, which would keep an old snapshot alive after a refresh.
        """
        return self._loaded_at

    async def all_cards(self) -> list[Card]:
        """All unique cards in the bulk data set.

        Ensures data is loaded before returning. This is the public
        interface for iterating over the full card pool — providers
        should use this instead of accessing ``_unique_cards`` directly.

        Returns:
            All unique Card objects.
        """
        await self.ensure_loaded()
        return self._unique_cards

    async def get_cards(self, names: list[str]) -> dict[str, Card | None]:
        """Batch exact-name lookup. Returns dict of name -> Card|None.

        Keys in the returned dict preserve the original casing from ``names``.
        Lookups are case-insensitive.

        Args:
            names: Card names to look up.

        Returns:
            Dict mapping each input name to its Card or None if not found.
        """
        await self.ensure_loaded()
        return {name: self._cards.get(name.lower()) for name in names}

    async def random_card(
        self,
        *,
        format: str | None = None,
        color_identity: frozenset[str] | None = None,
        type_contains: str | None = None,
        rarity: str | None = None,
    ) -> Card | None:
        """Random card from a filtered pool. None if pool is empty.

        Args:
            format: Only cards where ``legalities[format] == "legal"``.
            color_identity: Cards whose color identity is a subset of this set.
            type_contains: String that must appear in the type line (case-insensitive).
            rarity: Exact rarity match.

        Returns:
            A random matching card, or None if no cards match.
        """
        await self.ensure_loaded()

        pool: list[Card] = []
        type_lower = type_contains.lower() if type_contains else None

        for card in self._unique_cards:
            if format is not None and card.legalities.get(format) != "legal":
                continue
            if color_identity is not None and not frozenset(card.color_identity).issubset(
                color_identity
            ):
                continue
            if type_lower is not None and type_lower not in card.type_line.lower():
                continue
            if rarity is not None and card.rarity != rarity:
                continue
            pool.append(card)

        if not pool:
            return None
        return random.choice(pool)

    async def apply_cheapest_prices(self) -> int:
        """Give every loaded card its cheapest paper printing's prices.

        Oracle Cards lists one printing per card, often a preview with no
        price yet or a digital printing with none, so budget filters would
        drop Sol Ring. This reads every printing's prices from Scryfall's
        Default Cards file (shared by all clients, refreshed every 6 hours)
        and sets each of ``usd``, ``usd_foil`` and ``eur`` to the lowest
        among the card's paper printings. A field no paper printing has
        keeps its listed value. Runs in the background after each load when
        the client was created with ``cheapest_prices=True``.

        Returns:
            How many cards' prices changed.

        Raises:
            ScryfallBulkDownloadError: If Default Cards can't be downloaded.
            ScryfallBulkError: If it can't be read.
        """
        cheapest = await self._printing_prices()
        changed = 0
        for card in self._unique_cards:
            found = cheapest.get(card.oracle_id) if card.oracle_id is not None else None
            if found is None:
                continue
            prices = CardPrices(
                usd=found.usd if found.usd is not None else card.prices.usd,
                usd_foil=found.usd_foil if found.usd_foil is not None else card.prices.usd_foil,
                eur=found.eur if found.eur is not None else card.prices.eur,
            )
            if prices != card.prices:
                card.prices = prices
                changed += 1
        log.info(
            "scryfall_bulk.cheapest_prices_applied",
            changed=changed,
            cards=len(self._unique_cards),
            priced_cards=len(cheapest),
        )
        return changed

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _start_price_update(self) -> None:
        """Apply cheapest-printing prices in the background, when enabled.

        A running update reads ``self._unique_cards`` when it finishes, so it
        also covers cards loaded after it started.
        """
        if not self._cheapest_prices:
            return
        if self._prices_task is not None and not self._prices_task.done():
            return
        self._prices_task = asyncio.create_task(self._update_prices())

    async def _update_prices(self) -> None:
        """Background task: apply cheapest prices; on failure keep listed ones.

        Like ``_refresh_loop``, logs any error rather than letting it end the
        task unseen: cards with listed prices beat no cards.
        """
        try:
            await self.apply_cheapest_prices()
        except ScryfallBulkError:
            log.warning("scryfall_bulk.cheapest_prices_failed", exc_info=True)
        except Exception:
            log.error("scryfall_bulk.cheapest_prices_unexpected_error", exc_info=True)

    @async_cached(_printing_prices_cache)
    async def _printing_prices(self) -> dict[str, CheapestPrices]:
        """Cheapest paper prices by Oracle ID, from Scryfall's Default Cards file."""
        metadata = await self._fetch_bulk_metadata("default_cards")
        return await self._read_printing_prices(self._resolve_download_url(metadata))

    async def _read_printing_prices(self, url: str) -> dict[str, CheapestPrices]:
        """Stream Default Cards and keep each card's lowest price of each kind.

        Reads one printing (line) at a time, so memory stays flat whatever the
        file's size. Supports gzipped or plain JSON Lines.

        Raises:
            ScryfallBulkDownloadError: On HTTP errors or network failures.
            ScryfallBulkError: On a payload that isn't JSON Lines.
        """
        log.info("scryfall_bulk.cheapest_prices_loading", url=url)
        lowest: dict[str, dict[str, float]] = {}  # oracle_id -> field -> lowest price
        printings = 0
        headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
        try:
            async with (
                httpx.AsyncClient(timeout=120.0, headers=headers) as http,
                http.stream("GET", url) as response,
            ):
                if response.status_code != 200:
                    raise ScryfallBulkDownloadError(
                        f"HTTP {response.status_code} downloading printing prices from {url}"
                    )
                # aclosing: stop reading the download as soon as a line fails
                async with contextlib.aclosing(_jsonl_lines(response.aiter_bytes())) as lines:
                    async for line in lines:
                        try:
                            entry = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            raise ScryfallBulkError(
                                f"Failed to parse printing prices (line {printings + 1}): {exc}"
                            ) from exc
                        printings += 1
                        _add_printing_prices(lowest, entry)
        except httpx.RequestError as exc:
            raise ScryfallBulkDownloadError(
                f"Network error downloading printing prices: {exc}"
            ) from exc

        log.info("scryfall_bulk.cheapest_prices_loaded", printings=printings, cards=len(lowest))
        return {
            oracle_id: CheapestPrices(
                usd=_format_price(by_field.get("usd")),
                usd_foil=_format_price(by_field.get("usd_foil")),
                eur=_format_price(by_field.get("eur")),
            )
            for oracle_id, by_field in lowest.items()
        }

    def _is_stale(self) -> bool:
        """Check if the loaded data has exceeded the refresh interval."""
        if self._loaded_at == 0.0:
            return True
        return (time.monotonic() - self._loaded_at) >= self._refresh_seconds

    async def _fetch_metadata(self) -> dict:
        """Fetch the Oracle Cards metadata to get the current download URI.

        Raises:
            ScryfallBulkDownloadError: On HTTP errors or network failures.
        """
        return await self._fetch_bulk_metadata("oracle_cards")

    async def _fetch_bulk_metadata(self, bulk_type: str) -> dict:
        """Fetch one bulk file's metadata (``oracle_cards``, ``default_cards``, ...).

        Raises:
            ScryfallBulkDownloadError: On HTTP errors or network failures.
        """
        url = f"{self._base_url}/bulk-data/{bulk_type}"
        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept": "application/json",
                },
            ) as http:
                response = await http.get(url)
                if response.status_code != 200:
                    raise ScryfallBulkDownloadError(
                        f"HTTP {response.status_code} fetching bulk metadata from {url}"
                    )
                try:
                    return response.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ScryfallBulkDownloadError(
                        f"Metadata response is not valid JSON from {url}: {exc}"
                    ) from exc
        except httpx.RequestError as exc:
            raise ScryfallBulkDownloadError(f"Network error fetching bulk metadata: {exc}") from exc

    async def _download(self, url: str) -> bytes | None:
        """Download the Oracle Cards bulk data file.

        Sends ``If-None-Match`` header when the URL matches the previous
        download's URL and we have a saved ETag. Returns ``None`` on HTTP
        304 (data unchanged).

        Raises:
            ScryfallBulkDownloadError: On HTTP errors or network failures.
        """
        headers: dict[str, str] = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        }

        # Only send If-None-Match if the URL matches the one we got the ETag from
        if self._etag is not None and self._etag_url == url:
            headers["If-None-Match"] = self._etag

        try:
            async with httpx.AsyncClient(timeout=120.0, headers=headers) as http:
                response = await http.get(url)

                if response.status_code == 304:
                    log.info("scryfall_bulk.not_modified", url=url)
                    return None

                if response.status_code != 200:
                    raise ScryfallBulkDownloadError(
                        f"HTTP {response.status_code} downloading bulk data from {url}"
                    )

                # Save ETag for future conditional requests
                etag = response.headers.get("ETag")
                if etag:
                    self._etag = etag
                    self._etag_url = url

                return response.content
        except httpx.RequestError as exc:
            raise ScryfallBulkDownloadError(f"Network error downloading bulk data: {exc}") from exc

    @staticmethod
    def _resolve_download_url(metadata: dict) -> str:
        """Extract the download URL from bulk metadata across the field rename.

        Prefers ``jsonl_download_uri`` (current field); falls back to the
        legacy ``download_uri`` for older-shaped responses, with a log line
        so the fallback is never silent.

        Raises:
            ScryfallBulkError: If neither field carries a usable string.
        """
        jsonl_uri = metadata.get("jsonl_download_uri")
        if isinstance(jsonl_uri, str) and jsonl_uri:
            return jsonl_uri

        legacy_uri = metadata.get("download_uri")
        if isinstance(legacy_uri, str) and legacy_uri:
            log.info(
                "scryfall_bulk.legacy_download_uri_fallback",
                hint="metadata lacks jsonl_download_uri; using download_uri",
                metadata_keys=sorted(metadata.keys()),
            )
            return legacy_uri

        raise ScryfallBulkError(
            "Bulk metadata missing both 'jsonl_download_uri' and 'download_uri'. "
            f"Available keys: {list(metadata.keys())}"
        )

    @staticmethod
    def _decompress_if_gzip(raw: bytes) -> bytes:
        """Decompress the payload when it starts with gzip magic bytes.

        Scryfall's current bulk files are served as ``.jsonl.gz`` with
        ``Content-Type: application/gzip`` (no ``Content-Encoding``), so httpx
        does not auto-decompress them. Detection is content-based (magic
        bytes), never URL-based, so either field's download works.

        Raises:
            ScryfallBulkError: If the payload is corrupt gzip.
        """
        if raw[:2] != b"\x1f\x8b":
            return raw
        try:
            return gzip.decompress(raw)
        except (OSError, EOFError, zlib.error) as exc:
            raise ScryfallBulkError(f"Failed to parse bulk data: corrupt gzip: {exc}") from exc

    @staticmethod
    def _extract_entries(payload: bytes) -> list[JSONValue]:
        """Extract card entries from either payload shape.

        Legacy shape (``download_uri``): a single JSON document that is a
        JSON array of card objects. Current shape (``jsonl_download_uri``):
        JSONL — one card object per line. A lone JSON object is accepted as
        a single-entry payload (a degenerate one-line JSONL file).

        Raises:
            ScryfallBulkError: If the payload is neither shape.
        """
        try:
            raw = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ScryfallBulkClient._parse_jsonl(payload)

        if isinstance(raw, dict):
            return [raw]
        if not isinstance(raw, list):
            raise ScryfallBulkError("Bulk data is not a JSON array")
        return raw

    @staticmethod
    def _parse_jsonl(payload: bytes) -> list[JSONValue]:
        """Parse a JSONL payload: one JSON object per line, blank lines skipped.

        Raises:
            ScryfallBulkError: On undecodable bytes or a malformed line.
        """
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ScryfallBulkError(f"Failed to parse bulk data: {exc}") from exc

        entries: list[JSONValue] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ScryfallBulkError(
                    f"Failed to parse bulk data (JSONL line {lineno}): {exc}"
                ) from exc
        return entries

    def _parse(self, raw_bytes: bytes) -> None:
        """Parse Oracle Cards bulk data into the in-memory card dicts.

        Accepts both payload shapes across Scryfall's field rename: the
        legacy JSON array (``download_uri``) and the current gzipped JSONL
        (``jsonl_download_uri``). Shape is detected from content (gzip magic
        bytes), not from the URL. Each entry is validated through
        ``Card.model_validate()``.

        For DFCs, ``card.name`` contains the full ``"Front // Back"`` name.
        We key lookups by both the full name and the front face name
        (``name.split(" // ")[0]``).

        Raises:
            ScryfallBulkError: If the payload is malformed or yields no cards.
        """
        entries = self._extract_entries(self._decompress_if_gzip(raw_bytes))

        cards: dict[str, Card] = {}
        unique: list[Card] = []
        skipped = 0

        for entry in entries:
            if not isinstance(entry, dict):
                skipped += 1
                continue

            layout = entry.get("layout", "")
            if layout in _EXCLUDED_LAYOUTS:
                skipped += 1
                continue

            try:
                card = Card.model_validate(entry)
            except (ValidationError, ValueError) as exc:
                log.warning(
                    "scryfall_bulk.card_parse_error",
                    card_name=entry.get("name", "unknown"),
                    error=str(exc),
                )
                skipped += 1
                continue

            unique.append(card)

            # Key by full lowercase name for O(1) case-insensitive lookup
            full_name_lower = card.name.lower()
            cards[full_name_lower] = card

            # For DFCs, card.name is "Front // Back". Also key by front-face
            # name so lookup by either form works.
            if " // " in card.name:
                front_face = card.name.split(" // ")[0].lower()
                if front_face != full_name_lower:
                    cards[front_face] = card

        if skipped:
            log.info("scryfall_bulk.parse_summary", skipped=skipped, loaded=len(unique))

        if not unique:
            raise ScryfallBulkError(
                f"Parsed 0 cards from {len(entries)} entries ({skipped} skipped). "
                "Scryfall bulk data schema may have changed."
            )

        self._cards = cards
        self._unique_cards = unique

    async def _refresh_loop(self) -> None:
        """Background loop that periodically calls ensure_loaded().

        Runs until cancelled (via __aexit__). Errors are logged but do
        not stop the loop.
        """
        while True:
            await asyncio.sleep(self._refresh_seconds)
            try:
                await self.ensure_loaded()
            except ScryfallBulkError:
                log.warning("scryfall_bulk.background_refresh_error", exc_info=True)
            except Exception:
                log.error("scryfall_bulk.background_refresh_unexpected_error", exc_info=True)


# ---------------------------------------------------------------------------
# One card database per process
# ---------------------------------------------------------------------------


@dataclass
class _SharedClient:
    """A client handed out by :func:`shared_bulk_client`, and how many hold it."""

    client: ScryfallBulkClient
    users: int = 0


# Clients handed out by shared_bulk_client, by configuration. The bulk provider
# and the workflow server both need the card database, and each loaded copy
# holds about 300 MB of cards (8.5 KB each, measured on real Oracle Cards data).
_shared_clients: dict[Hashable, _SharedClient] = {}


@contextlib.asynccontextmanager
async def shared_bulk_client(
    *,
    base_url: str,
    refresh_hours: int,
    cheapest_prices: bool,
    factory: Callable[..., ScryfallBulkClient] = ScryfallBulkClient,
) -> AsyncGenerator[ScryfallBulkClient]:
    """Use the process's bulk client for these settings, creating it for the first user.

    Every lifespan that needs the card database enters this instead of making
    its own client, so the provider and the workflow server load the card
    data once between them. The client is entered for the first user and
    exited when the last one leaves, so a server running on its own still
    gets a client of its own.

    Args:
        base_url: Scryfall API base URL.
        refresh_hours: Hours before re-downloading data.
        cheapest_prices: See :class:`ScryfallBulkClient`.
        factory: Makes the client when no one holds one for these settings
            (the class itself; tests pass a stand-in).
    """
    key = (base_url, refresh_hours, cheapest_prices)
    shared = _shared_clients.get(key)
    if shared is None:
        client = factory(
            base_url=base_url, refresh_hours=refresh_hours, cheapest_prices=cheapest_prices
        )
        # Registered before the await, so a lifespan starting meanwhile shares it
        shared = _shared_clients[key] = _SharedClient(client)
        try:
            await client.__aenter__()
        except BaseException:
            del _shared_clients[key]
            raise
    shared.users += 1
    try:
        yield shared.client
    finally:
        shared.users -= 1
        if shared.users == 0:
            if _shared_clients.get(key) is shared:
                del _shared_clients[key]
            await shared.client.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Default Cards: every printing's prices
# ---------------------------------------------------------------------------


class _Gunzip:
    """Incremental gunzip for a download that may hold several gzip members."""

    def __init__(self) -> None:
        self._member = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
        self._mid_member = False  # True until the current member's end arrives

    def feed(self, data: bytes) -> bytes:
        """Decompress the next chunk of the download.

        Raises:
            zlib.error: On data that isn't gzip.
        """
        out = b""
        while data:
            out += self._member.decompress(data)
            self._mid_member = not self._member.eof
            data = b""
            if self._member.eof:
                # This member is complete; whatever follows starts the next one
                data = self._member.unused_data
                self._member = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
        return out

    def finish(self) -> None:
        """Check the download didn't stop part-way through a member.

        Raises:
            ScryfallBulkError: If it did (a truncated file).
        """
        if self._mid_member:
            raise ScryfallBulkError("Failed to parse bulk data: truncated gzip")


async def _jsonl_lines(chunks: AsyncIterator[bytes]) -> AsyncGenerator[bytes]:
    """Yield the non-blank lines of a streamed JSON Lines file, gunzipping as needed.

    Whether the file is gzipped is read from its first two bytes (Scryfall
    serves ``.jsonl.gz`` as plain ``application/gzip``, so httpx doesn't
    decompress it).

    Raises:
        ScryfallBulkError: On corrupt or truncated gzip, or a line longer than
            ``_MAX_LINE_BYTES`` (the file isn't JSON Lines).
    """
    gunzip: _Gunzip | None = None
    head = b""  # the first bytes, until there are enough to spot gzip
    started = False
    pending = b""  # an incomplete last line, waiting for the next chunk

    try:
        async for chunk in chunks:
            if not started:
                head += chunk
                if len(head) < len(_GZIP_MAGIC):
                    continue
                started, chunk, head = True, head, b""
                if chunk.startswith(_GZIP_MAGIC):
                    gunzip = _Gunzip()
            pending += gunzip.feed(chunk) if gunzip is not None else chunk
            *lines, pending = pending.split(b"\n")
            if len(pending) > _MAX_LINE_BYTES:
                raise ScryfallBulkError(
                    f"Bulk file has a line over {_MAX_LINE_BYTES} bytes; expected JSON Lines"
                )
            for line in lines:
                if line.strip():
                    yield line
        if gunzip is not None:
            gunzip.finish()
    except zlib.error as exc:
        raise ScryfallBulkError(f"Failed to parse bulk data: corrupt gzip: {exc}") from exc
    # The last line (no newline after it), or a file too short to have started
    for line in (pending + head).split(b"\n"):
        if line.strip():
            yield line


def _is_paper_printing(entry: dict) -> bool:
    """True for a printing sold as the playable paper card (see ``_NON_PLAYABLE_SET_TYPES``)."""
    return not (
        entry.get("digital")
        or entry.get("oversized")
        or entry.get("border_color") == "gold"
        or entry.get("set_type") in _NON_PLAYABLE_SET_TYPES
        or entry.get("layout") in _EXCLUDED_LAYOUTS
    )


def _add_printing_prices(lowest: dict[str, dict[str, float]], entry: object) -> None:
    """Fold one printing's prices into ``lowest`` (oracle_id -> field -> lowest price)."""
    if not isinstance(entry, dict) or not _is_paper_printing(entry):
        return
    oracle_id = entry.get("oracle_id")
    prices = entry.get("prices")
    if not isinstance(oracle_id, str) or not isinstance(prices, dict):
        return  # e.g. reversible cards keep their Oracle IDs on the faces
    for field in _PRICE_FIELDS:
        raw = prices.get(field)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        by_field = lowest.setdefault(oracle_id, {})
        current = by_field.get(field)
        if current is None or value < current:
            by_field[field] = value


def _format_price(value: float | None) -> str | None:
    """Back to Scryfall's price format ("1.23"), or None when there's no price."""
    return f"{value:.2f}" if value is not None else None
