"""Scryfall MCP provider — card search, lookup, pricing, rulings, and set info."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import TYPE_CHECKING, Annotated

import structlog
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan
from fastmcp.tools import ToolResult
from pydantic import Field

from mtg_mcp_server.config import Settings
from mtg_mcp_server.providers import (
    ATTRIBUTION_SCRYFALL,
    TAGS_LOOKUP,
    TAGS_PRICING,
    TAGS_SEARCH,
    TOOL_ANNOTATIONS,
    format_legalities,
)
from mtg_mcp_server.services.scryfall import CardNotFoundError, ScryfallClient, ScryfallError
from mtg_mcp_server.utils.formatters import ResponseFormat, format_card_detail, format_card_line
from mtg_mcp_server.utils.slim import slim_card

if TYPE_CHECKING:
    from mtg_mcp_server.types import Card

# Module-level client set by the lifespan. This pattern is required because
# FastMCP's Depends()/lifespan_context DI doesn't propagate through mount().
_client: ScryfallClient | None = None


@lifespan
async def scryfall_lifespan(server: FastMCP):
    """Manage the ScryfallClient lifecycle.

    Create the client once at startup using settings (honoring env var overrides),
    keep the httpx connection pool open for the server's lifetime, then tear down.
    """
    global _client
    settings = Settings()
    client = ScryfallClient(base_url=settings.scryfall_base_url)
    async with client:
        _client = client
        yield {}
    _client = None


scryfall_mcp = FastMCP("Scryfall", lifespan=scryfall_lifespan, mask_error_details=True)

log = structlog.get_logger(provider="scryfall")


def _get_client() -> ScryfallClient:
    """Return the initialized client or raise if the lifespan hasn't started."""
    if _client is None:
        raise RuntimeError("ScryfallClient not initialized — server lifespan not running")
    return _client


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_SEARCH)
async def search_cards(
    query: Annotated[
        str,
        Field(
            description="Scryfall search query (e.g. 'f:commander id:sultai t:creature cmc<=3'). See scryfall.com/docs/syntax"
        ),
    ],
    page: Annotated[int, Field(description="Page number for paginated results, 1-indexed")] = 1,
    limit: Annotated[
        int,
        Field(description="Max cards to return (default 30, 0 for all)"),
    ] = 30,
    response_format: Annotated[
        ResponseFormat,
        Field(description="Output verbosity: 'detailed' (default) or 'concise'"),
    ] = "detailed",
) -> ToolResult:
    """Search for Magic cards using Scryfall syntax.

    Examples: "f:commander id:sultai t:creature", "o:destroy t:instant cmc<=3"
    See https://scryfall.com/docs/syntax for full syntax reference.
    """
    if limit < 0:
        raise ToolError(f"limit must be >= 0 (0 for all), got {limit}")

    client = _get_client()
    try:
        result = await client.search_cards(query, page=page)
    except CardNotFoundError as exc:
        raise ToolError(f"No cards found for query: '{query}'. Check your search syntax.") from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    cards = result.data if limit == 0 else result.data[:limit]
    showing = len(cards)
    total = len(result.data)

    lines = [f"Found {result.total_cards} cards (showing {showing} of {total}, page {page}):"]
    for card in cards:
        lines.append(format_card_line(card, response_format=response_format))
    if showing < total:
        lines.append(f"\n{total - showing} more on this page — increase limit to see them.")
    if result.has_more:
        lines.append(f"More results available — use page={page + 1}")
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content={
            "query": query,
            "total_cards": result.total_cards,
            "page": page,
            "has_more": result.has_more,
            "showing": showing,
            "card_detail_uri_template": "mtg://card/{name}",
            "cards": [slim_card(card) for card in cards],
        },
    )


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_LOOKUP)
async def card_details(
    name: Annotated[
        str,
        Field(description="Card name — exact match by default (e.g. 'Muldrotha, the Gravetide')"),
    ],
    fuzzy: Annotated[
        bool,
        Field(
            description="Use fuzzy matching for approximate names (e.g. 'muldrotha' finds 'Muldrotha, the Gravetide')"
        ),
    ] = False,
    response_format: Annotated[
        ResponseFormat,
        Field(description="Output verbosity: 'detailed' (default) or 'concise'"),
    ] = "detailed",
) -> ToolResult:
    """Get full details for a Magic card by exact or fuzzy name."""
    client = _get_client()
    try:
        card = await client.get_card_by_name(name, fuzzy=fuzzy)
    except CardNotFoundError as exc:
        raise ToolError(f"Card not found: '{name}'. Check spelling or try fuzzy=true.") from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    lines = format_card_detail(card, response_format=response_format)
    if response_format == "detailed":
        lines.append(f"Legalities: {format_legalities(card.legalities)}")
        lines.append(f"Scryfall: {card.scryfall_uri}")
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content=card.model_dump(mode="json"),
    )


# A card's cheapest paper printing with a USD price, as a Scryfall search. Not
# counted: digital, oversized, gold-bordered and memorabilia printings (30th
# Anniversary Edition, World Championship decks), as in the card database.
_CHEAPEST_PRINTING_QUERY = (
    "{card} usd>=0 -is:digital -is:oversized -border:gold -st:memorabilia prefer:usd-low"
)


async def _cheapest_printing(client: ScryfallClient, card: Card) -> Card | None:
    """The card's cheapest paper printing that has a USD price, or None.

    None when no paper printing has a price, or when the search fails: the
    printing ``card`` came from still answers the question.
    """
    # The Oracle ID matches every printing; a quoted exact name is the fallback
    who = f"oracleid:{card.oracle_id}" if card.oracle_id else f'!"{card.name}"'
    try:
        result = await client.search_cards(_CHEAPEST_PRINTING_QUERY.format(card=who))
    except CardNotFoundError:
        return None
    except ScryfallError as exc:
        log.warning("card_price.cheapest_printing_failed", card=card.name, error=str(exc))
        return None
    return result.data[0] if result.data else None


def _usd(card: Card) -> float | None:
    """A printing's USD price as a number, or None if it has none."""
    try:
        return float(card.prices.usd) if card.prices.usd is not None else None
    except ValueError:
        return None


def _costs_less(other: Card, card: Card) -> bool:
    """True when ``other`` is another printing with a USD price below ``card``'s (or ``card`` has none)."""
    other_usd, card_usd = _usd(other), _usd(card)
    if other.id == card.id or other_usd is None:
        return False
    return card_usd is None or other_usd < card_usd


def _printing(card: Card) -> str:
    """A printing's set and collector number, e.g. "C21 #263"."""
    return f"{card.set_code.upper()} #{card.collector_number}"


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_PRICING)
async def card_price(
    name: Annotated[str, Field(description="Card name for price lookup (exact match)")],
) -> ToolResult:
    """Get current prices for a Magic card. Prices update once per day.

    Shows the printing Scryfall lists for the card, and its cheapest paper
    printing when that one costs less. The listed printing is often a preview
    with no price yet, or a special printing that costs more than a reprint.
    """
    client = _get_client()
    try:
        card = await client.get_card_by_name(name)
    except CardNotFoundError as exc:
        raise ToolError(f"Card not found: '{name}'. Check spelling.") from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    lines = [f"**{card.name}** — Prices ({_printing(card)})"]
    if card.prices.usd:
        lines.append(f"  USD: ${card.prices.usd}")
    if card.prices.usd_foil:
        lines.append(f"  USD (foil): ${card.prices.usd_foil}")
    if card.prices.eur:
        lines.append(f"  EUR: \u20ac{card.prices.eur}")
    if not any([card.prices.usd, card.prices.usd_foil, card.prices.eur]):
        lines.append("  No price data available.")

    # Worth a line only when it beats the listed printing's USD price
    cheapest = await _cheapest_printing(client, card)
    cheaper = cheapest if cheapest is not None and _costs_less(cheapest, card) else None
    if cheaper is not None:
        lines.append(f"  Cheapest printing: ${cheaper.prices.usd} ({_printing(cheaper)})")

    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content={
            "name": card.name,
            "set": card.set_code,
            "collector_number": card.collector_number,
            "prices": card.prices.model_dump(mode="json"),
            "cheapest_printing": (
                {
                    "set": cheaper.set_code,
                    "collector_number": cheaper.collector_number,
                    "usd": cheaper.prices.usd,
                }
                if cheaper is not None
                else None
            ),
        },
    )


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_LOOKUP)
async def card_rulings(
    name: Annotated[str, Field(description="Card name to get official rulings for (exact match)")],
) -> ToolResult:
    """Get official rulings and clarifications for a Magic card."""
    client = _get_client()
    try:
        card = await client.get_card_by_name(name)
        rulings = await client.get_rulings(card.id)
    except CardNotFoundError as exc:
        raise ToolError(f"Card not found: '{name}'. Check spelling.") from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    rulings_data = [r.model_dump(mode="json") for r in rulings]
    if not rulings:
        return ToolResult(
            content=f"**{card.name}** — No rulings available." + ATTRIBUTION_SCRYFALL,
            structured_content={"name": card.name, "total_rulings": 0, "rulings": []},
        )

    lines = [f"**{card.name}** — {len(rulings)} ruling(s):"]
    for ruling in rulings:
        lines.append(f"  [{ruling.published_at}] {ruling.comment}")
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content={
            "name": card.name,
            "total_rulings": len(rulings),
            "rulings": rulings_data,
        },
    )


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_LOOKUP)
async def set_info(
    set_code: Annotated[
        str,
        Field(description="Set code (e.g. 'dom', 'mh2', 'lci')"),
    ],
) -> ToolResult:
    """Get metadata for a Magic set by its code."""
    client = _get_client()
    try:
        info = await client.get_set(set_code)
    except CardNotFoundError as exc:
        raise ToolError(f"Set not found: '{set_code}'. Check the set code.") from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    lines = [
        f"**{info.name}** ({info.code.upper()})",
        f"Type: {info.set_type}",
    ]
    if info.released_at:
        lines.append(f"Released: {info.released_at}")
    lines.append(f"Card count: {info.card_count}")
    if info.digital:
        lines.append("Digital-only set")
    if info.scryfall_uri:
        lines.append(f"Scryfall: {info.scryfall_uri}")
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content=info.model_dump(mode="json"),
    )


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_SEARCH)
async def whats_new(
    days: Annotated[
        int,
        Field(description="Look back this many days for recent cards (minimum 1)"),
    ] = 30,
    set_code: Annotated[
        str | None,
        Field(description="Filter to a specific set code (e.g. 'mh3', 'lci')"),
    ] = None,
    format: Annotated[
        str | None,
        Field(
            description="Filter to cards legal in a format (e.g. 'standard', 'commander', 'modern')"
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Max cards to return (default 30, 0 for all)"),
    ] = 30,
    response_format: Annotated[
        ResponseFormat,
        Field(description="Output verbosity: 'detailed' (default) or 'concise'"),
    ] = "detailed",
) -> ToolResult:
    """Find recently printed or released Magic cards.

    Searches Scryfall for cards released within the given number of days.
    Optionally filter by set or format legality.
    """
    if days < 1:
        raise ToolError("days must be at least 1.")
    if limit < 0:
        raise ToolError(f"limit must be >= 0 (0 for all), got {limit}")

    date_str = (date.today() - timedelta(days=days)).isoformat()
    query_parts = [f"date>={date_str}"]
    if set_code:
        query_parts.append(f"s:{set_code}")
    if format:
        query_parts.append(f"f:{format}")
    query = " ".join(query_parts)

    client = _get_client()
    try:
        result = await client.search_cards(query)
    except CardNotFoundError as exc:
        raise ToolError(
            f"No new cards found in the last {days} day(s)"
            + (f" for set '{set_code}'" if set_code else "")
            + (f" in format '{format}'" if format else "")
            + "."
        ) from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    cards = result.data if limit == 0 else result.data[:limit]
    showing = len(cards)
    total = len(result.data)

    lines = [f"Found {result.total_cards} card(s) released in the last {days} day(s):"]
    for card in cards:
        if response_format == "concise":
            set_label = card.set_code.upper() if card.set_code else ""
            lines.append(f"  {card.name} {card.mana_cost or ''} [{set_label}]")
        else:
            set_label = card.set_code.upper() if card.set_code else ""
            lines.append(f"  {card.name} {card.mana_cost or ''} — {card.type_line} [{set_label}]")
    if showing < total:
        lines.append(f"\n{total - showing} more on this page — increase limit to see them.")
    if result.has_more:
        lines.append(
            "\nMore results available — refine your search with set_code or format filters."
        )
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content={
            "days": days,
            "set_code": set_code,
            "format": format,
            "total_cards": result.total_cards,
            "showing": showing,
            "has_more": result.has_more,
            "cards": [slim_card(card) for card in cards],
        },
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@scryfall_mcp.resource("mtg://set/{code}")
async def set_resource(code: str) -> str:
    """Set metadata as JSON."""
    client = _get_client()
    try:
        info = await client.get_set(code)
        return info.model_dump_json()
    except CardNotFoundError:
        log.debug("resource.set_not_found", code=code)
        return json.dumps({"error": f"Set not found: {code}"})
    except ScryfallError as exc:
        log.warning("resource.set_error", code=code, error=str(exc))
        return json.dumps({"error": f"Scryfall error: {exc}"})


@scryfall_mcp.resource("mtg://card/{name}")
async def card_resource(name: str) -> str:
    """Get card data as JSON by exact name."""
    client = _get_client()
    try:
        card = await client.get_card_by_name(name)
        return card.model_dump_json()
    except CardNotFoundError:
        log.debug("resource.card_not_found", name=name)
        return json.dumps({"error": f"Card not found: {name}"})
    except ScryfallError as exc:
        log.warning("resource.card_error", name=name, error=str(exc))
        return json.dumps({"error": f"Scryfall error: {exc}"})


@scryfall_mcp.resource("mtg://card/{name}/rulings")
async def card_rulings_resource(name: str) -> str:
    """Get card rulings as JSON by card name."""
    client = _get_client()
    try:
        card = await client.get_card_by_name(name)
        rulings = await client.get_rulings(card.id)
        return json.dumps([r.model_dump() for r in rulings])
    except CardNotFoundError:
        log.debug("resource.rulings_not_found", name=name)
        return json.dumps({"error": f"Card not found: {name}"})
    except ScryfallError as exc:
        log.warning("resource.rulings_error", name=name, error=str(exc))
        return json.dumps({"error": f"Scryfall error: {exc}"})
