"""Web research evidence for current approach planning.

The human-facing Pi orchestrator calls these helpers through ``prg.web_search``.
Queries and citations are persisted before a portfolio plan can be accepted, so
"use current tools" is an enforceable phase rather than prompt decoration.
"""

from __future__ import annotations

import html
import ipaddress
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from .errors import AutoProgrammingError


class WebResearchError(AutoProgrammingError):
    """A required web-research operation failed or lacks permission."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class SearchReport:
    query: str
    results: tuple[SearchResult, ...]
    searched_at: str

    def __str__(self) -> str:
        lines = [f"web search: {self.query!r} ({self.searched_at})"]
        for i, result in enumerate(self.results, 1):
            lines.append(f"  {i}. {result.title}\n     {result.url}")
            if result.snippet:
                lines.append(f"     {result.snippet}")
        return "\n".join(lines)


class _DuckResults(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._href: str | None = None
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        classes = set(str(values.get("class", "")).split())
        if tag == "a" and "result__a" in classes:
            self._href = str(values.get("href") or "")
            self._title = []
            self._in_title = True
        elif tag in ("a", "div") and "result__snippet" in classes:
            self._snippet = []
            self._in_snippet = True

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
        if self._in_snippet:
            self._snippet.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
            url = _unwrap_duck_url(self._href or "")
            title = _compact(" ".join(self._title))
            if url and title:
                self.results.append(SearchResult(title=title, url=url))
        if tag in ("a", "div") and self._in_snippet:
            self._in_snippet = False
            snippet = _compact(" ".join(self._snippet))
            if snippet and self.results:
                previous = self.results[-1]
                self.results[-1] = SearchResult(
                    title=previous.title,
                    url=previous.url,
                    snippet=snippet,
                )


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _is_public_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        host = parsed.hostname.casefold()
        if host == "localhost" or host.endswith((".localhost", ".local")):
            return False
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return True
        return not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_unspecified
        )
    except ValueError:
        return False


def _unwrap_duck_url(value: str) -> str:
    value = html.unescape(value)
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlparse(value)
    if "duckduckgo.com" in parsed.netloc:
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            unwrapped = unquote(target)
            return unwrapped if _is_public_http_url(unwrapped) else ""
    return value if _is_public_http_url(value) else ""


def search_web(query: str, *, limit: int = 6, timeout: float = 20.0) -> SearchReport:
    """Search the public web without requiring a second provider credential."""
    query = str(query).strip()
    if not query:
        raise WebResearchError("web_search requires a non-empty query.")
    if len(query) > 500 or "\x00" in query:
        raise WebResearchError(
            "web_search queries must be short abstract research questions "
            "(at most 500 characters)."
        )
    if limit < 1 or limit > 20:
        raise ValueError("web_search limit must be between 1 and 20.")
    request = Request(
        "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; AutoProgrammingResearch/0.2; "
                "+https://github.com/lillyguisnet/autoprogramming)"
            )
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(2_000_000).decode("utf-8", "replace")
    except Exception as exc:
        raise WebResearchError(
            f"Web search failed for {query!r}: {exc}. The approach plan remains "
            "paused because current-source research is required; fix network "
            "access or consult the user rather than planning from stale memory."
        ) from exc
    parser = _DuckResults()
    parser.feed(body)
    seen: set[str] = set()
    results: list[SearchResult] = []
    for item in parser.results:
        if item.url in seen:
            continue
        seen.add(item.url)
        results.append(item)
        if len(results) >= limit:
            break
    if not results:
        raise WebResearchError(
            f"Web search for {query!r} returned no parseable sources. Try a more "
            "specific query or another permitted search service; do not silently "
            "skip the research phase."
        )
    return SearchReport(
        query=query,
        results=tuple(results),
        searched_at=datetime.now(timezone.utc).isoformat(),
    )


def record_report(workspace, report: SearchReport) -> None:
    path = Path(workspace.research_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            record = {}
    else:
        record = {}
    searches = list(record.get("searches", []))
    searches.append({
        "query": report.query,
        "searched_at": report.searched_at,
        "results": [asdict(result) for result in report.results],
    })
    sources: dict[str, dict] = {}
    for search in searches:
        for result in search.get("results", []):
            if result.get("url"):
                sources[str(result["url"])] = dict(result)
    path.write_text(json.dumps({
        "searches": searches,
        "sources": list(sources.values()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")


def research_evidence(workspace) -> dict:
    path = Path(workspace.research_json)
    if not path.exists():
        return {"searches": [], "sources": []}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise WebResearchError(f"Malformed web research record at {path}: {exc}") from exc
    return value if isinstance(value, dict) else {"searches": [], "sources": []}


def ensure_researched(workspace, *, min_searches: int = 2, min_sources: int = 2) -> dict:
    evidence = research_evidence(workspace)
    searches = evidence.get("searches") or []
    sources = evidence.get("sources") or []
    if len(searches) < min_searches or len(sources) < min_sources:
        raise WebResearchError(
            "Portfolio planning was refused until the human-facing Pi "
            f"orchestrator performs current web research (need {min_searches} "
            f"queries and {min_sources} distinct sources; have {len(searches)} "
            f"and {len(sources)}). Use prg.web_search(...) with task-specific "
            "queries, inspect the sources, then submit the portfolio plan."
        )
    return evidence
