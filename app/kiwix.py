"""Kiwix client. Book-agnostic: everything is discovered from the OPDS catalog.

kiwix-serve exposes each book's content root as an Atom <link type="text/html">,
so titles resolve against that root with no assumption about the publication.
"""
import html
import re
from dataclasses import dataclass
from urllib.parse import quote, urlencode

import httpx

try:
    # Rejects DTDs / entity expansion outright.
    import defusedxml.ElementTree as ElementTree
except ModuleNotFoundError:  # pragma: no cover - dependency is declared
    import xml.etree.ElementTree as ElementTree  # type: ignore[no-redef]

ATOM = "{http://www.w3.org/2005/Atom}"

MAX_CATALOG_BYTES = 8 * 1024 * 1024

# Wikipedia-family ZIMs historically nest articles under an "A/" namespace;
# bare paths also work and both are tried before falling back to search.
PATH_CANDIDATES = ("{title}", "A/{title}")

RESULTS_RE = re.compile(r'<div class="results">(.*?)</div>\s*(?:<div class="footer"|$)', re.S)
LINK_RE = re.compile(r'<a href="([^"]+)"[^>]*>(.*?)</a>', re.S)


class KiwixError(Exception):
    pass


class BookNotFound(KiwixError):
    pass


class ArticleNotFound(KiwixError):
    pass


@dataclass(frozen=True)
class Book:
    key: str
    title: str
    name: str
    flavour: str | None
    language: str | None
    article_count: int | None
    content_root: str
    uuid: str | None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "name": self.name,
            "flavour": self.flavour,
            "language": self.language,
            "article_count": self.article_count,
            "content_root": self.content_root,
            "uuid": self.uuid,
        }


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    html: str
    resolved_via: str


def _text(el, tag: str) -> str | None:
    child = el.find(f"{ATOM}{tag}")
    if child is None:
        child = el.find(tag)
    return " ".join((child.text or "").split()) if child is not None and child.text else None


def parse_search_results(page: str, limit: int) -> list["SearchResult"]:
    """Pull (url, title) pairs out of kiwix-serve's HTML search response."""
    m = RESULTS_RE.search(page)
    body = m.group(1) if m else page
    out: list[SearchResult] = []
    for href, label in LINK_RE.findall(body):
        title = " ".join(html.unescape(re.sub(r"<[^>]+>", "", label)).split())
        if not href.startswith("/content/") or not title:
            continue
        out.append(SearchResult(url=html.unescape(href), title=title))
        if len(out) >= limit:
            break
    return out


def _norm(title: str) -> str:
    """Compare-friendly form of a page title."""
    return re.sub(r"[_\-\s]+", " ", title).strip().casefold()


class KiwixClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        # kiwix-serve redirects /content/book/A/Title -> /content/book/Title
        return httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout, follow_redirects=True
        )

    async def books(self) -> list[Book]:
        """All publications currently served, straight from the OPDS catalog."""
        async with self._client() as c:
            r = await c.get("/catalog/v2/entries")
            if r.status_code != 200:
                raise KiwixError(f"catalog returned HTTP {r.status_code}")
            raw = r.content
            if len(raw) > MAX_CATALOG_BYTES:
                raise KiwixError(f"catalog response too large ({len(raw)} bytes)")
            if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
                raise KiwixError("catalog response contains a DTD; refusing to parse")
            root = ElementTree.fromstring(raw)

        out: list[Book] = []
        for entry in root.findall(f"{ATOM}entry"):
            href = None
            for link in entry.findall(f"{ATOM}link"):
                if link.get("type") == "text/html" and (link.get("href") or "").startswith(
                    "/content/"
                ):
                    href = link.get("href")
                    break
            if not href:
                continue
            key = href.split("/content/", 1)[1].strip("/")
            if not key:
                continue
            name = _text(entry, "name") or key
            count_raw = _text(entry, "articleCount")
            out.append(
                Book(
                    key=key,
                    title=_text(entry, "title") or name,
                    name=name,
                    flavour=_text(entry, "flavour"),
                    language=_text(entry, "language"),
                    article_count=int(count_raw) if count_raw and count_raw.isdigit() else None,
                    content_root=f"/content/{key}",
                    uuid=(_text(entry, "id") or "").replace("urn:uuid:", "") or None,
                )
            )
        return out

    async def get_book(self, key: str) -> Book:
        for b in await self.books():
            if b.key == key or b.name == key:
                return b
        raise BookNotFound(f"book {key!r} is not served by this Kiwix instance")

    async def search(self, book_key: str, pattern: str, limit: int = 10) -> list[SearchResult]:
        """Full-text search. kiwix-serve answers HTML, so parse the result list."""
        query = urlencode({"books.name": book_key, "pattern": pattern})
        async with self._client() as c:
            r = await c.get(f"/search?{query}")
            if r.status_code != 200:
                raise KiwixError(f"search returned HTTP {r.status_code}")
            page = r.text

        return parse_search_results(page, limit)

    async def resolve(self, book: Book, title: str, strict: bool = False) -> Article:
        """Find the article for a human-supplied title.

        Direct path first (exact, cheap), then search with an exact-title
        preference -- search alone is unreliable because a query for
        "alan turing" ranks "List of things named after Alan Turing" first.

        With strict=True, a non-exact search hit is refused rather than
        summarised. Kiwix relevance ranking will happily return "Polish Enigma
        double" for a query about the Enigma machine, and silently summarising
        the wrong article is worse than failing.
        """
        slug = quote(title.strip().replace(" ", "_"), safe="")
        async with self._client() as c:
            for tmpl in PATH_CANDIDATES:
                url = f"{book.content_root}/{tmpl.format(title=slug)}"
                r = await c.get(url)
                if r.status_code == 200 and r.content:
                    return Article(
                        title=title.strip(),
                        url=str(r.url),
                        html=r.text,
                        resolved_via=f"path:{tmpl}",
                    )

            results = await self.search(book.key, title)
            if not results:
                raise ArticleNotFound(f"no article matching {title!r} in book {book.key!r}")

            want = _norm(title)
            exact = next((res for res in results if _norm(res.title) == want), None)
            if exact is None and strict:
                names = ", ".join(repr(r.title) for r in results[:5])
                raise ArticleNotFound(
                    f"no article titled exactly {title!r} in book {book.key!r}; "
                    f"closest matches: {names}"
                )
            pick = exact or results[0]

            r = await c.get(pick.url)
            if r.status_code != 200:
                raise ArticleNotFound(f"resolved URL {pick.url} returned HTTP {r.status_code}")
            return Article(
                title=pick.title,
                url=str(r.url),
                html=r.text,
                resolved_via="search:exact" if exact else "search:fuzzy",
            )

    async def healthcheck(self) -> tuple[bool, str]:
        # Short timeout of its own: content fetches need headroom for a cold ZIM
        # read, but /healthz should answer quickly enough to be useful.
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=10.0) as c:
                r = await c.get("/catalog/v2/entries")
                return r.status_code == 200, f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in /healthz
            return False, f"{type(exc).__name__}: {exc}"
