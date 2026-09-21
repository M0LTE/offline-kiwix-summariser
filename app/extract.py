"""HTML -> prose extraction for ZIM-served articles. Stdlib parser only.

Two hard-won details:
  * Class matching must be on whole tokens. The <html> root carries
    class="skin-thumbsize-clientpref-standard", and substring-matching the
    chrome token "thumb" marks the entire document as skipped.
  * End tags must close tolerantly (drop back to the nearest matching open
    tag). Popping exactly one stack entry misaligns permanently on malformed
    HTML and every subsequent text node is discarded.
"""
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

SKIP_TAGS = {
    "script", "style", "noscript", "table", "figure", "figcaption", "sup",
    "form", "svg", "math", "button", "select", "audio", "video", "iframe",
}

BLOCK_TAGS = {
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt",
    "blockquote", "section", "ul", "ol", "pre", "tr",
}

VOID = {
    "br", "hr", "img", "input", "link", "meta", "source", "area", "base",
    "col", "embed", "param", "track", "wbr",
}

CHROME_EXACT = {
    "infobox", "reflist", "refbegin", "refend", "toc", "hatnote", "metadata",
    "sidebar", "noprint", "catlinks", "thumbinner", "thumb", "mw-editsection",
    "mw-references-wrap", "mw-empty-elt", "shortdescription", "navbox",
    "sistersitebox", "mw-cite-backlink", "citation", "reference", "gallery",
    "mw-jump-link", "breadcrumbs", "navigation", "menu", "advert",
}
CHROME_PREFIX = ("navbox", "vector-", "mw-heading-", "portal-", "sidebar")

SITE_SUFFIX_RE = re.compile(r"\s+[-–|]\s+(?:Wikipedia|.*Stack Exchange)$", re.I)

JUNK_TEXT = re.compile(
    r"\[\s*(?:edit|citation needed|clarify|who|when|update|source)\s*\]", re.I
)


@dataclass(frozen=True)
class Section:
    """One heading and its content. `id` is the document-order index."""

    id: int
    title: str
    level: int
    words: int       # this section plus its subsections
    own_words: int   # this section alone

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "words": self.words,
            "own_words": self.own_words,
        }


@dataclass
class Extracted:
    text: str
    title: str | None
    sections: list[Section]
    # Parallel to `sections`; kept off the public shape because subtree text is
    # derived on demand rather than stored (that would be quadratic).
    own_texts: list[str] = field(default_factory=list, repr=False, compare=False)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def headings(self) -> list[str]:
        """Real headings only, in document order (the lead is not one)."""
        return [s.title for s in self.sections if s.title != LEAD_TITLE]

    def find_section(self, ref: str | int) -> Section | None:
        """Look a section up by id or by title, case/punctuation loose.

        A numeric ref is always an id, and ids coincide with list positions
        because build_sections numbers them from zero in document order. A
        title is matched with case and punctuation ignored, so "personal-life"
        finds "Personal life" but "early life" does not find "Early life and
        education".
        """
        s = str(ref).strip()
        if s.lstrip("-").isdigit():
            idx = int(s)
            return self.sections[idx] if 0 <= idx < len(self.sections) else None
        want = _norm_heading(s)
        for sec in self.sections:
            if _norm_heading(sec.title) == want:
                return sec
        return None

    def section_text(self, sec: Section, include_subsections: bool = True) -> str:
        """Prose for a section.

        By default includes nested subsections, which is what "summarise the
        History section" means to a reader. Pass include_subsections=False for
        the heading's own paragraphs only.
        """
        i = sec.id
        # The lead is a sibling of the top-level sections, not their parent,
        # even though it carries the synthetic level 1.
        if not include_subsections or sec.title == LEAD_TITLE:
            return self.own_texts[i]
        parts = [self.own_texts[i]]
        for j in range(i + 1, len(self.sections)):
            if self.sections[j].level <= sec.level:
                break
            parts.append(self.own_texts[j])
        return "\n\n".join(p for p in parts if p)


def _norm_heading(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.casefold())


def is_chrome(cls: str | None) -> bool:
    for tok in (cls or "").split():
        if tok in CHROME_EXACT or tok.startswith(CHROME_PREFIX):
            return True
    return False


def content_region(raw: str) -> str:
    """Inner HTML of the article body.

    MediaWiki/mwoffliner ZIMs wrap prose in .mw-parser-output. Other books
    (Gutenberg, wikivoyage, stackexchange, ...) do not, so fall back to
    <body>, then to the whole document.
    """
    for pattern, tag in (
        (r"""<div[^>]*class=["'][^"']*mw-parser-output[^"']*["'][^>]*>""", "div"),
        (r"<body[^>]*>", "body"),
    ):
        m = re.search(pattern, raw, re.I)
        if not m:
            continue
        i, depth = m.end(), 1
        for t in re.finditer(rf"<(/?){tag}\b", raw[i:], re.I):
            depth += -1 if t.group(1) else 1
            if depth == 0:
                return raw[i:i + t.start()]
        return raw[i:]
    return raw


class _Pending:
    """A section under construction: heading metadata plus raw text chunks."""

    __slots__ = ("title", "level", "chunks")

    def __init__(self, title: str | None, level: int):
        self.title = title
        self.level = level
        self.chunks: list[str] = []


HEADING_TAGS = ("h1", "h2", "h3", "h4")

# Pseudo-heading for the prose that precedes the first real heading.
LEAD_TITLE = "(lead)"


def clean_text(t: str) -> str:
    t = JUNK_TEXT.sub(" ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


class Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # Index 0 is the lead text that precedes the first heading.
        self._secs: list[_Pending] = [_Pending(None, 1)]
        self._skip: list[bool] = []
        self._open: list[str] = []
        self._heading: list[str] | None = None
        self._heading_level = 0
        self.headings: list[str] = []

    @property
    def skipping(self) -> bool:
        return any(self._skip)

    @property
    def _cur(self) -> _Pending:
        return self._secs[-1]

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        cls = dict(attrs).get("class", "")
        self._skip.append(tag in SKIP_TAGS or is_chrome(cls))
        self._open.append(tag)
        if self.skipping:
            return
        if tag in HEADING_TAGS:
            self._heading = []
            self._heading_level = int(tag[1])
        if tag in BLOCK_TAGS:
            self._cur.chunks.append("\n")

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        for i in range(len(self._open) - 1, -1, -1):
            if self._open[i] == tag:
                del self._skip[i:]
                del self._open[i:]
                break
        if tag in HEADING_TAGS:
            if self._heading is not None:
                h = " ".join("".join(self._heading).split())
                if h:
                    self.headings.append(h)
                    # The heading becomes the first line of its own section, so
                    # full-text output is unchanged from the flat version.
                    self._secs.append(_Pending(h, self._heading_level))
                    self._cur.chunks.append(f"\n{h}\n")
                self._heading = None
                self._heading_level = 0
        if tag in BLOCK_TAGS and not self.skipping:
            self._cur.chunks.append("\n")

    def handle_data(self, data):
        if self._heading is not None:
            # Heading text belongs to the section it opens, not the one before.
            self._heading.append(data)
            return
        if not self.skipping:
            self._cur.chunks.append(data)

    @property
    def sections(self) -> list[_Pending]:
        return self._secs

    def result(self) -> str:
        return clean_text("".join("".join(s.chunks) for s in self._secs))


def build_sections(pending: list[_Pending]) -> tuple[list[Section], list[str]]:
    """Assign stable ids and compute subtree word counts."""
    own_texts: list[str] = []
    levels: list[int] = []
    titles: list[str] = []
    for sec in pending:
        text = clean_text("".join(sec.chunks))
        # Drop the lead section when it is empty, so ids stay contiguous.
        if sec.title is None and not text:
            continue
        own_texts.append(text)
        levels.append(sec.level)
        titles.append(sec.title if sec.title is not None else LEAD_TITLE)

    out: list[Section] = []
    for i, title in enumerate(titles):
        own_words = len(own_texts[i].split())
        words = own_words
        if title != LEAD_TITLE:
            for j in range(i + 1, len(titles)):
                if levels[j] <= levels[i]:
                    break
                words += len(own_texts[j].split())
        out.append(Section(id=i, title=title, level=levels[i],
                           words=words, own_words=own_words))
    return out, own_texts


def extract(html: str, prefer_heading_as_title: bool = True) -> Extracted:
    p = Extractor()
    p.feed(content_region(html))
    text = p.result()
    sections, own_texts = build_sections(p.sections)

    title = None
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if m:
        title = " ".join(m.group(1).split())
        # Strip the site suffix, e.g. "Alan Turing - Wikipedia" or
        # "Periodic backup of Rpi3 - Raspberry Pi Stack Exchange".
        title = SITE_SUFFIX_RE.sub("", title).strip() or None
    if title is None and prefer_heading_as_title and p.headings:
        title = p.headings[0]

    return Extracted(
        text=text, title=title, sections=sections, own_texts=own_texts,
    )
