"""Turn a (book, title, target word count) request into a summary."""
import re
from dataclasses import asdict, dataclass, field

from .config import Settings
from .extract import extract
from .kiwix import Article, KiwixClient
from .llm import OllamaClient

# Split after sentence-ending punctuation. The lookbehind suppresses single
# letter abbreviations ("U.S. government"); over-splitting is preferred to
# under-splitting, since a missed split degrades the trim to a mid-sentence cut.
SENTENCE_RE = re.compile(r"(?<![A-Z]\.)(?<=[.!?])\s+")

SYSTEM_RULES = (
    "Cover only the most important facts from the supplied text. "
    "Never introduce information that is not present in it. "
    "Write continuous prose. Output only the summary -- no heading, no "
    "preamble, no commentary."
)


class SummariseError(Exception):
    pass


class ArticleTooLong(SummariseError):
    pass


class NothingToSummarise(SummariseError):
    pass


class SectionNotFound(SummariseError):
    pass


@dataclass
class SummaryResult:
    summary: str
    target_words: int
    actual_words: int
    passes: int
    trimmed: bool
    within_tolerance: bool
    article_title: str
    book_key: str
    source_url: str
    resolved_via: str
    source_words: int
    prompt_tokens: int | None
    prefill_tok_s: float | None
    decode_tok_s: float | None
    llm_seconds: float
    section_id: int | None = None
    section_title: str | None = None
    include_subsections: bool = True
    sections: list[dict] = field(default_factory=list)

    @property
    def word_delta(self) -> int:
        return self.actual_words - self.target_words

    def to_dict(self) -> dict:
        d = asdict(self)
        d["word_delta"] = self.word_delta
        # `sections` is not truncated: it is the index a second request picks
        # its `section` from, and even a 55-section article serialises to ~5KB.
        return d


def _tolerance(n: int) -> tuple[int, int]:
    span = max(3, round(n * 0.15))
    return max(1, n - span), n + span


def fit_to_budget(text: str, words: int) -> tuple[str, bool]:
    """Trim to the target at a sentence boundary.

    Prompting alone cannot hold a tight word budget on a small model: qwen3:8b
    plateaus near ~45 words however the instruction is phrased. Sentences are
    the smallest unit that still reads as prose, so pack them greedily up to the
    upper tolerance. Returns (text, whether a trim was applied).
    """
    stripped = text.strip()
    if not stripped:
        return "", False

    _, hi = _tolerance(words)
    sentences = [s for s in SENTENCE_RE.split(stripped) if s.strip()]
    kept: list[str] = []
    count = 0
    for s in sentences:
        n = len(s.split())
        if count + n > hi:
            break
        kept.append(s)
        count += n

    if not kept:
        # First sentence alone exceeds the budget: cut on a word boundary.
        clipped = " ".join(stripped.split()[:hi]).rstrip(" ,;:-")
        return (clipped + "." if clipped else stripped), True

    fitted = " ".join(kept)
    return fitted, fitted != stripped


def _draft_prompt(title: str, text: str, words: int) -> str:
    lo, hi = _tolerance(words)
    return (
        f'Below is the full text of the article "{title}".\n'
        f"Write a summary of exactly {words} words (acceptable range {lo}-{hi}). "
        f"{SYSTEM_RULES}\n\nARTICLE:\n{text}\n"
    )


def _section_prompt(title: str, heading: str, text: str, words: int) -> str:
    lo, hi = _tolerance(words)
    return (
        f'Below is the "{heading}" section of the article "{title}".\n'
        f"Write a summary of that section only, of exactly {words} words "
        f"(acceptable range {lo}-{hi}). {SYSTEM_RULES}\n\nSECTION:\n{text}\n"
    )


def _compress_prompt(draft: str, words: int, prev_count: int | None = None) -> str:
    lo, hi = _tolerance(words)
    feedback = (
        f"An earlier attempt came in at {prev_count} words, which is too long.\n"
        if prev_count
        else ""
    )
    return (
        f"{feedback}"
        f"Rewrite the text below as a summary of exactly {words} words "
        f"(acceptable range {lo}-{hi}). Keep every fact already present and "
        f"add nothing new. Count the words carefully before answering. "
        f"{SYSTEM_RULES}\n\n"
        f"TEXT:\n{draft}\n"
    )


def _num_predict(words: int, headroom: float = 1.6) -> int:
    """Cap generation near the target. A loose cap invites overshoot."""
    return int(words * headroom) + 32


class Summariser:
    def __init__(self, kiwix: KiwixClient, llm: OllamaClient, settings: Settings):
        self.kiwix = kiwix
        self.llm = llm
        self.s = settings

    def _num_ctx(self, approx_tokens: int) -> int:
        return (
            self.s.ollama_num_ctx_long
            if approx_tokens > self.s.long_article_tokens
            else self.s.ollama_num_ctx
        )

    async def run(
        self,
        book_key: str,
        title: str,
        words: int,
        strict: bool = False,
        section: str | int | None = None,
        include_subsections: bool = True,
    ) -> SummaryResult:
        book = await self.kiwix.get_book(book_key)
        article = await self.kiwix.resolve(book, title, strict=strict)
        return await self._produce(
            article, book.key, words,
            section=section, include_subsections=include_subsections,
        )

    async def _produce(
        self,
        article: Article,
        book_key: str,
        words: int,
        section: str | int | None = None,
        include_subsections: bool = True,
    ) -> SummaryResult:
        extracted = extract(article.html)
        title = extracted.title or article.title

        if extracted.word_count < 20:
            raise NothingToSummarise(
                f"only {extracted.word_count} words of prose could be extracted from "
                f"{article.url}; the page may be a redirect, index or image-only entry"
            )

        sec = None
        if section is not None:
            sec = extracted.find_section(section)
            if sec is None:
                names = ", ".join(repr(s.title) for s in extracted.sections[:10])
                raise SectionNotFound(
                    f"no section {section!r} in {title!r}; first sections: {names}"
                )
            source = extracted.section_text(sec, include_subsections)
            scope = sec.words if include_subsections else sec.own_words
        else:
            source = extracted.text
            scope = extracted.word_count

        if len(source.split()) < 20:
            if sec is None:
                raise NothingToSummarise(
                    f"only {len(source.split())} words of prose could be extracted"
                )
            hint = (
                f" Section {sec.id} is a container heading whose content lives in "
                f"its subsections ({sec.words} words in total); retry with "
                f"include_subsections=true."
                if not include_subsections and sec.words >= 20
                else ""
            )
            raise NothingToSummarise(
                f"section {sec.title!r} holds only {len(source.split())} words of "
                f"prose at this scope.{hint}"
            )
        if scope > self.s.max_source_words:
            raise ArticleTooLong(
                f"requested text is {scope} words, above the "
                f"{self.s.max_source_words} limit"
            )

        approx_tokens = int(scope * 1.35)
        ctx = self._num_ctx(approx_tokens)
        two_pass = words < self.s.two_pass_below_words
        draft_target = max(words * 3, 120) if two_pass else words
        lo, hi = _tolerance(words)

        prompt = (
            _section_prompt(title, sec.title, source, draft_target)
            if sec
            else _draft_prompt(title, source, draft_target)
        )

        gens = []
        # Drafting a tight target straight out of a 13k-token article loses
        # content selection quality, so small targets draft wide first. The
        # second pass has a tiny prompt, so its prefill is negligible.
        g = await self.llm.generate(
            prompt,
            num_ctx=ctx,
            num_predict=_num_predict(draft_target, 1.8),
            passed=1,
        )
        gens.append(g)
        summary = g.text

        # The model overshoots word targets by ~35% regardless of instruction.
        # Telling it the previous count and re-compressing converges cheaply.
        correction_ctx = min(ctx, 8192)
        prev_count = len(summary.split())
        for _ in range(self.s.max_correction_passes):
            if prev_count <= hi:
                break
            g = await self.llm.generate(
                _compress_prompt(summary, words, prev_count),
                num_ctx=correction_ctx,
                num_predict=_num_predict(words),
                passed=len(gens) + 1,
            )
            gens.append(g)
            new_count = len(g.text.split())
            # Stop on plateau: the model has a floor it will not compress past,
            # and further passes only burn GPU time.
            if new_count >= prev_count:
                break
            summary, prev_count = g.text, new_count

        summary, trimmed = fit_to_budget(summary, words)
        actual_words = len(summary.split())

        return SummaryResult(
            summary=summary,
            target_words=words,
            actual_words=actual_words,
            passes=len(gens),
            trimmed=trimmed,
            within_tolerance=lo <= actual_words <= hi,
            article_title=title,
            book_key=book_key,
            source_url=article.url,
            resolved_via=article.resolved_via,
            source_words=scope,
            prompt_tokens=sum(x.prompt_tokens or 0 for x in gens),
            prefill_tok_s=gens[0].prefill_tok_s,
            decode_tok_s=gens[-1].decode_tok_s,
            llm_seconds=round(sum(x.duration_s for x in gens), 3),
            section_id=sec.id if sec else None,
            section_title=sec.title if sec else None,
            include_subsections=include_subsections if sec else True,
            sections=[s.to_dict() for s in extracted.sections],
        )
