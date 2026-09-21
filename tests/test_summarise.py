"""Tests for length fitting and section scoping (no network, no GPU)."""
import pytest

from app.config import Settings
from app.kiwix import Article, Book
from app.llm import Generation
from app.summarise import (
    ArticleTooLong,
    NothingToSummarise,
    SectionNotFound,
    Summariser,
    _tolerance,
    fit_to_budget,
)

BOOK_KEY = "wikipedia_en_all_maxi"


@pytest.mark.parametrize(
    "n,expected",
    [(10, (7, 13)), (25, (21, 29)), (100, (85, 115)), (150, (128, 172))],
)
def test_tolerance_scales_with_target(n, expected):
    assert _tolerance(n) == expected


def test_short_text_left_untouched():
    text = "One short sentence."
    out, trimmed = fit_to_budget(text, 100)
    assert out == text
    assert trimmed is False


def test_trims_at_sentence_boundary_not_mid_word():
    s1 = " ".join(f"w{i}" for i in range(20)) + "."
    s2 = " ".join(f"x{i}" for i in range(20)) + "."
    out, trimmed = fit_to_budget(f"{s1} {s2}", 25)
    assert trimmed is True
    assert out == s1
    assert out.endswith(".")
    assert len(out.split()) <= _tolerance(25)[1]


def test_never_exceeds_upper_tolerance():
    text = " ".join(" ".join(f"t{i}" for i in range(9)) + "." for _ in range(12))
    for target in (15, 30, 60, 120):
        out, _ = fit_to_budget(text, target)
        assert len(out.split()) <= _tolerance(target)[1], target


def test_packs_multiple_sentences_when_they_fit():
    text = "A b c. D e f. G h i. J k l."
    # hi(8) == 11, so three 3-word sentences fit and the fourth does not.
    out, _ = fit_to_budget(text, 8)
    assert out == "A b c. D e f. G h i."


def test_splits_lowercase_sentence_starts():
    # Under-splitting would degrade the trim to a mid-sentence word cut.
    s1 = " ".join(f"a{i}" for i in range(20)) + "."
    s2 = " ".join(f"b{i}" for i in range(20)) + "."
    out, trimmed = fit_to_budget(f"{s1} {s2}", 25)
    assert trimmed is True
    assert out == s1


def test_single_oversized_sentence_clipped_on_word_boundary():
    text = " ".join(f"word{i}" for i in range(60))
    out, trimmed = fit_to_budget(text, 10)
    assert trimmed is True
    assert len(out.split()) <= _tolerance(10)[1]
    assert out.endswith(".")
    # No dangling separator before the added full stop.
    assert not out.rstrip(".").endswith((",", ";", ":", "-"))


def test_empty_input_is_safe():
    out, trimmed = fit_to_budget("", 50)
    assert out == ""
    assert trimmed is False


def test_abbreviation_periods_do_not_split_sentences():
    text = "He worked for the U.S. government in Britain. Then he moved."
    out, _ = fit_to_budget(text, 50)
    assert out == text


# --- section scoping --------------------------------------------------------
# These exercise Summariser._produce with a stub LLM: the section errors below
# all fire before any generation, and the prompt assertions capture what the
# model would have been sent.
ARTICLE_HTML = (
    "<html><head><title>Alan Turing - Wikipedia</title></head><body>"
    '<div class="mw-parser-output">'
    "<p>Lead sentence about the article as a whole.</p>"
    "<h2>Personal life</h2>"
    "<h3>Family</h3><p>" + " ".join(f"word{i}." for i in range(60)) + "</p>"
    "<h2>Career</h2><p>" + " ".join(f"term{i}." for i in range(80)) + "</p>"
    "</div></body></html>"
)


class StubLLM:
    def __init__(self, reply: str = "Stub summary sentence."):
        self.reply = reply
        self.prompts: list[str] = []

    async def generate(self, prompt, num_ctx, num_predict, passed=1):
        self.prompts.append(prompt)
        return Generation(text=self.reply, prompt_tokens=1, output_tokens=1,
                          prefill_tok_s=1.0, decode_tok_s=1.0, duration_s=0.0,
                          passed=passed)


class StubKiwix:
    async def get_book(self, key: str) -> Book:
        return Book(key=key, title="Wikipedia", name=key, flavour="maxi",
                    language="eng", article_count=1, content_root=f"/content/{key}",
                    uuid=None)

    async def resolve(self, book, title, strict=False) -> Article:
        return Article(title=title, url=f"http://kiwix/content/{book.key}/A/Alan_Turing",
                       html=ARTICLE_HTML, resolved_via="path:A/{title}")


def make_summariser(reply="Stub summary sentence.", **overrides):
    llm = StubLLM(reply)
    s = Settings(**overrides) if overrides else Settings()
    return Summariser(StubKiwix(), llm, s), llm


async def run(sm, words=50, **kw):
    """One article, resolved through the stub, so the tests read like the API."""
    return await sm.run(BOOK_KEY, "Alan Turing", words, **kw)


async def test_unknown_section_raises_before_any_llm_call():
    sm, llm = make_summariser()
    with pytest.raises(SectionNotFound) as exc:
        await run(sm, section="Banburismus")
    assert "Banburismus" in str(exc.value)
    assert llm.prompts == []


async def test_container_heading_without_subsections_explains_the_fix():
    sm, llm = make_summariser()
    with pytest.raises(NothingToSummarise) as exc:
        await run(sm, section="Personal life", include_subsections=False)
    msg = str(exc.value)
    assert "Personal life" in msg
    assert "include_subsections=true" in msg
    assert llm.prompts == []


async def test_container_heading_with_subsections_succeeds():
    sm, llm = make_summariser()
    r = await run(sm, section="Personal life")
    assert r.section_title == "Personal life"
    # Subtree scope, not the 2-word heading alone.
    assert r.source_words > 50
    assert "Family" in llm.prompts[0]


async def test_section_prompt_scopes_the_model_to_one_section():
    sm, llm = make_summariser()
    await run(sm, section="Career")
    p = llm.prompts[0]
    assert '"Career" section' in p
    assert "summary of that section only" in p
    # The sibling section's prose must not be in the prompt.
    assert "word0." not in p
    assert "term0." in p


async def test_section_result_carries_scope_metadata():
    sm, llm = make_summariser()
    r = await run(sm, section=3, include_subsections=False)
    d = r.to_dict()
    assert d["section_id"] == 3
    assert d["section_title"] == "Career"
    assert d["include_subsections"] is False
    assert d["source_words"] == 81  # heading + 80 words
    assert d["book_key"] == BOOK_KEY
    assert [x["title"] for x in d["sections"]] == [
        "(lead)", "Personal life", "Family", "Career",
    ]


async def test_whole_article_leaves_section_fields_null():
    sm, llm = make_summariser()
    r = await run(sm)
    d = r.to_dict()
    assert d["section_id"] is None
    assert d["section_title"] is None
    assert "Lead sentence" in llm.prompts[0]


async def test_oversized_section_is_refused():
    sm, llm = make_summariser(max_source_words=10)
    with pytest.raises(ArticleTooLong):
        await run(sm, section="Career")
    assert llm.prompts == []


async def test_sections_are_not_truncated_in_the_payload():
    # result.sections is the index a second request picks its section from, so
    # hiding entries would make valid ids unresolvable.
    sm, llm = make_summariser()
    r = await run(sm)
    d = r.to_dict()
    assert "sections_truncated" not in d
    assert len(d["sections"]) == 4
