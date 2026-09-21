"""Regression tests for the extractor.

Each of the first three cases is a bug that silently produced zero words
rather than an error, which is why they are pinned here explicitly.
"""
from app.extract import content_region, extract, is_chrome


def test_substring_chrome_token_does_not_poison_document():
    # "skin-thumbsize-..." contains the substring "thumb". Matching on
    # substrings marked <html> as chrome and dropped the entire article.
    html = (
        '<html class="client-nojs skin-thumbsize-clientpref-standard">'
        "<body><div class=\"mw-parser-output\">"
        "<p>Real prose that must survive.</p>"
        "</div></body></html>"
    )
    assert is_chrome("skin-thumbsize-clientpref-standard") is False
    assert is_chrome("thumb") is True
    out = extract(html)
    assert "Real prose that must survive." in out.text


def test_tolerant_close_on_malformed_html():
    # A stray </span> must not misalign the stack and swallow later text.
    html = (
        '<body><div class="mw-parser-output">'
        "<p>First paragraph.</p></span><p>Second paragraph.</p>"
        "</div></body>"
    )
    out = extract(html)
    assert "First paragraph." in out.text
    assert "Second paragraph." in out.text


def test_skipping_is_any_of_stack_not_depth():
    # A non-empty stack is not the same as an active skip.
    html = (
        '<body><div class="mw-parser-output">'
        "<div><p>Inside a plain nested div.</p></div>"
        "</div></body>"
    )
    assert "Inside a plain nested div." in extract(html).text


def test_chrome_regions_are_dropped():
    html = """<body><div class="mw-parser-output">
      <p>Keep this sentence.</p>
      <table class="infobox"><tr><td>born 1912 infobox junk</td></tr></table>
      <div class="navbox"><div>navbox junk</div></div>
      <div class="thumb"><div class="thumbinner">caption junk</div></div>
      <p>Also keep this.<sup class="reference">[3]</sup></p>
      <div class="reflist"><ol><li>citation junk</li></ol></div>
    </div></body>"""
    out = extract(html)
    assert "Keep this sentence." in out.text
    assert "Also keep this." in out.text
    for junk in ("infobox junk", "navbox junk", "caption junk", "citation junk"):
        assert junk not in out.text
    assert "[3]" not in out.text


def test_mw_parser_output_preferred_over_body():
    html = (
        "<body><div id='site-notice'>Banner chrome</div>"
        "<div class='mw-parser-output'><p>Article body.</p></div>"
        "<div id='footer'>Footer chrome</div></body>"
    )
    out = extract(html)
    assert "Article body." in out.text
    assert "Banner chrome" not in out.text
    assert "Footer chrome" not in out.text


def test_falls_back_to_body_for_non_mediawiki_books():
    # Gutenberg / wikivoyage / stackexchange ZIMs have no mw-parser-output.
    html = (
        "<html><head><style>p{color:red}</style><title>Chapter One</title></head>"
        "<body><header>Site nav</header><main><p>Plain book prose here.</p></main>"
        "<script>var x=1;</script></body></html>"
    )
    out = extract(html)
    assert "Plain book prose here." in out.text
    assert "p{color:red}" not in out.text
    assert "var x=1" not in out.text


def test_content_region_falls_back_to_whole_document():
    assert "orphan" in content_region("<div>orphan text</div>")


def test_headings_collected_in_order():
    html = (
        '<body><div class="mw-parser-output">'
        "<h2>Early life</h2><p>a</p><h2>Career</h2><p>b</p><h3>Subsection</h3>"
        "</div></body>"
    )
    out = extract(html)
    assert out.headings[:3] == ["Early life", "Career", "Subsection"]


def test_title_from_title_tag_strips_wikipedia_suffix():
    html = (
        "<html><head><title>Alan Turing - Wikipedia</title></head>"
        '<body><div class="mw-parser-output"><p>text</p></div></body></html>'
    )
    assert extract(html).title == "Alan Turing"


def test_title_strips_stack_exchange_site_suffix():
    html = (
        "<html><head><title>Periodic backup of Rpi3 - Raspberry Pi Stack Exchange"
        "</title></head><body><main><p>text</p></main></body></html>"
    )
    assert extract(html).title == "Periodic backup of Rpi3"


def test_title_falls_back_to_first_heading():
    html = (
        "<html><body><div class=\"mw-parser-output\">"
        "<h2>Photosynthesis</h2><p>text</p></div></body></html>"
    )
    assert extract(html).title == "Photosynthesis"


def test_word_count_and_whitespace_normalisation():
    html = (
        '<body><div class="mw-parser-output">'
        "<p>one   two\tthree</p>\n\n\n<p>four</p>"
        "</div></body>"
    )
    out = extract(html)
    assert out.word_count == 4
    assert "   " not in out.text


def test_editorial_markers_stripped():
    html = (
        '<body><div class="mw-parser-output">'
        "<p>Claim [citation needed] and more [edit].</p>"
        "</div></body>"
    )
    out = extract(html)
    assert "citation needed" not in out.text
    assert "[edit]" not in out.text
    assert "Claim" in out.text


# --- sections ---------------------------------------------------------------
NESTED = (
    "<html><head><title>Alan Turing - Wikipedia</title></head><body>"
    '<div class="mw-parser-output">'
    "<p>Lead one. Lead two.</p>"
    "<h2>Early life</h2><p>Born in London.</p>"
    "<h3>Family</h3><p>Father Julius.</p>"
    "<h3>School</h3><p>Attended Sherborne.</p>"
    "<h2>Career</h2><p>Worked at Bletchley.</p>"
    "<h3>Cryptanalysis</h3><p>Broke Enigma.</p>"
    "</div></body></html>"
)

TITLES = ["(lead)", "Early life", "Family", "School", "Career", "Cryptanalysis"]


def test_section_index_and_levels():
    out = extract(NESTED)
    assert [s.title for s in out.sections] == TITLES
    # Ids are contiguous from zero and match list position, which is what lets
    # a numeric `section` reference be resolved positionally.
    assert [s.id for s in out.sections] == [0, 1, 2, 3, 4, 5]
    assert [s.level for s in out.sections] == [1, 2, 3, 3, 2, 3]


def test_subtree_word_counts_include_nested_subsections():
    out = extract(NESTED)
    by_title = {s.title: s for s in out.sections}
    # "Early life" owns 5 words (heading + "Born in London.") and has two
    # 3-word subsections under it.
    assert by_title["Early life"].own_words == 5
    assert by_title["Early life"].words == 11
    assert by_title["Career"].own_words == 4
    assert by_title["Career"].words == 7
    # A leaf section has no subtree beyond itself.
    assert by_title["Cryptanalysis"].words == by_title["Cryptanalysis"].own_words == 3


def test_lead_section_is_a_sibling_not_the_root():
    # The lead carries a synthetic level 1. Treating it as the parent of every
    # h2 would make "summarise section 0" re-summarise the whole article.
    out = extract(NESTED)
    lead = out.sections[0]
    assert lead.words == lead.own_words == 4
    assert out.section_text(lead) == out.section_text(lead, include_subsections=False)
    assert "Broke Enigma" not in out.section_text(lead)


def test_section_text_subtree_versus_own():
    out = extract(NESTED)
    early = out.find_section("Early life")
    subtree = out.section_text(early)
    own = out.section_text(early, include_subsections=False)
    assert "Father Julius." in subtree and "Attended Sherborne." in subtree
    assert "Father Julius." not in own
    assert "Born in London." in own
    # A sibling h2 must not leak into the subtree.
    assert "Worked at Bletchley." not in subtree


def test_full_text_is_unchanged_by_section_splitting():
    # Sectioning must not alter the whole-article text: the flat summarise path
    # has to keep producing exactly what it did before.
    out = extract(NESTED)
    for prose in ("Lead one.", "Born in London.", "Father Julius.",
                  "Attended Sherborne.", "Worked at Bletchley.", "Broke Enigma."):
        assert prose in out.text
    for heading in TITLES[1:]:
        assert heading in out.text
    assert out.word_count == sum(s.own_words for s in out.sections)


def test_empty_lead_is_dropped_so_ids_stay_contiguous():
    html = (
        '<body><div class="mw-parser-output">'
        "<h2>Only</h2><p>Text here.</p>"
        "</div></body>"
    )
    out = extract(html)
    assert [s.title for s in out.sections] == ["Only"]
    assert out.sections[0].id == 0


def test_find_section_by_id_and_by_loose_title():
    out = extract(NESTED)
    assert out.find_section(4).title == "Career"
    assert out.find_section("4").title == "Career"
    assert out.find_section(0).title == "(lead)"
    assert out.find_section("(lead)").id == 0
    # Case and punctuation are ignored.
    assert out.find_section("early life").title == "Early life"
    assert out.find_section("  EARLY-LIFE  ").title == "Early life"
    # Out of range and unknown titles resolve to None rather than raising.
    assert out.find_section(99) is None
    assert out.find_section(-1) is None
    assert out.find_section("no such heading") is None
