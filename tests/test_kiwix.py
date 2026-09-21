"""Kiwix client unit tests (no network)."""
from app.kiwix import _norm, parse_search_results

# Trimmed from a real kiwix-serve response.
SEARCH_PAGE = """
<html><head><style>.results a { font-size: 110%; }</style><title>Search: x</title></head>
<body bgcolor="white">
  <div class="header">Results <b>1-25</b> of <b>4,000</b> for <b>"x"</b></div>
  <div class="results">
    <ul>
      <li>
        <a href="/content/wikipedia_en_all_maxi/List_of_things_named_after_Alan_Turing">
          List of things named after Alan Turing
        </a>
        <cite>...<b>Alan</b> Turing (1912-1954), a pioneer</cite>
      </li>
      <li>
        <a href="/content/wikipedia_en_all_maxi/Alan_Turing">Alan Turing</a>
        <cite>...mathematician</cite>
      </li>
      <li>
        <a href="/content/book_x/Highest_Voted">Highest Voted &apos;backup&apos; Questions</a>
      </li>
      <li>
        <a href="/search?books.name=y">Not a content link</a>
      </li>
    </ul>
  </div>
  <div class="footer"><a href="/foo">footer</a></div>
</body></html>
"""


def test_parses_content_links_in_order():
    out = parse_search_results(SEARCH_PAGE, limit=10)
    urls = [r.url for r in out]
    assert urls[0].endswith("/List_of_things_named_after_Alan_Turing")
    assert urls[1].endswith("/Alan_Turing")


def test_skips_non_content_links_and_footer():
    out = parse_search_results(SEARCH_PAGE, limit=10)
    assert all(r.url.startswith("/content/") for r in out)
    assert not any("footer" in r.url for r in out)


def test_collapses_whitespace_in_titles():
    out = parse_search_results(SEARCH_PAGE, limit=10)
    assert out[0].title == "List of things named after Alan Turing"


def test_unescapes_html_entities_in_titles():
    # Regression: an earlier version shadowed the html module with a local
    # variable and raised "'str' object has no attribute 'unescape'".
    out = parse_search_results(SEARCH_PAGE, limit=10)
    titles = [r.title for r in out]
    assert "Highest Voted 'backup' Questions" in titles
    assert not any("&apos;" in t for t in titles)


def test_limit_is_respected():
    assert len(parse_search_results(SEARCH_PAGE, limit=1)) == 1
    assert len(parse_search_results(SEARCH_PAGE, limit=2)) == 2


def test_strips_inline_highlight_tags_from_titles():
    page = '<div class="results"><a href="/content/b/T">Al<b>an</b> Turing</a></div>'
    assert parse_search_results(page, 5)[0].title == "Alan Turing"


def test_empty_and_garbage_input():
    assert parse_search_results("", 10) == []
    assert parse_search_results("<html>no results here</html>", 10) == []


def test_norm_treats_underscores_hyphens_and_case_as_equal():
    assert _norm("Alan_Turing") == _norm("alan turing")
    assert _norm("Spider-Man") == _norm("spider man")
    assert _norm("  Alan   Turing ") == _norm("alan turing")
    assert _norm("Alan Turing") != _norm("Alanine Turing")
