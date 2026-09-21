"""HTTP contract tests.

Kiwix and the summariser are stubbed, so these exercise routing and validation
without a GPU or a ZIM. Everything runs on one event loop via ASGITransport --
TestClient would spin up its own loop and strand the runner's worker tasks on
the wrong one.
"""
import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.extract import extract
from app.main import State, app, get_state
from app.summarise import SummaryResult

NESTED = (
    "<html><head><title>Alan Turing - Wikipedia</title></head><body>"
    '<div class="mw-parser-output">'
    "<p>Lead one. Lead two.</p>"
    "<h2>Early life</h2><p>Born in London.</p>"
    "<h3>Family</h3><p>Father Julius.</p>"
    "<h2>Career</h2><p>Worked at Bletchley.</p>"
    "</div></body></html>"
)

BOOK_KEY = "wikipedia_en_all_maxi"
SOURCE_URL = f"http://kiwix/content/{BOOK_KEY}/A/Alan_Turing"
SECTION_TITLES = ["(lead)", "Early life", "Family", "Career"]


class StubSummariser:
    def __init__(self) -> None:
        self.run_calls: list[dict] = []

    @staticmethod
    def _result(**kw) -> SummaryResult:
        base = dict(
            summary="A short summary.", target_words=150, actual_words=3, passes=1,
            trimmed=True, within_tolerance=False, article_title="Alan Turing",
            book_key=BOOK_KEY, source_url=SOURCE_URL,
            resolved_via="path:A/{title}", source_words=18, prompt_tokens=10,
            prefill_tok_s=1.0, decode_tok_s=2.0, llm_seconds=0.5,
            sections=[s.to_dict() for s in extract(NESTED).sections],
        )
        base.update(kw)
        return SummaryResult(**base)

    async def run(self, **kw) -> SummaryResult:
        self.run_calls.append(kw)
        return self._result()


@pytest.fixture
def st():
    s = State(get_settings())
    s.summariser = StubSummariser()
    app.dependency_overrides[get_state] = lambda: s
    return s


@pytest.fixture
async def client(st):
    await st.runner.start()
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            yield c
    finally:
        await st.runner.stop()
        app.dependency_overrides.clear()


async def wait_terminal(client: AsyncClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        view = (await client.get(f"/v1/jobs/{job_id}")).json()
        if view["status"] in ("ready", "failed"):
            assert view["status"] == "ready", view
            return view
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"job {job_id} never finished: {view}")
        await asyncio.sleep(0.02)


async def submit(client: AsyncClient, **overrides) -> str:
    payload = {"book": BOOK_KEY, "title": "Alan Turing", "words": 150, **overrides}
    r = await client.post("/v1/summarise", json=payload)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


# --- the async job contract -------------------------------------------------
async def test_summarise_returns_202_with_a_poll_url(client):
    r = await client.post("/v1/summarise", json={
        "book": BOOK_KEY, "title": "Alan Turing", "words": 150,
    })
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert body["status_url"] == f"/v1/jobs/{body['job_id']}"
    assert body["poll_after_seconds"] > 0
    assert r.headers["location"].endswith(body["job_id"])


async def test_ready_view_carries_the_result(client, st):
    view = await wait_terminal(client, await submit(client))
    assert view["result"]["summary"] == "A short summary."
    assert view["result"]["article_title"] == "Alan Turing"
    assert view["processing_seconds"] >= 0
    assert view["error"] is None
    assert st.summariser.run_calls[0]["book_key"] == BOOK_KEY


async def test_result_lists_the_sections_a_second_request_can_pick_from(client):
    """The index is in the payload; there is no separate discovery endpoint."""
    view = await wait_terminal(client, await submit(client))
    sections = view["result"]["sections"]
    assert [s["title"] for s in sections] == SECTION_TITLES
    assert [s["id"] for s in sections] == list(range(len(sections)))
    # "Early life" owns 5 words and carries one 3-word subsection under it.
    assert sections[1]["words"] == 8
    assert sections[1]["own_words"] == 5


async def test_unknown_job_404s(client):
    r = await client.get("/v1/jobs/deadbeefdeadbeefdeadbeefdeadbeef")
    assert r.status_code == 404


async def test_summarise_rejects_a_bad_word_target(client):
    r = await client.post("/v1/summarise", json={
        "book": BOOK_KEY, "title": "Alan Turing", "words": 5,
    })
    assert r.status_code == 422


# --- the section parameter --------------------------------------------------
async def test_summarise_passes_section_through_to_the_handler(client, st):
    job_id = await submit(client, words=60, section="Early life",
                          include_subsections=False)
    await wait_terminal(client, job_id)
    assert st.summariser.run_calls[0]["section"] == "Early life"
    assert st.summariser.run_calls[0]["include_subsections"] is False


async def test_summarise_without_section_leaves_it_unset(client, st):
    await wait_terminal(client, await submit(client, words=60))
    assert st.summariser.run_calls[0]["section"] is None
    assert st.summariser.run_calls[0]["include_subsections"] is True
