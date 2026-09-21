# Design notes

How `offline-kiwix-summariser` works and why it is shaped the way it is. For
setup and usage see [README.md](README.md).

---

## Architecture

```
client ──POST /v1/summarise──> FastAPI ──> JobRunner (asyncio.Queue)
                │                              │
                └──202 + job_id (~10ms)        └─> worker(s)
                                                      │
                        GET /v1/jobs/{id} <───────────┤
                                                      ├─ KiwixClient  (OPDS + content)
                                                      ├─ extract.py   (HTML -> prose)
                                                      ├─ OllamaClient (draft + corrections)
                                                      └─ fit_to_budget (deterministic trim)
```

Three deliberately separate concerns:

- **`kiwix.py`** knows how to find an article and nothing about summarising.
- **`extract.py`** turns HTML into prose and a section index using only the
  stdlib parser. No BeautifulSoup, no lxml — one less thing to break offline.
- **`summarise.py`** owns length control and orchestration; it never parses HTML
  or builds URLs itself.

Jobs are the seam. Nothing in the HTTP layer blocks on GPU time.

## Why asynchronous

A single summary takes 20–30 seconds on a consumer GPU. Holding an HTTP
connection open that long invites proxy timeouts, retries that duplicate GPU
work, and a poor experience for anyone queueing behind it. So:

- `POST /v1/summarise` returns **202** with a job id in ~10ms.
- The job id is a 128-bit random hex token and acts as the **capability** for
  reading the result. There is no auth layer; possession of the token is the
  authorization.
- `GET /v1/jobs/{id}` reports `queued` → `processing` → `ready` / `failed`, with
  a `queue_position` while waiting.
- `DELETE` withdraws a job that has not started. A running job cannot be
  cancelled mid-inference — the GPU call is not interruptible.

The default worker count is **1**. This is not a limitation of the design but of
the hardware: one GPU serialises inference, so extra workers only contend for
VRAM and make every request slower. The queue absorbs bursts (capacity 64, then
`429`) rather than failing them.

Job state is in-process memory. A restart drops queued and finished jobs. For a
personal offline service that is the right trade; `JobRunner` is the single place
to swap in Redis or SQLite if you need durability.

## Length accuracy

Prompting alone cannot hold a word budget on a small model. `qwen3:8b` overshoots
by roughly 35% regardless of how the instruction is phrased, and plateaus near
~45 words however hard you push. So length is enforced in **two layers**:

1. **Feedback-driven re-compression.** If the draft is over budget, the model is
   re-prompted *with its previous word count* and asked to compress. Telling it
   the number it missed by converges far better than restating the target. The
   loop stops on success or on plateau — when a pass fails to shrink the text,
   the model has hit its floor and further passes only burn GPU time.
2. **Deterministic sentence-boundary trim.** Greedily pack whole sentences up to
   the upper tolerance. Sentences are the smallest unit that still reads as
   prose, so this is where the guarantee actually comes from. If the first
   sentence alone exceeds the budget, it clips on a word boundary instead.

Below `TWO_PASS_BELOW_WORDS` (120) the draft pass deliberately aims wide
(`words * 3`, minimum 120) and lets the second pass compress. Drafting a tight
target straight out of a 13k-token article loses content-selection quality — the
model spends its attention on length instead of choosing what matters. The second
pass has a tiny prompt, so its prefill is negligible.

Measured on `Alan Turing` (9,536 source words), RTX 3060:

| target | actual | delta | passes |
|---|---|---|---|
| 25 | 29 | +16.0% | 4 |
| 40 | 40 | 0.0% | 4 |
| 75 | 86 | +14.7% | 4 |
| 150 | 144 | −4.0% | 1 |
| 300 | 300 | 0.0% | 1 |

5/5 within tolerance, mean absolute error 6.9%. Before the trim layer existed this
was **2/5** with a mean error of 27% and a worst case of **+92%**.

Tolerance is ±15% with a 3-word floor, so short targets are not impossible:
`_tolerance(25)` yields 21–29 rather than 21.25–28.75.

**These figures move a few points run to run.** Sampling is non-deterministic at
`temperature 0.2`; an earlier run of identical code measured a 3.9% mean. Treat
the table as typical, not exact. What is stable is the *tolerance hit rate*,
because the trim layer enforces the upper bound mechanically rather than hoping
the model complies.

`result.actual_words`, `result.target_words`, `result.word_delta`,
`result.trimmed` and `result.within_tolerance` are always reported, so a caller
can decide what to do with an out-of-tolerance result rather than having it
silently accepted.

## Performance

On an RTX 3060 (12 GB) with `qwen3:8b` Q4_K_M:

- prefill ~1,500–1,800 tok/s, decode ~40–59 tok/s
- typical Wikipedia article: 7.4k–14k prose words ≈ 10k–19k prompt tokens
- **end-to-end 17–31s per request**; fetch + extract is under 0.2s of that
- cold model load ~13s, which is why `OLLAMA_KEEP_ALIVE=-1` is the default

A whole article fits in **one** context at `num_ctx=24576`, so no chunking or
map-reduce is needed. That was the main open design question and settling it
empirically first is what kept the implementation simple. Section-scoped requests
are cheaper still — a few hundred to a couple of thousand words of source — and
measured 12–19s.

## Sections

Every result carries a `sections` index: the article's headings in document order.

```json
{"id": 7, "title": "Cryptanalysis", "level": 3, "words": 1123, "own_words": 1123}
{"id": 8, "title": "Bombe",         "level": 3, "words": 459,  "own_words": 196}
```

`words` counts the heading plus everything nested under it; `own_words` counts
that heading's paragraphs alone. **The gap between them is what tells you a
heading is a container** rather than a leaf.

The list is not truncated. Even a 55-section article serialises to about 5 KB,
and it is the index a second request picks from — hiding entries would make valid
ids unresolvable by a client that only has the response body.

`section` accepts an `id` or a heading title. Numeric references are always ids.
Titles match **exactly** after ignoring case and punctuation — not by prefix or
substring — so `"Personal life"` and `"personal-life"` both work but `"early
life"` does not match `"Early life and education"`. This is deliberately strict:
a fuzzy heading match that summarises the wrong section is worse than an error
that lists the valid options.

### Sectioning must not change whole-article text

The parser segments prose into sections *while* producing the full text. A heading
is re-emitted as the first line of the section it opens, so the whole-article
output is byte-identical to what the flat parser produced. That invariant is
pinned by a test, because adding section support must not quietly change what a
plain whole-article request summarises.

Consequence worth knowing: a section's `own_words` includes its heading words.
That is intentional — the model sees the heading as context.

### The lead is a sibling, not the root

Prose before the first heading becomes a synthetic `(lead)` section. It carries
level 1, which naively makes it the parent of every `h2` — so "summarise section
0" would have re-summarised the entire article. `(lead)` is special-cased to be
own-words only, both in its reported `words` and in the text it yields.

### Container headings

Some headings own almost no prose because their content lives in subsections.
`"Personal life"` on Alan Turing has 2 own words and 832 in its subtree. Asking
for it with `include_subsections: false` fails, but explains itself:

```
section 'Personal life' holds only 2 words of prose at this scope. Section 16 is
a container heading whose content lives in its subsections (832 words in total);
retry with include_subsections=true.
```

`include_subsections: true` is the default because that is what "summarise the
History section" means to a reader.

### There is no separate discovery endpoint

The index travels in the result, so the intended flow is two ordinary requests:
summarise the article, read `result.sections`, then summarise one of them. A
`GET /v1/books/{key}/sections` would save the GPU time of the first request, but
it would be a second route to keep in step with `section`'s matching rules, for a
caller who is about to pay for a summary anyway.

Worth being explicit about the consequence: the second request **re-resolves its
title** through the normal path. Nothing pins it to the article the first request
found, so a vague title that matched fuzzily the first time could in principle
match differently the second. Two things cover that — every result reports
`article_title` and `resolved_via`, and `strict: true` turns a non-exact match
into an error listing near misses instead of a summary of the wrong page.

## Three Ollama defaults that fail silently

These are dangerous precisely because nothing errors.

1. **Unset `num_ctx` defaults to 4096.** A 19k-token article is truncated with no
   warning and you summarise only the opening section — the output looks
   plausible, it is just incomplete. `num_ctx` is always sent explicitly.
2. **Thinking models return an empty `response`.** Left in thinking mode the model
   spends the entire `num_predict` budget on internal reasoning and returns
   nothing, with a normal-looking `eval_count`. `think: false` is sent explicitly,
   and an empty response raises rather than returning blank text.
3. **`keep_alive: "-1"` must be an integer, not a string.** The string form is
   rejected with `time: missing unit in duration "-1"`. `coerce_keep_alive`
   converts integer-looking values and passes `"5m"`-style durations through
   untouched.

## HTML extraction traps

Naive extraction returns **zero words rather than an error**, which is what makes
these worth pinning as regression tests. Three separate bugs each produced an
empty summary with a `200 OK`:

- **Class matching must be on whole tokens.** The `<html>` root carries
  `class="skin-thumbsize-clientpref-standard"`. Substring-matching the chrome
  token `thumb` marks the entire document as skipped and discards everything.
- **`skipping` must be `any(stack)`, not `bool(stack)`.** The stack is non-empty
  as soon as `<html>` opens, so a depth-based test skips the whole page.
- **End tags must close tolerantly**, dropping back to the nearest matching open
  tag. Popping exactly one stack entry misaligns permanently on malformed HTML and
  every subsequent text node is silently discarded.

Content selection: MediaWiki/mwoffliner prose is taken from `.mw-parser-output`.
Other books do not have it — Gutenberg, wikivoyage and the StackExchange ZIMs wrap
prose differently — so the extractor falls back to `<body>`, then to the whole
document. Navboxes, infoboxes, reference lists, thumbnails, tables and citation
markers are dropped, and editorial markers like `[citation needed]` are stripped
from the text.

The OPDS catalog is parsed with `defusedxml` (stdlib `ElementTree` as fallback),
rejects DTDs and entity declarations, and caps the response at 8 MB. Catalog XML
comes from a service you run, but it describes files other people made.

Article titles are cleaned of site suffixes, so `"Alan Turing - Wikipedia"` and
`"Periodic backup of Rpi3 - Raspberry Pi Stack Exchange"` both reduce to the real
title.

## Title resolution and `strict`

Titles resolve by direct content path first (`{title}`, then `A/{title}`), then by
Kiwix search with an exact-title preference. Search ranking alone is misleading —
querying `"the enigma cipher machine"` ranks *"Polish Enigma double"* first.

By default a fuzzy hit is used and reported honestly in `result.resolved_via`
(`search:fuzzy`) and `result.article_title`. With `"strict": true` the job fails
instead of summarising the wrong article, and lists the closest matches:

```
no article titled exactly 'the enigma cipher machine' in book
'wikipedia_en_all_maxi'; closest matches: 'Polish Enigma double',
'Rotor machine', 'Cipher Bureau (Poland)', 'Enigma machine', ...
```

`resolved_via` distinguishes `path:{title}`, `path:A/{title}`, `search:exact` and
`search:fuzzy` so a caller can tell how much to trust the match.

## Configuration reference

| Variable | Default | Notes |
|---|---|---|
| `KIWIX_BASE_URL` | `http://kiwix` | **Use a routable IP.** Short hostnames only resolve in-container if it shares the host DNS search domain. |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | **Use a routable IP.** Ollama must listen on the LAN (`OLLAMA_HOST=0.0.0.0`). |
| `OLLAMA_MODEL` | `qwen3:8b` | Must already be pulled on the Ollama host. |
| `OLLAMA_NUM_CTX` | `24576` | Always sent explicitly. Lowering it silently truncates long articles. |
| `OLLAMA_NUM_CTX_LONG` | `32768` | Used above `LONG_ARTICLE_TOKENS`. |
| `LONG_ARTICLE_TOKENS` | `20000` | Threshold for switching to the long context. |
| `OLLAMA_TEMPERATURE` | `0.2` | Low, for factual grounding. Non-zero, so runs vary slightly. |
| `OLLAMA_THINK` | `false` | **Keep false.** Enabling it returns empty responses. |
| `OLLAMA_KEEP_ALIVE` | `-1` | Integer `-1` pins weights in VRAM. Use `"5m"` to reclaim it. |
| `TWO_PASS_BELOW_WORDS` | `120` | Draft wide then compress below this target. |
| `MAX_CORRECTION_PASSES` | `3` | Feedback-driven re-compression attempts. |
| `MAX_SOURCE_WORDS` | `60000` | Rejects absurdly long sources. |
| `LLM_WORKERS` | `1` | One GPU serialises inference; more workers only contend for VRAM. |
| `QUEUE_MAX` | `64` | Beyond this, submissions get `429`. |
| `JOB_TTL_SECONDS` | `3600` | Terminal jobs are reaped after this. |
| `HTTP_TIMEOUT_SECONDS` | `60` | Kiwix requests. Generous because the first read of a page from a cold ZIM is disk-bound; a warm one is milliseconds. `/healthz` uses its own 10s timeout so it stays responsive. |
| `LLM_TIMEOUT_SECONDS` | `900` | Generation ceiling; long articles need headroom. |
| `CORS_ORIGINS` | `*` | Comma-separated list. |
| `PORT` | `8080` | Host port (docker-compose only). |

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install pytest pytest-asyncio

PYTHONPATH=. .venv/bin/python -m pytest tests/ -q    # 76 tests, no network or GPU
```

The suite is offline by design. `tests/test_api.py` stubs Kiwix and the
summariser and drives the app through `httpx.ASGITransport` on a single event
loop — `TestClient` would spin up its own loop and strand the runner's worker
tasks on the wrong one.

Live checks against a running instance:

```bash
.venv/bin/python scripts/smoke.py                    # full API + error paths
.venv/bin/python scripts/length_sweep.py 25 40 75 150 300
.venv/bin/python scripts/get_summary.py "Alan Turing" -n 70 -s Cryptanalysis -b <book>
```

`smoke.py` submits two jobs back-to-back so the second has to wait its turn, then
checks every error path, the section index invariants, that a section-scoped
result reports the section's word count rather than the article's, and that a
container heading fails with an actionable hint.

```
app/
  config.py      environment-injected settings
  kiwix.py       catalog discovery, title resolution, search
  extract.py     HTML -> prose and section index (stdlib parser only)
  llm.py         async Ollama client
  summarise.py   orchestration + length control
  jobs.py        job store and worker pool
  main.py        FastAPI routes
scripts/
  get_summary.py    CLI client
  smoke.py          live end-to-end + error-path checks
  length_sweep.py   word-count accuracy measurement
tests/              76 offline tests
```

## Limitations

- **Job state is in-process memory.** A restart drops queued and completed jobs.
- **Single-user throughput.** At 17–31s per request, several concurrent users
  queue rather than run in parallel.
- **Very small targets** (≤25 words) land slightly under, because whole sentences
  are the smallest unit that still reads as prose.
- **Section ids are document-order positions, not stable identifiers.** They are
  valid for the article as currently served. A ZIM update that adds or reorders a
  heading shifts every id after it, so a client that caches ids across updates
  should prefer titles, or re-read `result.sections`.
- **No authentication.** The job token is the only capability, and the service
  binds to whatever port you publish. Keep it on a trusted network.
- **Summaries can contain model errors.** They are grounded in article text,
  which is far safer than open-ended generation, but a small local model can still
  be wrong. `result.source_url` is returned so a UI can link to the original.
