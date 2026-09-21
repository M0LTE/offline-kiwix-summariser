# offline-kiwix-summariser

Ask for any article in your local [Kiwix](https://kiwix.org) library and get back
a summary of exactly the length you asked for — written by a local LLM. Nothing
leaves your network: no internet, no API keys, no cloud.

```
"Alan Turing, in 150 words"  →  a 151-word summary, 17 seconds later
```

It works with **any** ZIM file you add to Kiwix, not just Wikipedia. Books are
discovered automatically at request time.

---

## What you need

| | |
|---|---|
| A machine running **Kiwix** | serving at least one ZIM file |
| A machine running **Ollama** | with a model pulled, e.g. `ollama pull qwen3:8b` |
| A GPU | ~8 GB VRAM for a 7B–8B model at 4-bit. CPU works, just slowly |
| **Docker** | on the machine that will run this service |

Verified on an RTX 3060 (12 GB) with `qwen3:8b`: about 15–30 seconds for a full
Wikipedia article, a little less for a single section.

---

## Setup

**1. Clone and configure**

```bash
git clone https://github.com/M0LTE/offline-kiwix-summariser.git
cd offline-kiwix-summariser
cp .env.example .env
```

**2. Edit `.env`** — only the first two lines are required:

```ini
KIWIX_BASE_URL=http://192.0.2.10          # your Kiwix host
OLLAMA_BASE_URL=http://192.0.2.20:11434   # your Ollama host
OLLAMA_MODEL=qwen3:8b                     # must already be pulled
```

Use **IP addresses**, not names like `http://kiwix`. A short hostname only
resolves inside the container if it shares your host's DNS search domain; an IP
always works. (There is a commented-out `dns:` block in `docker-compose.yml` if
you would rather use names.)

**3. Start it**

```bash
docker compose up -d --build
```

**4. Check it can see both services**

```bash
curl -s localhost:8080/healthz | python3 -m json.tool
```

You want `"status": "ok"` with `kiwix.ok` and `ollama.ok` both `true`. If either
is `false`, the `detail` field says why — almost always an unreachable address or
a model that has not been pulled.

---

## Usage

All the output below is real, captured from an RTX 3060 running `qwen3:8b`. The
`192.0.2.10` in the URLs is the placeholder Kiwix address from `.env.example`;
yours will be whatever you put in `.env`.

### 1. Summarise an article

```bash
python scripts/get_summary.py "Alan Turing" -n 150 -b wikipedia_en_all_maxi
```

```
job 41d8b77e551b42e58199f8374b550cd3  ->  http://127.0.0.1:8080/v1/jobs/41d8b77e…
  [  0.0s] processing
  [ 16.6s] ready

Alan Turing, an English mathematician and computer scientist, was pivotal in
developing theoretical computer science with the Turing machine. Born in 1912, he
graduated from King's College, Cambridge, and earned a doctorate from Princeton.
During World War II, he worked at Bletchley Park, leading efforts to crack German
Enigma codes, significantly aiding the Allies. Post-war, he contributed to early
computer design and explored mathematical biology, predicting chemical reaction
patterns. Despite his achievements, his work was classified due to the Official
Secrets Act. In 1952, he was prosecuted for homosexuality, leading to chemical
castration. He died in 1954, presumed a suicide, though some suggest accidental
poisoning. A 2013 pardon and a 2017 law retroactively pardoned men convicted
under historical anti-homosexual laws. Turing's legacy includes the Turing test,
the Turing machine, and numerous honors, including a £50 note and statues. His
contributions to computing and cryptography remain foundational, with his work
recognized globally.

-- Alan Turing | 151/150 words | from 9536 | 1 pass(es) | 16.5s
   http://192.0.2.10/content/wikipedia_en_all_maxi/Alan_Turing
```

Read the trailer line as a receipt: **151 words delivered against a target of
150**, compressed **from 9,536** words of article, in **1** generation pass, in
**16.5 seconds**. `--json` gives you the same thing as structured data.

If your Kiwix serves only one book you can drop `-b` entirely.

### 2. Summarise one section instead

Ask for the full JSON once and the result carries a `sections` index for the
article it used:

```bash
python scripts/get_summary.py "Alan Turing" -n 150 -b wikipedia_en_all_maxi --json
```

```json
"sections": [                                              // 33 entries; excerpt
  {"id": 0, "title": "(lead)",                   "level": 1, "words": 543,  "own_words": 543},
  {"id": 1, "title": "Early life and education", "level": 2, "words": 2013, "own_words": 4},
  {"id": 6, "title": "Career and research",      "level": 2, "words": 3870, "own_words": 65},
  {"id": 7, "title": "Cryptanalysis",            "level": 3, "words": 1123, "own_words": 1123},
  {"id": 8, "title": "Bombe",                    "level": 3, "words": 459,  "own_words": 196}
]
```

`words` is the heading plus everything nested under it; `own_words` is that
heading's own paragraphs. **A big gap between the two means the heading is a
container** — `Early life and education` owns 4 words and delegates 2,009 to its
subsections.

Now ask for one of them, by name or by number:

```bash
python scripts/get_summary.py "Alan Turing" -n 70 -s Cryptanalysis -b wikipedia_en_all_maxi
python scripts/get_summary.py "Alan Turing" -n 70 -s 7             -b wikipedia_en_all_maxi
```

```
  [ 14.6s] ready

During World War II, Alan Turing played a pivotal role at Bletchley Park in
breaking German ciphers, notably the Enigma machine. Collaborating with Dilly
Knox, he developed a broader solution to Enigma, surpassing the Polish method.
Turing created the bombe, an advanced machine, and made five major cryptanalytical
advances, including deducing the German navy's indicator procedure and developing
Banburismus and Turingery. His work shortened the war, though its exact impact is
debated.

-- Alan Turing | section 7: Cryptanalysis | 72/70 words | from 1123 | 3 pass(es) | 14.2s
```

Only 1,123 words went to the model instead of 9,536, so it is quicker (14s vs
17s) and stays on topic. Add `--own-only` to exclude subsections — useful for a
container heading you want summarised from its own paragraphs alone.

### 3. It is not Wikipedia-specific

Any ZIM in your Kiwix works. This one is a StackExchange archive:

```bash
python scripts/get_summary.py "How likely is SD card corruption?" -n 60 \
    -b raspberrypi.stackexchange.com_en_all
```

```
  [ 16.6s] ready

SD card corruption risks arise from power disconnection during writes, influenced
by factors like activity level, mount type, quality, environment, and hardware.
While rare, corruption can occur with abrupt power cuts. Users report mixed
experiences, with some facing issues due to poor power or cards, while others
experience no problems. Proper shutdowns reduce risks, but prediction remains
uncertain due to multiple variables.

-- How likely is SD card corruption? | 62/60 words | from 2274 | 2 pass(es) | 16.5s
```

Not sure of an exact title? Search the book first:

```bash
curl -s "localhost:8080/v1/books/raspberrypi.stackexchange.com_en_all/search?q=sd+card+corrupt"
```

```json
{"count": 3, "results": [
  {"title": "How likely is SD card corruption?", "url": "/content/…/questions/43212/…"},
  {"title": "Raspberry Pi corrupts the SD Card often", "url": "/content/…/questions/8707/…"},
  {"title": "Is SD card corruption still an issue?", "url": "/content/…/questions/14657/…"}]}
```

### 4. When a title is ambiguous

Without `strict`, a title that is not an exact match falls back to Kiwix search —
and search ranking is not what you would guess. Asking for *"the enigma cipher
machine"* ranks **"Polish Enigma double"** first, not "Enigma machine". `--strict`
turns that into an error listing the candidates:

```bash
python scripts/get_summary.py "the enigma cipher machine" -n 50 --strict -b wikipedia_en_all_maxi
```

```
  [ 15.6s] failed
{
 "kind": "ArticleNotFound",
 "detail": "no article titled exactly 'the enigma cipher machine' in book
            'wikipedia_en_all_maxi'; closest matches: 'Polish Enigma double',
            'Rotor machine', 'Cipher Bureau (Poland)', 'Enigma machine',
            'Zygalski sheets'"
}
```

Either way you are never left guessing: every result reports `article_title` and
`resolved_via` (`path:{title}`, `path:A/{title}`, `search:exact`, `search:fuzzy`).

### The HTTP API

Interactive docs at **`http://localhost:8080/docs`**.

A summary takes 15–30 seconds, so the API never makes you wait on it. You submit,
get a job id back immediately, then poll:

```bash
curl -s -X POST localhost:8080/v1/summarise \
  -H 'Content-Type: application/json' \
  -d '{"book":"wikipedia_en_all_maxi","title":"Alan Turing","words":70,
       "section":"Cryptanalysis"}'
```

Returns in ~10ms with `202` and a `Location` header:

```json
{"job_id":"fffd034f0a9f4ff5b9fe3e6ec4652f0d","status":"queued",
 "status_url":"/v1/jobs/fffd034f0a9f4ff5b9fe3e6ec4652f0d",
 "poll_after_seconds":2.0,"queue_position":0}
```

```bash
curl -s localhost:8080/v1/jobs/fffd034f0a9f4ff5b9fe3e6ec4652f0d
```

`status` walks `queued → processing → ready` (or `failed` / `cancelled`), and the
finished job looks like this:

```json
{
 "job_id": "fffd034f0a9f4ff5b9fe3e6ec4652f0d",
 "status": "ready",
 "request": {"book": "wikipedia_en_all_maxi", "title": "Alan Turing", "words": 70,
             "strict": false, "section": "Cryptanalysis", "include_subsections": true},
 "processing_seconds": 14.24,
 "result": {
  "summary": "During World War II, Alan Turing played a pivotal role at Bletchley Park…",
  "target_words": 70,       "actual_words": 72,     "word_delta": 2,
  "within_tolerance": true, "trimmed": true,        "passes": 3,
  "article_title": "Alan Turing",
  "book_key": "wikipedia_en_all_maxi",
  "source_url": "http://192.0.2.10/content/wikipedia_en_all_maxi/Alan_Turing",
  "resolved_via": "path:{title}",
  "source_words": 1123,     "prompt_tokens": 2132,
  "prefill_tok_s": 2007.1,  "decode_tok_s": 58.2,   "llm_seconds": 14.089,
  "section_id": 7,          "section_title": "Cryptanalysis",
  "include_subsections": true,
  "sections": [ …33 entries… ]
 },
 "error": null
}
```

| Method | Path | What it does |
|---|---|---|
| `GET` | `/healthz` | Can we reach Kiwix and Ollama? |
| `GET` | `/v1/books` | List every publication Kiwix serves |
| `GET` | `/v1/books/{key}/search?q=` | Find exact article titles |
| `POST` | `/v1/summarise` | Start a summary job (`202` + job id) |
| `GET` | `/v1/jobs/{job_id}` | Poll a job; carries the result when ready |
| `DELETE` | `/v1/jobs/{job_id}` | Cancel a job that has not started yet |
| `GET` | `/v1/stats` | Queue counters |

`POST /v1/summarise` accepts:

```json
{"book": "wikipedia_en_all_maxi", "title": "Alan Turing", "words": 150,
 "section": null, "include_subsections": true, "strict": false}
```

`words` may be 10–4000. `section` may be an id or a heading title. The job id is a
128-bit random token and acts as the capability for reading that result.

Failures come back on the job, not on the submit — the article has to be fetched
before the service can know whether it exists. `error.kind` is the exception name
(`BookNotFound`, `ArticleNotFound`, `ArticleTooLong`, `SectionNotFound`,
`NothingToSummarise`, `ModelNotFound`) and `error.detail` says what to do about
it. Asking for a container heading with subsections excluded, for example:

```json
{"kind": "NothingToSummarise",
 "detail": "section 'Early life and education' holds only 4 words of prose at this
            scope. Section 1 is a container heading whose content lives in its
            subsections (2013 words in total); retry with include_subsections=true."}
```

### Adding another publication

Drop a `.zim` file into Kiwix. It appears in `GET /v1/books` immediately — no
restart, no code change, no configuration.

---

## Things worth knowing

**How accurate is the length?** Within ±15% of your target, and usually much
closer. A small model cannot count words reliably, so the service drafts, then
re-compresses with feedback, then trims at a sentence boundary to land on the
number. Very short targets (25 words or fewer) tend to come in a little under,
because whole sentences are the smallest unit that still reads as prose. The
result always reports `actual_words`, `target_words` and `within_tolerance`, so
you never have to guess.

**Is it one request at a time?** Yes. A single consumer GPU serialises inference,
so extra requests queue rather than fail. The queue holds 64; beyond that you get
`429`. Set `LLM_WORKERS` higher only if you have the VRAM for it.

**What if my title is vague?** Titles resolve by direct path first, then by Kiwix
search, and search ranking is misleading — see
[when a title is ambiguous](#4-when-a-title-is-ambiguous). Every result names the
article it actually used, so you are never left with a summary of the wrong page
and no way to tell.

**Will the summaries be correct?** They are grounded in the article text, which
is far safer than open-ended generation, but a small local model can still get
things wrong. `result.source_url` is returned so you can link back to the
original.

**Do jobs survive a restart?** No. Job state lives in process memory; a restart
drops queued and completed jobs. Fine for a personal service — see `DESIGN.md` if
you need durability.

**Why is the first request slow?** Cold model load costs ~13 seconds. The default
`OLLAMA_KEEP_ALIVE=-1` pins weights in VRAM so only the first request after a
restart pays it.

---

## Configuration

Everything is an environment variable and `.env.example` is fully annotated. The
defaults work, so most people only need these three:

| Variable | Default | Notes |
|---|---|---|
| `KIWIX_BASE_URL` | `http://kiwix` | **Use a routable IP.** |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | **Use a routable IP.** |
| `OLLAMA_MODEL` | `qwen3:8b` | Must already be pulled on the Ollama host. |

Two settings look harmless but are not: leave `OLLAMA_THINK=false` and do not
lower `OLLAMA_NUM_CTX` from `24576`. Both cause failures that produce *no error* —
see [DESIGN.md](DESIGN.md).

The remaining variables (context sizes, queue depth, worker count, timeouts,
length-control tuning, `PORT`) are documented in
[DESIGN.md → Configuration reference](DESIGN.md#configuration-reference).

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `healthz` says `kiwix.ok: false` | Wrong `KIWIX_BASE_URL`. Test with `curl $KIWIX_BASE_URL/catalog/v2/entries` from the Docker host. |
| `healthz` says `ollama.ok: false` | Wrong `OLLAMA_BASE_URL`, or Ollama is not listening on the LAN. Set `OLLAMA_HOST=0.0.0.0` on that machine. |
| `model 'X' not found` | The model is not pulled on the Ollama host. Run `ollama pull X` there. |
| Container cannot resolve `http://kiwix` | Use an IP address instead, or uncomment the `dns:` block in `docker-compose.yml`. |
| Job fails with `BookNotFound` | That ZIM is not loaded in Kiwix. Check `GET /v1/books`. |
| Job fails with `SectionNotFound` | No heading matched. The `detail` lists the first few real titles; `result.sections` has all of them. |
| Job fails with `ArticleTooLong` | Source exceeds `MAX_SOURCE_WORDS`. Raise it if you have the VRAM. |
| Job fails with `ReadTimeout` | Kiwix or Ollama did not answer within `HTTP_TIMEOUT_SECONDS` / `LLM_TIMEOUT_SECONDS`. Seen once on a cold ZIM read; retry, or raise the timeout. |
| Summary came back empty | You enabled `OLLAMA_THINK`. Turn it back off — see `DESIGN.md`. |

---

## More detail

[`DESIGN.md`](DESIGN.md) covers how it works and why: the async job pattern, how
length accuracy is enforced (with measurements), the section-scoping design,
three Ollama defaults that fail **silently**, the HTML-extraction traps
that return zero words instead of an error, the full configuration reference, the
project layout, running the tests, and the known limitations.

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q    # 76 tests, no network or GPU
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
