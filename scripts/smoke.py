"""Live end-to-end smoke test against the running API. Not part of pytest.

Usage: SUMMARISER_BASE=http://127.0.0.1:8081 python scripts/smoke.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("SUMMARISER_BASE", "http://127.0.0.1:8080").rstrip("/")


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, _lower(resp.headers), json.load(resp)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        try:
            return e.code, _lower(e.headers), json.loads(raw)
        except ValueError:
            return e.code, _lower(e.headers), {"raw": raw[:300]}


def _lower(headers) -> dict:
    """HTTP header names are case-insensitive; normalise for lookups."""
    return {k.lower(): v for k, v in (headers or {}).items()}


def submit(book, title, words):
    code, hdrs, body = req("POST", "/v1/summarise",
                           {"book": book, "title": title, "words": words})
    assert code == 202, f"expected 202, got {code}: {body}"
    assert "location" in hdrs, "202 must carry a Location header"
    assert hdrs["location"].endswith(body["status_url"]), (hdrs, body)
    assert body["status"] == "queued", body
    assert body["status_url"] == f"/v1/jobs/{body['job_id']}"
    return body


def poll(status_url, timeout=300):
    """Poll until terminal, recording every status transition observed."""
    seen, t0, last = [], time.time(), None
    while time.time() - t0 < timeout:
        code, _, body = req("GET", status_url)
        assert code == 200, f"poll returned {code}: {body}"
        if body["status"] != last:
            seen.append((round(time.time() - t0, 2), body["status"],
                         body.get("queue_position")))
            last = body["status"]
        if body["status"] in ("ready", "failed", "cancelled"):
            return body, seen
        time.sleep(0.5)
    raise TimeoutError(f"{status_url} still not terminal after {timeout}s")


def section_checks(book, whole):
    """Scope a summary to one section, then check the rejection paths.

    `whole` is a finished whole-article result: its `sections` index is exactly
    what a caller picks a section from, so no separate discovery request is
    needed. Costs one GPU job -- the failures below never reach the model.
    """
    sections = whole["sections"]
    print("\n=== section index from the first response ===")
    assert len(sections) > 1, sections
    assert sections[0]["title"] == "(lead)", sections[0]
    for s in sections:
        assert {"id", "title", "level", "words", "own_words"} <= s.keys(), s
        # A subtree is never smaller than the heading's own prose.
        assert s["words"] >= s["own_words"], s
    assert [s["id"] for s in sections] == list(range(len(sections))), \
        "ids must be contiguous"
    print(f"  {whole['article_title']}: {len(sections)} sections, "
          f"{whole['source_words']} words")

    # Pick a small leaf-ish section so this stays quick on the GPU.
    target = next(s for s in sections if s["id"] != 0 and 100 <= s["words"] <= 700)
    print(f"  targeting [{target['id']}] {target['title']!r} "
          f"({target['words']} words)")

    print("\n=== summarise just that section ===")
    code, hdrs, a = req("POST", "/v1/summarise",
                        {"book": book, "title": whole["article_title"],
                         "words": 60, "section": target["id"]})
    assert code == 202, a
    assert "location" in hdrs, a
    body, _ = poll(a["status_url"])
    assert body["status"] == "ready", body
    r = body["result"]
    assert r["section_id"] == target["id"], r
    assert r["section_title"] == target["title"], r
    # The scope must be the section, not the whole article.
    assert r["source_words"] == target["words"], (r["source_words"], target["words"])
    assert r["source_words"] < whole["source_words"], (r["source_words"],
                                                       whole["source_words"])
    assert r["source_url"] == whole["source_url"], "same article, same URL"
    assert len(r["sections"]) == len(sections), "result must carry the full index"
    print(f"  {r['actual_words']}/{r['target_words']} words from "
          f"{r['source_words']} | {r['passes']} pass(es) | "
          f"{body['processing_seconds']}s | within={r['within_tolerance']}")
    print("  SUMMARY:", r["summary"][:160])

    print("\n=== section rejection paths ===")
    cases = [
        ("bad section title", "Not A Real Heading", True, "SectionNotFound"),
        ("out-of-range id", len(sections) + 500, True, "SectionNotFound"),
    ]
    # A container heading with subsections excluded must fail with a fix-it hint,
    # and must not have burned GPU time to find out.
    container = next((s for s in sections
                      if s["own_words"] < 20 and s["words"] > 200), None)
    if container:
        cases.append(("container heading", container["id"], False,
                      "NothingToSummarise"))
    for label, section, subs, kind in cases:
        code, _, c = req("POST", "/v1/summarise",
                         {"book": book, "title": whole["article_title"],
                          "words": 40, "section": section,
                          "include_subsections": subs})
        assert code == 202, (label, code, c)
        b, _ = poll(c["status_url"], timeout=120)
        assert b["status"] == "failed", (label, b)
        assert b["error"]["kind"] == kind, (label, b["error"])
        if kind == "NothingToSummarise":
            assert "include_subsections=true" in b["error"]["detail"], b["error"]
        print(f"  {label:18} -> failed / {b['error']['kind']}: "
              f"{str(b['error']['detail'])[:80]}")


def main():
    book = "wikipedia_en_all_maxi"
    cases = [("Alan Turing", 150), ("Alan Turing", 40)]

    # Submit both up front so the second has to queue behind the first.
    accepted = [submit(book, t, w) for t, w in cases]
    print("=== 202 responses ===")
    for a in accepted:
        print(json.dumps(a))

    whole = None
    for (title, words), a in zip(cases, accepted):
        print(f"\n=== {title} -> {words} words ===")
        body, seen = poll(a["status_url"])
        print("transitions (t, status, queue_position):")
        for s in seen:
            print("   ", s)
        if body["status"] != "ready":
            print("FAILED:", json.dumps(body.get("error"), indent=1))
            return 1
        r = body["result"]
        if whole is None:
            whole = r  # its sections index drives the section checks below
        print(json.dumps({
            "target_words": r["target_words"], "actual_words": r["actual_words"],
            "word_delta": r["word_delta"], "passes": r["passes"],
            "source_words": r["source_words"], "prompt_tokens": r["prompt_tokens"],
            "resolved_via": r["resolved_via"],
            "prefill_tok_s": r["prefill_tok_s"], "decode_tok_s": r["decode_tok_s"],
            "llm_seconds": r["llm_seconds"],
            "processing_seconds": body.get("processing_seconds"),
            "article_title": r["article_title"],
        }, indent=1))
        print("SUMMARY:", r["summary"])

    print("\n=== error paths ===")
    for label, payload, expect in [
        ("unknown book", {"book": "nope_nope", "title": "X", "words": 50}, "failed"),
        ("unknown article", {"book": book, "title": "Zqxvw Not A Real Page 12345",
                             "words": 50}, "failed"),
        ("words too small", {"book": book, "title": "Alan Turing", "words": 2}, 422),
        ("missing field", {"book": book}, 422),
    ]:
        if expect == 422:
            code, _, body = req("POST", "/v1/summarise", payload)
            print(f"  {label:18} -> HTTP {code} (expected 422)")
            assert code == 422, body
            continue
        code, hdrs, body = req("POST", "/v1/summarise", payload)
        assert code == 202, body
        b, _ = poll(body["status_url"], timeout=120)
        err = (b.get("error") or {})
        print(f"  {label:18} -> {b['status']} / {err.get('kind')}: "
              f"{str(err.get('detail'))[:90]}")
        assert b["status"] == expect, b

    code, _, body = req("GET", "/v1/jobs/deadbeefdeadbeefdeadbeefdeadbeef")
    print(f"  unknown job id      -> HTTP {code} (expected 404)")
    assert code == 404

    section_checks(book, whole)

    print("\n=== stats ===")
    print(json.dumps(req("GET", "/v1/stats")[2], indent=1))
    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
