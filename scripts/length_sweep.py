"""Measure word-count accuracy across target lengths. Not part of pytest."""
import statistics
import sys
import time

from smoke import poll, req  # type: ignore

BOOK = "wikipedia_en_all_maxi"


def run(title, words):
    code, _, body = req("POST", "/v1/summarise",
                        {"book": BOOK, "title": title, "words": words})
    assert code == 202, body
    job, _ = poll(body["status_url"], timeout=600)
    if job["status"] != "ready":
        return {"words": words, "error": job.get("error")}
    r = job["result"]
    return {
        "target": words,
        "actual": r["actual_words"],
        "delta": r["word_delta"],
        "pct": round(r["word_delta"] / words * 100, 1),
        "ok": r["within_tolerance"],
        "passes": r["passes"],
        "trimmed": r["trimmed"],
        "seconds": job.get("processing_seconds"),
    }


def main():
    targets = [int(x) for x in (sys.argv[1:] or [25, 40, 75, 150, 300])]
    title = "Alan Turing"
    print(f"book={BOOK} article={title!r} targets={targets}\n")
    rows = []
    for w in targets:
        t0 = time.time()
        row = run(title, w)
        rows.append(row)
        flag = "OK " if row.get("ok") else "OFF"
        print(f"  n={w:<4} {flag} actual={row.get('actual'):<5} "
              f"delta={row.get('delta'):+<5} ({row.get('pct')}%) "
              f"passes={row.get('passes')} trimmed={str(row.get('trimmed')):<5} "
              f"wall={time.time()-t0:.1f}s")

    good = [r for r in rows if "delta" in r]
    if good:
        pct = [abs(r["pct"]) for r in good]
        print(f"\nin tolerance: {sum(1 for r in good if r['ok'])}/{len(good)}")
        print(f"mean |overshoot|: {statistics.mean(pct):.1f}%   max: {max(pct):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
