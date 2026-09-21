#!/usr/bin/env python3
"""Tiny CLI client: submit a summarisation job and poll it to completion.

    python scripts/get_summary.py "Alan Turing" -n 150
    python scripts/get_summary.py "Enigma machine" -n 40 --strict
    python scripts/get_summary.py "Alan Turing" -n 150 --json   # shows sections
    python scripts/get_summary.py "Alan Turing" -n 70 --section Cryptanalysis
    python scripts/get_summary.py "Alan Turing" -n 45 --section 13 --own-only
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("SUMMARISER_BASE", "http://127.0.0.1:8080").rstrip("/")
BOOK = os.environ.get("KIWIX_BOOK", "")


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"detail": raw[:400]}


def pick_book() -> str:
    if BOOK:
        return BOOK
    code, body = call("GET", "/v1/books")
    if code != 200:
        sys.exit(f"could not list books: HTTP {code} {body}")
    books = body.get("books", [])
    if not books:
        sys.exit("this Kiwix instance serves no books")
    if len(books) == 1:
        return books[0]["key"]
    print("Multiple books available; pass --book or set KIWIX_BOOK:")
    for b in books:
        print(f"  {b['key']:<40} {b['title']} ({b.get('language')})")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("title", help="article title")
    ap.add_argument("-n", "--words", type=int, default=150)
    ap.add_argument("-b", "--book", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="fail rather than summarise a fuzzy title match")
    ap.add_argument("-s", "--section", default=None,
                    help="summarise one section, by id or heading title "
                         "(see --json for the list)")
    ap.add_argument("--own-only", action="store_true",
                    help="with --section, exclude nested subsections")
    ap.add_argument("--json", action="store_true", help="print the full job JSON")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    payload = {"book": args.book or pick_book(), "title": args.title,
               "strict": args.strict, "words": args.words,
               "section": args.section,
               "include_subsections": not args.own_only}

    code, body = call("POST", "/v1/summarise", payload)
    if code != 202:
        sys.exit(f"submit failed: HTTP {code} {json.dumps(body)[:300]}")
    status_url = body["status_url"]
    print(f"job {body['job_id']}  ->  {BASE}{status_url}", file=sys.stderr)

    t0, last = time.time(), None
    while time.time() - t0 < args.timeout:
        code, job = call("GET", status_url)
        if code != 200:
            sys.exit(f"poll failed: HTTP {code}")
        if job["status"] != last:
            last = job["status"]
            extra = f" (queue position {job['queue_position']})" \
                if job.get("queue_position") is not None else ""
            print(f"  [{time.time()-t0:5.1f}s] {job['status']}{extra}", file=sys.stderr)
        if job["status"] == "ready":
            if args.json:
                print(json.dumps(job, indent=1))
            else:
                r = job["result"]
                print(r["summary"])
                where = f" | section {r['section_id']}: {r['section_title']}" \
                    if r.get("section_id") is not None else ""
                print(f"\n-- {r['article_title']}{where} | {r['actual_words']}/"
                      f"{r['target_words']} words | from {r['source_words']} | "
                      f"{r['passes']} pass(es) | {job.get('processing_seconds')}s | "
                      f"{r['source_url']}", file=sys.stderr)
            return 0
        if job["status"] in ("failed", "cancelled"):
            print(json.dumps(job.get("error"), indent=1), file=sys.stderr)
            return 1
        time.sleep(0.5)

    print("timed out waiting for the job", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
