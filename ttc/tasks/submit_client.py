#!/usr/bin/env python3
"""`submit`: score a C++17 solution on the hidden polyomino test set (stdlib only).

Reads SUBMIT_URL and SUBMIT_TOKEN from the environment.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def call(method, path, payload=None, timeout=3600):
    url = os.environ["SUBMIT_URL"].rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-Agent-Token", os.environ["SUBMIT_TOKEN"])
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"submit: scorer error {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"submit: cannot reach the scorer: {e.reason}")


def render(r):
    if r.get("message") and r.get("status") != "compile_error":
        print(r["message"])
    if r.get("submission") is None:
        if r.get("your_best"):
            print(f"your best valid: #{r['your_best']['submission']} score={r['your_best']['score']:.6f}")
        return
    print(f"submission #{r['submission']}: status={r['status']} score={r['score']:.6f} valid={r['valid']}")
    if r["status"] == "compile_error":
        print("--- compiler output ---")
        print(r.get("message", ""))
    else:
        print(f"verdicts={r.get('verdicts')} max_cpu_s={r.get('max_cpu_s')} max_rss_mb={r.get('max_rss_mb')} "
              f"judge_seconds={r.get('judge_seconds')}")
    if r.get("your_best") is not None:
        b = r["your_best"]
        print(f"your best valid: #{b['submission']} score={b['score']:.6f}")


def main():
    p = argparse.ArgumentParser(prog="submit", description="Score a C++17 solution on the hidden test set.")
    p.add_argument("file", nargs="?", help="C++17 source file")
    p.add_argument("--best", action="store_true", help="show your best valid submission")
    p.add_argument("--ping", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true", help="print the raw JSON response")
    a = p.parse_args()
    if a.ping:
        r = call("GET", "/ping", timeout=30)
    elif a.best:
        r = call("GET", "/best", timeout=30)
    elif a.file:
        with open(a.file, encoding="utf-8", errors="replace") as f:
            r = call("POST", "/submit", {"source": f.read(), "filename": os.path.basename(a.file)})
    else:
        p.error("give a .cpp file, or --best")
    if a.json:
        print(json.dumps(r))
    else:
        render(r)


if __name__ == "__main__":
    main()
