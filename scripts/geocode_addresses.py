#!/usr/bin/env python3
"""Geocode address_index.json keys against PDOK Locatieserver.

Usage:
    python scripts/geocode_addresses.py [--workers N] [--limit N]
                                        [--retry-failed] [--reset-gemeente]
"""

import argparse
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "output" / "combined" / "address_index.json"
OUTPUT = ROOT / "output" / "geocoded" / "addresses.json"
CHECKPOINT_EVERY = 500

PDOK_URL = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free"
HAS_NUMBER = re.compile(r"\d")
# Directional qualifiers that prefix street names in the 1926 book but confuse PDOK
_DIR_PREFIX = re.compile(r"^(noord(?:elijke|zijde)|zuid(?:elijke|zijde)|oost(?:elijke|zijde)|west(?:elijke|zijde))\s+")

# Historical → current name corrections applied to the PDOK query only (not the key).
# Phrases before single words so longer patterns are substituted first.
STREET_ALIASES = [
    ("verloren heereweg",  "verlengde hereweg"),
    ("verloren hereweg",   "verlengde hereweg"),
    ("friesche straatweg", "friesestraatweg"),
    ("hoornsche dijk",     "hoornsedijk"),
    ("sint jans straat",   "sint jansstraat"),
    ("sint janstraat",     "sint jansstraat"),
    ("heerestraat",        "herestraat"),
    ("heereweg",           "hereweg"),
    ("heerebinnensingel",  "herebinnensingel"),
    ("heeresingel",        "heresingel"),
    ("heereplein",         "hereplein"),
    ("hoornschedijk",      "hoornsedijk"),
    ("hoornschediep",      "hoornsediep"),
    ("helperwestsingel",   "helper westsingel"),
    ("helperoostsingel",   "helper oostsingel"),
    ("roodeweg",           "rodeweg"),
    ("verloren",           "verlengde"),   # systematic OCR misread of "Verlengde"
]


def normalize_query(address: str) -> str:
    q = _DIR_PREFIX.sub("", address)
    for old, new in STREET_ALIASES:
        q = q.replace(old, new)
    return q


def compute_flags(result: dict) -> list:
    """Return research flags for a result. Empty list means high confidence."""
    status = result.get("status")
    if status in ("no_match", "error"):
        return ["not_found"]
    if status == "ok":
        t = result.get("type", "")
        score = result.get("score") or 0
        if t in ("gemeente", "woonplaats") or score < 10:
            return ["uncertain"]  # matched to municipality/city, not a specific address
        if t == "weg":
            return ["uncertain"]  # street-level only, no house number matched
    return []


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def pdok_geocode(address: str) -> dict:
    if not HAS_NUMBER.search(address):
        return {"status": "no_number"}

    query = normalize_query(f"{address}, Groningen")
    params = urllib.parse.urlencode({
        "q": query,
        "fq": "gemeentenaam:Groningen",
        "rows": "1",
        "fl": "weergavenaam,centroide_ll,score,type",
    })
    url = f"{PDOK_URL}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"status": "error", "query": query, "detail": str(exc)}
    except Exception as exc:
        return {"status": "error", "query": query, "detail": str(exc)}

    docs = data.get("response", {}).get("docs", [])
    if not docs:
        return {"status": "no_match", "query": query}

    doc = docs[0]
    # centroide_ll is "POINT(lon lat)" — lon/lat order, not lat/lon
    m = re.search(r"POINT\(([0-9.]+)\s+([0-9.]+)\)", doc.get("centroide_ll", ""))
    if not m:
        return {"status": "no_match", "query": query}

    return {
        "status": "ok",
        "query": query,
        "lat": float(m.group(2)),
        "lng": float(m.group(1)),
        "score": doc.get("score"),
        "matched": doc.get("weergavenaam"),
        "type": doc.get("type"),
    }


def load_results() -> dict:
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp.replace(OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--reset-gemeente", action="store_true",
        help="Reset all 'Gemeente Groningen' fallback entries to no_match before running"
             " (combine with --retry-failed to re-geocode them with updated aliases)",
    )
    args = parser.parse_args()

    with INPUT.open(encoding="utf-8") as f:
        address_index = json.load(f)

    all_addresses = list(address_index.keys())
    log.info("Loaded %d unique addresses", len(all_addresses))

    results = load_results()

    if args.reset_gemeente:
        reset_count = sum(
            1 for r in results.values() if r.get("matched") == "Gemeente Groningen"
        )
        for addr, result in results.items():
            if result.get("matched") == "Gemeente Groningen":
                results[addr] = {"status": "no_match", "query": result.get("query", "")}
        log.info("Reset %d 'Gemeente Groningen' entries to no_match", reset_count)
        save_results(results)

    retry_statuses = {"no_match", "error"} if args.retry_failed else set()
    already = sum(
        1 for a in all_addresses
        if a in results and results[a].get("status") not in retry_statuses
    )
    log.info("Resuming: %d addresses already attempted", already)

    pending = [
        a for a in all_addresses
        if a not in results or results[a].get("status") in retry_statuses
    ]
    if args.limit:
        pending = pending[: args.limit]

    log.info("Queued %d addresses for PDOK lookup (%d workers)", len(pending), args.workers)

    lock = threading.Lock()
    counters = {"done": 0, "hits": 0, "no_match": 0, "no_number": 0, "errors": 0}
    start = time.monotonic()

    def process(addr: str) -> tuple[str, dict]:
        result = pdok_geocode(addr)
        result["flags"] = compute_flags(result)
        return addr, result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, a): a for a in pending}
        for fut in as_completed(futures):
            addr, result = fut.result()
            with lock:
                results[addr] = result
                counters["done"] += 1
                s = result["status"]
                if s == "ok":           counters["hits"] += 1
                elif s == "no_match":   counters["no_match"] += 1
                elif s == "no_number":  counters["no_number"] += 1
                else:                   counters["errors"] += 1

                n = counters["done"]
                if n % CHECKPOINT_EVERY == 0:
                    save_results(results)
                    elapsed = time.monotonic() - start
                    rate = n / elapsed if elapsed else 0
                    log.info(
                        "[%d/%d] hits=%d no_match=%d no_number=%d err=%d (%.1f req/s)",
                        n, len(pending),
                        counters["hits"], counters["no_match"],
                        counters["no_number"], counters["errors"],
                        rate,
                    )

    # Backfill flags on entries geocoded in earlier runs that predate this field.
    for result in results.values():
        if "flags" not in result:
            result["flags"] = compute_flags(result)

    save_results(results)
    elapsed = time.monotonic() - start
    total_ok   = sum(1 for r in results.values() if r.get("status") == "ok")
    needs_research = sum(1 for r in results.values() if r.get("flags"))

    print("=" * 62)
    print(f"Done in {elapsed:.0f}s ({counters['done']} processed)")
    print(f"  hits:          {counters['hits']}")
    print(f"  no_match:      {counters['no_match']}")
    print(f"  no_number:     {counters['no_number']}")
    print(f"  errors:        {counters['errors']}")
    print(f"  cumulative ok: {total_ok}/{len(all_addresses)} ({100*total_ok/len(all_addresses):.0f}%)")
    print(f"  needs_research: {needs_research}  (flags: uncertain or not_found)")
    print(f"  output:        {OUTPUT}")


if __name__ == "__main__":
    main()
