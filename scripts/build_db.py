#!/usr/bin/env python3
"""
Build web/data/adresboek.sqlite from pipeline outputs:

  output/json/<stem>.json           per-page entries (with bboxes)
  output/overrides/<stem>.json      CRM corrections (merged at build time)
  output/geocoded/addresses.json    PDOK geocoding results
  output/combined/page_manifest.json (unused — section comes from per-page JSON)

Schema: pages, entries (+ FTS5 mirror), cross_references.
Idempotent: drops + recreates everything.

Run: .venv/bin/python scripts/build_db.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.json_export import _collect_entries_for_index  # noqa: E402

JSON_DIR = ROOT / "output" / "json"
OVERRIDES_DIR = ROOT / "output" / "overrides"
GEOCODED_PATH = ROOT / "output" / "geocoded" / "addresses.json"
DB_PATH = ROOT / "web" / "data" / "adresboek.sqlite"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_db")


SCHEMA = """
DROP TABLE IF EXISTS entries_fts;
DROP TABLE IF EXISTS cross_references;
DROP TABLE IF EXISTS entries;
DROP TABLE IF EXISTS pages;

CREATE TABLE pages (
    id INTEGER PRIMARY KEY,
    scan_file TEXT UNIQUE NOT NULL,
    stem TEXT UNIQUE NOT NULL,
    page_number INTEGER,
    section TEXT,
    width INTEGER,
    height INTEGER,
    header_text TEXT,
    footer_text TEXT
);

CREATE TABLE entries (
    id INTEGER PRIMARY KEY,
    page_id INTEGER NOT NULL REFERENCES pages(id),
    entry_index INTEGER NOT NULL,
    stable_id TEXT UNIQUE NOT NULL,    -- <stem>:<idx>
    name TEXT,
    initials TEXT,
    name_prefix TEXT,
    name_prefix_expanded TEXT,
    occupation TEXT,
    occupation_expanded TEXT,
    address_street TEXT,
    address_street_expanded TEXT,
    address_number TEXT,
    address_full TEXT,
    address_full_normalized TEXT,
    phone TEXT,
    notes TEXT,
    entry_bbox TEXT,                   -- JSON [x1,y1,x2,y2]
    name_bbox TEXT,
    address_bbox TEXT,
    word_ids TEXT,
    name_word_ids TEXT,
    address_word_ids TEXT,
    lat REAL,
    lng REAL,
    geocode_score REAL,
    geocode_type TEXT,                 -- adres | weg | gemeente | woonplaats | postcode | buurt
    geocode_matched TEXT,
    geocode_flags TEXT,                -- JSON array (e.g. ["uncertain"])
    flag_verified INTEGER DEFAULT 0,
    flag_needs_review INTEGER DEFAULT 0,
    flag_bbox_unreliable INTEGER DEFAULT 0,
    fingerprint TEXT,
    edited_at TEXT,
    searchable_text TEXT
);

CREATE INDEX idx_entries_page ON entries(page_id);
CREATE INDEX idx_entries_coords ON entries(lat, lng) WHERE lat IS NOT NULL;
CREATE INDEX idx_entries_address_norm ON entries(address_full_normalized);
CREATE INDEX idx_entries_name ON entries(name);

CREATE VIRTUAL TABLE entries_fts USING fts5(
    name, initials, name_prefix_expanded,
    occupation, occupation_expanded,
    address_street, address_street_expanded, address_number, address_full,
    searchable_text,
    content='entries',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE cross_references (
    id INTEGER PRIMARY KEY,
    source_entry_id INTEGER NOT NULL REFERENCES entries(id),
    target_text TEXT,
    target_page_number INTEGER,
    raw TEXT
);
CREATE INDEX idx_xref_source ON cross_references(source_entry_id);
"""


# ── Override merge (mirrors web/lib/overrides.ts → applyOverride) ─────────────


def apply_override(entry: dict, ov: dict | None) -> dict:
    if not ov:
        return entry
    merged = {**entry, **(ov.get("fields") or {})}
    bbox_ov = ov.get("bbox", {}).get("value") if ov.get("bbox") else None
    if bbox_ov:
        merged["entry_bbox"] = bbox_ov
    if ov.get("flags"):
        merged["_flags_override"] = ov["flags"]
    if ov.get("fingerprint"):
        merged["_fingerprint"] = ov["fingerprint"]
    if ov.get("edited_at"):
        merged["_edited_at"] = ov["edited_at"]
    fields = ov.get("fields") or {}
    if (
        fields
        and "address_full" not in fields
        and any(k in fields for k in ("address_street", "address_street_expanded", "address_number"))
    ):
        street = merged.get("address_street_expanded") or merged.get("address_street") or ""
        num = merged.get("address_number") or ""
        merged["address_full"] = " ".join([s for s in (street, num) if s]).strip()
    # Refresh searchable_text — same fields the FTS index uses
    merged["searchable_text"] = " ".join(
        str(merged.get(k) or "")
        for k in (
            "name",
            "initials",
            "name_prefix",
            "name_prefix_expanded",
            "occupation",
            "occupation_expanded",
            "address_street",
            "address_street_expanded",
            "address_number",
        )
    ).strip()
    return merged


def normalize_address(addr: str | None) -> str | None:
    if not addr:
        return None
    s = addr.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s or None


def fingerprint(entry: dict) -> str:
    norm = lambda v: re.sub(r"\s+", " ", (v or "").lower()).strip() if isinstance(v, str) else ""
    sig = "|".join([
        norm(entry.get("name")),
        norm(entry.get("initials")),
        norm(entry.get("name_prefix")),
        norm(entry.get("address_street_expanded") or entry.get("address_street")),
        norm(entry.get("address_number")),
        norm(entry.get("occupation_expanded") or entry.get("occupation")),
    ])
    return "sha1:" + hashlib.sha1(sig.encode("utf-8")).hexdigest()


def jdumps(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


# ── Build ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if not JSON_DIR.exists():
        log.error(f"Missing input dir: {JSON_DIR}")
        sys.exit(1)
    geocoded: dict[str, dict] = {}
    if GEOCODED_PATH.exists():
        geocoded = json.loads(GEOCODED_PATH.read_text(encoding="utf-8"))
        log.info(f"Loaded {len(geocoded)} geocoded address keys")
    else:
        log.warning("No geocoded file — entries will have NULL lat/lng")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    cur = conn.cursor()
    t0 = time.time()

    page_files = sorted(JSON_DIR.glob("*.json"))
    log.info(f"Importing {len(page_files)} pages")

    n_entries = 0
    n_geocoded = 0
    n_overridden = 0
    n_xrefs = 0

    for pf in page_files:
        page = json.loads(pf.read_text(encoding="utf-8"))
        stem = pf.stem
        section = page.get("section", "unknown")
        dims = page.get("dimensions") or {}
        header = (page.get("header") or {}).get("text") or None
        footer = (page.get("footer") or {}).get("text") or None
        cur.execute(
            """INSERT INTO pages
               (scan_file, stem, page_number, section, width, height, header_text, footer_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                page.get("scan_file") or f"{stem}.jpg",
                stem,
                page.get("page_number"),
                section,
                dims.get("width"),
                dims.get("height"),
                header,
                footer,
            ),
        )
        page_id = cur.lastrowid

        ov_path = OVERRIDES_DIR / f"{stem}.json"
        overrides = {}
        if ov_path.exists():
            try:
                overrides = json.loads(ov_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning(f"  override unreadable for {stem}: {e}")

        entries = _collect_entries_for_index(page)
        for idx, raw_entry in enumerate(entries):
            stable_id = f"{stem}:{idx}"
            ov = overrides.get(stable_id)
            entry = apply_override(raw_entry, ov)
            if ov:
                n_overridden += 1

            address_full = entry.get("address_full")
            address_norm = normalize_address(address_full)

            geo = geocoded.get(address_norm) if address_norm else None
            geo_lat = geo.get("lat") if geo and geo.get("status") == "ok" else None
            geo_lng = geo.get("lng") if geo and geo.get("status") == "ok" else None
            if geo_lat is not None:
                n_geocoded += 1

            flags = entry.get("_flags_override") or {}

            cur.execute(
                """INSERT INTO entries (
                       page_id, entry_index, stable_id,
                       name, initials, name_prefix, name_prefix_expanded,
                       occupation, occupation_expanded,
                       address_street, address_street_expanded, address_number,
                       address_full, address_full_normalized,
                       phone, notes,
                       entry_bbox, name_bbox, address_bbox,
                       word_ids, name_word_ids, address_word_ids,
                       lat, lng, geocode_score, geocode_type, geocode_matched, geocode_flags,
                       flag_verified, flag_needs_review, flag_bbox_unreliable,
                       fingerprint, edited_at, searchable_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?)""",
                (
                    page_id, idx, stable_id,
                    entry.get("name"),
                    entry.get("initials"),
                    entry.get("name_prefix"),
                    entry.get("name_prefix_expanded"),
                    entry.get("occupation"),
                    entry.get("occupation_expanded"),
                    entry.get("address_street"),
                    entry.get("address_street_expanded"),
                    entry.get("address_number"),
                    address_full,
                    address_norm,
                    entry.get("phone"),
                    entry.get("notes"),
                    jdumps(entry.get("entry_bbox")),
                    jdumps(entry.get("name_bbox")),
                    jdumps(entry.get("address_bbox")),
                    jdumps(entry.get("word_ids")),
                    jdumps(entry.get("name_word_ids")),
                    jdumps(entry.get("address_word_ids")),
                    geo_lat, geo_lng,
                    geo.get("score") if geo else None,
                    geo.get("type") if geo else None,
                    geo.get("matched") if geo else None,
                    jdumps(geo.get("flags")) if geo else None,
                    1 if flags.get("verified") else 0,
                    1 if flags.get("needs_review") else 0,
                    1 if flags.get("bbox_unreliable") else 0,
                    entry.get("_fingerprint") or fingerprint(entry),
                    entry.get("_edited_at"),
                    entry.get("searchable_text"),
                ),
            )
            entry_id = cur.lastrowid
            n_entries += 1

            for xref in entry.get("cross_references") or []:
                cur.execute(
                    """INSERT INTO cross_references (source_entry_id, target_text, target_page_number, raw)
                       VALUES (?, ?, ?, ?)""",
                    (
                        entry_id,
                        xref.get("text") if isinstance(xref, dict) else None,
                        xref.get("page_number") if isinstance(xref, dict) else None,
                        json.dumps(xref, ensure_ascii=False) if not isinstance(xref, str) else xref,
                    ),
                )
                n_xrefs += 1

        if (page_files.index(pf) + 1) % 100 == 0:
            log.info(f"  imported {page_files.index(pf) + 1}/{len(page_files)} pages")

    log.info("Rebuilding FTS5 index...")
    cur.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")

    log.info("ANALYZE + VACUUM...")
    conn.commit()
    conn.execute("ANALYZE")
    conn.execute("VACUUM")
    conn.close()

    elapsed = time.time() - t0
    size_mb = DB_PATH.stat().st_size / 1_000_000
    log.info("=" * 60)
    log.info(f"Done in {elapsed:.1f}s")
    log.info(f"  pages:       {len(page_files)}")
    log.info(f"  entries:     {n_entries}")
    log.info(f"  geocoded:    {n_geocoded} ({100*n_geocoded/max(n_entries,1):.1f}%)")
    log.info(f"  overridden:  {n_overridden}")
    log.info(f"  xrefs:       {n_xrefs}")
    log.info(f"  output:      {DB_PATH.relative_to(ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
