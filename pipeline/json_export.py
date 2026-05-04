"""
JSON export module.

Generates per-page JSON files and combined index files for the website.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from pipeline.config import COMBINED_DIR, JSON_DIR
from pipeline.ocr import OcrPage

logger = logging.getLogger(__name__)


def export_page_json(
    ocr_page: OcrPage,
    aligned_result: dict,
    output_dir: Path = JSON_DIR,
) -> Path:
    """
    Save the aligned result as a per-page JSON file.
    
    Adds the full word list with bounding boxes for the website viewer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ocr_page.scan_file).stem

    # Add the complete word list (for on-page highlighting)
    aligned_result["words"] = [
        {
            "id": w.id,
            "text": w.text,
            "bbox": w.bbox,
            "conf": w.confidence,
            "line_id": w.line_id,
        }
        for w in ocr_page.all_words
    ]
    
    # Add the plain-text transcription for the whole page
    aligned_result["full_text"] = ocr_page.full_text

    output_path = output_dir / f"{stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, ensure_ascii=False, indent=2)

    logger.debug(f"  Saved page JSON: {output_path}")
    return output_path


def build_combined_indexes(
    page_results: dict[str, dict],
    output_dir: Path = COMBINED_DIR,
):
    """
    Build combined indexes across all pages for the website.
    
    Creates:
    - search_index.json: All entries flattened with page references
    - address_index.json: All addresses → list of entries
    - street_index.json: All streets → list of entries
    - cross_reference_index.json: Cross-reference mappings
    - page_manifest.json: Page metadata list
    
    Args:
        page_results: Dict mapping scan filenames to aligned results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    search_entries = []
    address_entries = defaultdict(list)
    street_entries = defaultdict(list)
    cross_references = []
    page_manifest = []

    entry_counter = 0

    for scan_file, result in sorted(page_results.items()):
        page_number = result.get("page_number")
        section = result.get("section", "unknown")

        # Page manifest
        page_manifest.append({
            "scan_file": scan_file,
            "page_number": page_number,
            "section": section,
            "dimensions": result.get("dimensions"),
        })

        # Collect all entries from this page
        entries = _collect_entries_for_index(result)

        for entry in entries:
            entry_counter += 1
            global_id = f"entry_{entry_counter}"

            # Search index entry
            search_entry = {
                "id": global_id,
                "scan_file": scan_file,
                "page_number": page_number,
                "section": section,
                "type": entry.get("type", "entry"), # entry, entity, ad, page_text
                "name": entry.get("name"),
                "initials": entry.get("initials"),
                "name_prefix": entry.get("name_prefix"),
                "name_prefix_expanded": entry.get("name_prefix_expanded"),
                "occupation": entry.get("occupation"),
                "occupation_expanded": entry.get("occupation_expanded"),
                "address_street": entry.get("address_street"),
                "address_street_expanded": entry.get("address_street_expanded"),
                "address_number": entry.get("address_number"),
                "address_full": entry.get("address_full"),
                "phone": entry.get("phone"),
                "searchable_text": entry.get("searchable_text", ""),
                "entry_bbox": entry.get("entry_bbox"),
                "name_bbox": entry.get("name_bbox"),
                "address_bbox": entry.get("address_bbox"),
                "word_ids": entry.get("word_ids", []),
            }
            search_entries.append(search_entry)

            # Address index (only if we have a full address)
            address_full = entry.get("address_full")
            if address_full:
                address_entries[address_full.lower()].append({
                    "id": global_id,
                    "name": entry.get("name"),
                    "occupation": entry.get("occupation_expanded", entry.get("occupation")),
                    "scan_file": scan_file,
                    "page_number": page_number,
                })

            # Street index
            street = (
                entry.get("address_street_expanded")
                or entry.get("address_street")
            )
            if street:
                street_entries[street.lower()].append({
                    "id": global_id,
                    "name": entry.get("name"),
                    "address_number": entry.get("address_number"),
                    "scan_file": scan_file,
                    "page_number": page_number,
                })

            # Cross-references
            for xref in entry.get("cross_references", []):
                cross_references.append({
                    "source_id": global_id,
                    "source_page": page_number,
                    "target_type": xref.get("type"),
                    "target_page": xref.get("page"),
                })

    # Save all indexes
    _save_json(search_entries, output_dir / "search_index.json")
    _save_json(dict(address_entries), output_dir / "address_index.json")
    _save_json(dict(street_entries), output_dir / "street_index.json")
    _save_json(cross_references, output_dir / "cross_reference_index.json")
    _save_json(page_manifest, output_dir / "page_manifest.json")

    logger.info(
        f"Combined indexes built: "
        f"{len(search_entries)} entries, "
        f"{len(address_entries)} unique addresses, "
        f"{len(street_entries)} unique streets, "
        f"{len(cross_references)} cross-references, "
        f"{len(page_manifest)} pages"
    )


def _collect_entries_for_index(result: dict) -> list[dict]:
    """
    Collect all entries from a page result, regardless of section type.
    Normalizes the structure so every entry has the same base fields.
    """
    section = result.get("section", "generic")
    entries = []

    if section == "name_register":
        entries = result.get("entries", [])

    elif section == "street_register":
        for street in result.get("streets", []):
            for entry in street.get("entries", []):
                if not entry.get("address_street"):
                    entry["address_street"] = street.get("street_name")
                    entry["address_street_expanded"] = street.get("street_name_expanded")
                entries.append(entry)

    elif section == "occupation_register":
        for occ in result.get("occupations", []):
            for entry in occ.get("entries", []):
                if not entry.get("occupation"):
                    entry["occupation"] = occ.get("occupation_name")
                    entry["occupation_expanded"] = occ.get("occupation_name_expanded")
                entries.append(entry)

    elif section == "institutional":
        entries = result.get("entities", [])

    elif section == "advertisement":
        for ad in result.get("advertisements", []):
            if not ad.get("name"):
                ad["name"] = ad.get("business_name")
            ad["type"] = "advertisement"
            entries.append(ad)

    elif section == "other" or section == "generic":
        for addr in result.get("addresses_found", []):
            entries.append({
                "name": addr.get("context", "Unknown"),
                "address_street": addr.get("address_street"),
                "address_street_expanded": addr.get("address_street_expanded"),
                "address_number": addr.get("address_number"),
                "address_full": (
                    f"{addr.get('address_street_expanded', addr.get('address_street', ''))} "
                    f"{addr.get('address_number', '')}"
                ).strip() or None,
                "word_ids": addr.get("address_word_ids", []),
                "address_word_ids": addr.get("address_word_ids", []),
                "type": "entry",
            })

    # If NO entries were found (even in name_register etc.), or if it's an 'untyped' page,
    # add a fallback entry representing the full text of the page.
    if not entries and result.get("full_text"):
        entries.append({
            "name": f"Page {result.get('page_number', 'Text')}",
            "searchable_text": result.get("full_text", ""),
            "type": "page_text",
            "section": section,
        })

    return entries


def _save_json(data, path: Path):
    """Save data as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug(f"  Saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
