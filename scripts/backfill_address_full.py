"""
One-shot: rewrite address_full in already-exported per-page JSONs to drop the
duplicated-token bug ("51 51" -> "51"). Pilot data only; future runs use the
fixed align.py path.
"""
import json
import glob
import sys


def fix_obj(o, counter):
    if isinstance(o, dict):
        af = o.get("address_full")
        if isinstance(af, str):
            p = af.strip().split()
            if len(p) == 2 and p[0] == p[1]:
                o["address_full"] = p[0]
                counter[0] += 1
        for v in o.values():
            fix_obj(v, counter)
    elif isinstance(o, list):
        for v in o:
            fix_obj(v, counter)


def main():
    total_fixed = 0
    files_changed = 0
    for f in glob.glob("output/json/*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)
            continue
        before = [0]
        fix_obj(d, before)
        if before[0]:
            json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            total_fixed += before[0]
            files_changed += 1
    print(f"fixed {total_fixed} entries across {files_changed} files")


if __name__ == "__main__":
    main()
