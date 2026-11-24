#!/usr/bin/env python3
"""Compare two LLM picker prediction files entity by entity, e.g. ours against the original RAG picker's output.

  python picker_compare.py --ours PICKER_DIR/predictions/rrf_csv+fsn+syn__RETRIEVER__k20.jsonl --original FILE.jsonl

Rows are matched on (note_id, start, end), which both scripts write. For the matched entities it counts whether the two
runs had the same reference code, the same entity text, the same candidates (same codes in the same order) and the
same pick, and gives each run's top-1 (exact code, dots and case ignored). Picks that differ although entity text
and candidates are the same come from generation (stop tokens, parse fallback, batch size). Reads only; prints counts
and codes only, never entity or note text. Standard library only.
"""
import argparse
import collections
import json
import sys


def dotless(code):
    return str(code or "").upper().replace(".", "").strip()


def load(path):
    rows = [json.loads(line) for line in open(path, encoding="utf8") if line.strip()]
    for k in ("note_id", "start", "end", "icd_code", "llm_icd_code", "top_k"):
        if rows and k not in rows[0]:
            sys.exit(f"STOPPED: {path} rows have no '{k}' (not a picker prediction file?)")
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(str(r["note_id"]), r["start"], r["end"])].append(r)
    return rows, by_key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="a prediction file from llm_picker.py")
    ap.add_argument("--original", required=True, help="a prediction file from the original run_rag_picker.py")
    args = ap.parse_args()
    ours, a = load(args.ours)
    orig, b = load(args.original)
    dup = sum(len(v) > 1 for v in a.values()) + sum(len(v) > 1 for v in b.values())
    keys = sorted(k for k in a.keys() & b.keys() if len(a[k]) == 1 and len(b[k]) == 1)
    if not keys:
        sys.exit("STOPPED: no entity matches on (note_id, start, end); are these runs on the same entities?")
    pairs = [(a[k][0], b[k][0]) for k in keys]
    n = len(pairs)

    def share(x):
        return f"{x}/{n} ({x / n:.3f})"

    right = lambda r: dotless(r["llm_icd_code"]) == dotless(r["icd_code"])   # noqa: E731
    same_gt = sum(dotless(x["icd_code"]) == dotless(y["icd_code"]) for x, y in pairs)
    same_text = sum(x.get("entity_text") == y.get("entity_text") for x, y in pairs)
    same_list = [list(map(dotless, x["top_k"])) == list(map(dotless, y["top_k"])) for x, y in pairs]
    same_set = sum(set(map(dotless, x["top_k"])) == set(map(dotless, y["top_k"])) for x, y in pairs)
    same_pick = [dotless(x["llm_icd_code"]) == dotless(y["llm_icd_code"]) for x, y in pairs]
    same_input = [s and x.get("entity_text") == y.get("entity_text") for s, (x, y) in zip(same_list, pairs)]
    outside = lambda r: dotless(r["llm_icd_code"]) not in set(map(dotless, r["top_k"]))   # noqa: E731
    empty = lambda r: not dotless(r["llm_icd_code"])   # noqa: E731
    out_list_ours = sum(outside(x) and not empty(x) for x, _ in pairs)
    out_list_orig = sum(outside(y) and not empty(y) for _, y in pairs)
    k_ours = collections.Counter(len(x["top_k"]) for x, _ in pairs).most_common(3)
    k_orig = collections.Counter(len(y["top_k"]) for _, y in pairs).most_common(3)

    print(f"rows: ours {len(ours)}, original {len(orig)}; matched on (note_id, start, end): {n}"
          + (f"; {dup} keys held more than one row and were left out" if dup else ""))
    print(f"  list length (K): ours {k_ours}, original {k_orig}")
    print(f"  same reference code:        {share(same_gt)}")
    print(f"  same entity text:           {share(same_text)}")
    print(f"  same candidates and order:  {share(sum(same_list))}   (same codes, any order: {share(same_set)})")
    print(f"  same pick:                  {share(sum(same_pick))}")
    print(f"  same input (text + list), different pick: {sum(si and not sp for si, sp in zip(same_input, same_pick))}")
    print(f"  no pick (unreadable reply): ours {sum(empty(x) for x, _ in pairs)}, original {sum(empty(y) for _, y in pairs)}")
    print(f"  a code outside its own list: ours {out_list_ours}, original {out_list_orig}")
    print(f"top-1 on the {n} matched entities: ours {sum(right(x) for x, _ in pairs) / n:.3f}, "
          f"original {sum(right(y) for _, y in pairs) / n:.3f}")
    print(f"  only ours right: {sum(right(x) and not right(y) for x, y in pairs)}, "
          f"only original right: {sum(right(y) and not right(x) for x, y in pairs)}")
    if any(same_input):
        m = [(x, y) for si, (x, y) in zip(same_input, pairs) if si]
        print(f"  on the {len(m)} entities with the same input: ours {sum(right(x) for x, _ in m) / len(m):.3f}, "
              f"original {sum(right(y) for _, y in m) / len(m):.3f}")
    print(f"top-1 on each whole file: ours {sum(map(right, ours)) / len(ours):.3f} ({len(ours)} rows), "
          f"original {sum(map(right, orig)) / len(orig):.3f} ({len(orig)} rows)")


if __name__ == "__main__":
    main()
