import json, re, statistics as st, sys
from collections import Counter

data = json.load(open(sys.argv[1], encoding="utf8"))
CLEAN = re.compile(r"^[A-Z]\d{2}[0-9A-Z]*$")   # e.g. A049, I10; no dot, uppercase, no spaces

def check(name, codes):
    c = Counter()
    odd = set()
    for x in codes:
        c["total"] += 1
        if not isinstance(x, str): c["not_string"] += 1; odd.add(repr(x)); continue
        c["with_dot"] += "." in x
        c["has_lowercase"] += x != x.upper()
        c["has_whitespace"] += bool(re.search(r"\s", x))
        c["empty"] += x == ""
        if not CLEAN.match(x): c["unusual_format"] += 1; odd.add(x)
    print(f"{name:18} {dict(c)}  length={dict(sorted(Counter(len(x) for x in codes if isinstance(x, str)).items()))}"
          + (f"  examples={sorted(odd)[:5]}" if odd else ""))

print("entities:", len(data), "| notes:", len({d.get("note_id") for d in data}),
      "| duplicate entity_ids:", len(data) - len({str(d.get("entity_id")) for d in data}))
print("top-level key sets:", Counter(tuple(sorted(d)) for d in data).most_common(3))
print("empty query_text:", sum(not (d.get("query_text") or "").strip() for d in data),
      "| query_text == entity_text:", sum((d.get("query_text") or "").strip().lower() == str(d.get("entity_text") or "").strip().lower() for d in data))
print("# ground-truth codes per entity:", dict(sorted(Counter(len(d["ground_truth_codes"]) for d in data).items())),
      "| with duplicates:", sum(len(set(d["ground_truth_codes"])) < len(d["ground_truth_codes"]) for d in data))

print("\nCODE FORMAT")
check("ground_truth", [c for d in data for c in d["ground_truth_codes"]])
sources = sorted({s for d in data for s in d["rankings"]})
for s in sources:
    check(s, [c for d in data for c in d["rankings"].get(s, [])])

print("\nRANKINGS")
print("entities where all sources have the same list length:", sum(len({len(v) for v in d["rankings"].values()}) == 1 for d in data))
for s in sources:
    rows = [(d["rankings"].get(s, []), set(d["ground_truth_codes"])) for d in data]
    lens = [len(r) for r, _ in rows]
    recall = {k: round(sum(bool(set(r[:k]) & g) for r, g in rows) / len(rows), 3) for k in (1, 5, 10, 20, 100)}
    print(f"{s:18} len min/median/max={min(lens)}/{st.median(lens)}/{max(lens)}  empty={lens.count(0)}"
          f"  shorter_than_20={sum(x < 20 for x in lens)}  duplicates={sum(len(set(r)) < len(r) for r, _ in rows)}  recall@k={recall}")
