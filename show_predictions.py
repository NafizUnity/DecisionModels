"""Per-request view of one model's predictions: ground truth, the model's top-1 and its probability, the probability and
rank of the ground truth, and how a wrong pick relates to the ground truth in the ICD hierarchy. Ids and codes only;
never prints entity or query text.

  python show_predictions.py --run-dir results/NAME --model DREX                       # first 30 requests + summary
  python show_predictions.py --run-dir results/NAME --model DREX --wrong-only --source snomed_syn --order shuf1 --limit 50
  python show_predictions.py --run-dir results/NAME --model DREX --request 'mdace_25762_374_377|csv_dense|original'
  python show_predictions.py --run-dir results/NAME --model DREX --csv per_request_DREX.csv --limit 0   # every request to a CSV

Columns:
  gt         reference code            gt_p      model probability of the reference code
  top1       model's pick              top1_p    its probability
  gt_rank    rank of the reference code in the model's ordering (1 = picked; '-' = not among the options)
  retr_rank  the retriever's rank of the reference code in this list
  relation   pick vs reference: correct | parent/child (one code extends the other) | sibling (same 3-character
             category) | same chapter (same first letter) | other chapter | not in list (reference not an option)
Ties go to the earlier shown position, the rule the runner and the report use.

Summary (over every selected request, also with --wrong-only):
  1 outcomes by relation   2 where the reference ranks when missed (top-2/3/5/10)   3 how close the misses are
  4 model vs retriever     5 accuracy by confidence and at answer-only-if-confident thresholds
  6 option-order sensitivity (same pick in original and shuf1? right in at least one?)
  7 most frequent reference -> pick confusions and most-missed reference codes
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core  # noqa: E402
from run_decision_model import rank_options  # noqa: E402

RELATIONS = ("correct", "parent/child", "sibling", "same chapter", "other chapter", "not in list")


def relation(pick, gt, answerable):
    if not answerable:
        return "not in list"
    if pick == gt:
        return "correct"
    if pick.startswith(gt) or gt.startswith(pick):
        return "parent/child"
    if pick[:3] == gt[:3]:
        return "sibling"
    return "same chapter" if pick[:1] == gt[:1] else "other chapter"


def load(run_dir, model, smoke):
    out = Path(run_dir).resolve()
    req_path = out / "requests.jsonl"
    pred_path = out / f"predictions_{model}{'_smoke' if smoke else ''}.jsonl"
    for p in (req_path, pred_path):
        if not p.exists():
            sys.exit(f"STOPPED: {p} not found")
    meta = pred_path.with_name(pred_path.name + ".meta.json")
    if meta.exists():
        recorded = json.loads(meta.read_text()).get("requests_sha256")
        if recorded != hashlib.sha256(req_path.read_bytes()).hexdigest():
            sys.exit(f"STOPPED: {pred_path.name} was not made from this requests.jsonl")
    reqs = {}
    for line in open(req_path, encoding="utf8"):
        r = json.loads(line)
        if "n_options" not in r:
            sys.exit(f"STOPPED: {out} was built by the earlier top-k version; this script reads whole-list / --top runs")
        reqs[r["request_id"]] = {k: r[k] for k in ("source", "order", "gt_code", "answerable", "retriever_rank", "n_options")}
        reqs[r["request_id"]]["keys"] = list(r["record"]["questions"][lh_core.QUESTION_ID]["criteria"])   # shown order
    for rid, r in reqs.items():   # the retriever's order = the codes of the 'original' twin (build writes both orders)
        twin = reqs.get(rid.rsplit("|", 1)[0] + "|original")
        if twin is None:
            sys.exit(f"STOPPED: {rid} has no '|original' request; this script needs both orders in requests.jsonl")
        r["ranking"] = twin["keys"]
    preds = {p["request_id"]: p for p in lh_core.read_jsonl_readonly(pred_path)[0] if p.get("status") == "ok"}
    return reqs, preds


def rows_for(reqs, preds):
    for rid, p in preds.items():
        r = reqs.get(rid)
        if r is None:
            continue
        probs = p["probabilities"]
        ranks = rank_options(r["keys"], probs)
        top1 = min(ranks, key=ranks.get)
        gt = r["gt_code"]
        yield {"request_id": rid, "source": r["source"], "order": r["order"], "n_options": r["n_options"], "gt": gt,
               "gt_p": probs.get(gt) if r["answerable"] else None, "gt_rank": ranks.get(gt) if r["answerable"] else None,
               "retr_rank": r["retriever_rank"], "top1": top1, "top1_p": probs[top1],
               "relation": relation(top1, gt, r["answerable"]),
               "pick_retr_rank": r["ranking"].index(top1) + 1}     # the retriever's rank of the model's pick


def pct(a, b):
    return f"{a / b:.1%}" if b else "-"


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def summarize(rows, title, top_pairs=10):
    """Failure analysis over the selected requests (counts and codes only)."""
    n = len(rows)
    c = Counter(x["relation"] for x in rows)
    ans = [x for x in rows if x["gt_rank"] is not None]          # reference among the options
    miss = [x for x in ans if x["relation"] != "correct"]       # reference among the options, but not picked
    print(f"\n=== summary: {title}, {n} requests ===")

    print("\n1. Outcomes (share of all requests)")
    for k in RELATIONS:
        extra = f"   {pct(c[k], len(miss))} of misses" if k not in ("correct", "not in list") else ""
        print(f"  {k:<14} {c[k]:>6}  {c[k] / n:.3f}{extra}")
    print(f"  top-1 {c['correct'] / n:.3f} of all; {pct(c['correct'], len(ans))} when the reference is in the list "
          f"(ceiling {len(ans) / n:.3f})")
    if not miss:
        return

    print(f"\n2. Where the reference ranks when the model misses it ({len(miss)} misses with the reference in the list)")
    for k in (2, 3, 5, 10):
        m = sum(x["gt_rank"] <= k for x in miss)
        print(f"  in model top-{k:<3} {m:>6}  {pct(m, len(miss))} of misses   -> top-1 would rise by {m / n:.3f} if all were fixed")
    print(f"  rank >10     {sum(x['gt_rank'] > 10 for x in miss):>6}   median rank {median(x['gt_rank'] for x in miss)}")

    print("\n3. How close the misses are (model probabilities)")
    gaps = [x["top1_p"] - x["gt_p"] for x in miss]
    ratio = [x["gt_p"] / x["top1_p"] for x in miss if x["top1_p"] > 0]
    print(f"  median p(pick) {median(x['top1_p'] for x in miss):.3f}, median p(reference) {median(x['gt_p'] for x in miss):.3f}, "
          f"median gap {median(gaps):.3f}")
    for t in (0.5, 0.25):
        m = sum(r >= t for r in ratio)
        print(f"  p(reference) >= {t:g} x p(pick): {m:>6}  {pct(m, len(miss))} of misses (close calls)")
    m = sum(x["gt_p"] < 0.01 for x in miss)
    print(f"  p(reference) < 0.01:          {m:>6}  {pct(m, len(miss))} of misses (model is confidently elsewhere)")

    print("\n4. Model vs retriever on the same requests")
    rr = Counter((x["retr_rank"] == 1, x["relation"] == "correct") for x in rows)
    print(f"  both right {rr[(True, True)]:>6}   retriever right, model wrong {rr[(True, False)]:>6}   "
          f"model right, retriever wrong {rr[(False, True)]:>6}   both wrong {rr[(False, False)]:>6}")
    m = sum(x["retr_rank"] is not None and x["retr_rank"] <= 5 for x in miss)
    print(f"  misses with the reference in the retriever's top 5: {m} ({pct(m, len(miss))})")
    picks_rr = Counter(x["pick_retr_rank"] for x in miss)
    m = picks_rr.get(1, 0)
    print(f"  misses where the model picked the retriever's #1: {m} ({pct(m, len(miss))})")

    print("\n5. Confidence: accuracy by the model's top-1 probability (all requests)")
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]
    for lo, hi in bins:
        b = [x for x in rows if lo <= x["top1_p"] < hi]
        print(f"  p(pick) {lo:.1f}-{min(hi, 1):.1f}: {len(b):>6} requests ({pct(len(b), n)}), top-1 {pct(sum(x['relation'] == 'correct' for x in b), len(b))}")
    for t in (0.3, 0.5, 0.7):
        b = [x for x in rows if x["top1_p"] >= t]
        print(f"  answer only when p(pick) >= {t:g}: coverage {pct(len(b), n)}, top-1 {pct(sum(x['relation'] == 'correct' for x in b), len(b))}")

    pairs = {}
    for x in rows:
        pairs.setdefault(x["request_id"].rsplit("|", 1)[0], {})[x["order"]] = x
    both = [p for p in pairs.values() if len(p) == 2]
    if both:
        agree = [p for p in both if p["original"]["top1"] == p["shuf1"]["top1"]]
        acc = lambda ps: sum(q["original"]["relation"] == "correct" for q in ps)   # noqa: E731
        either = sum(any(q["relation"] == "correct" for q in p.values()) for p in both)
        print(f"\n6. Option-order sensitivity ({len(both)} entity x source pairs with both orders)")
        print(f"  same pick in both orders: {pct(len(agree), len(both))}; top-1 (original) {pct(acc(agree), len(agree))} when they agree, "
              f"{pct(acc([p for p in both if p not in agree]), len(both) - len(agree))} when they disagree")
        print(f"  right in at least one order: {pct(either, len(both))} (if one could always choose the better order)")

    print(f"\n7. Most frequent confusions (reference -> pick), top {top_pairs}")
    for (g, p), k in Counter((x["gt"], x["top1"]) for x in miss).most_common(top_pairs):
        print(f"  {g:<7} -> {p:<7} {k:>5}  {relation(p, g, True)}")
    gt_n = Counter(x["gt"] for x in ans)
    gt_miss = Counter(x["gt"] for x in miss)
    print("  most-missed reference codes: " + ", ".join(f"{g} {k}/{gt_n[g]}" for g, k in gt_miss.most_common(top_pairs)))


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True, choices=["ELEPHANT", "CLEF", "DREX"])
    ap.add_argument("--smoke", action="store_true", help="read predictions_<MODEL>_smoke.jsonl")
    ap.add_argument("--source"); ap.add_argument("--order", choices=["original", "shuf1"])
    ap.add_argument("--request", help="one request_id")
    ap.add_argument("--wrong-only", action="store_true", help="only requests whose top-1 is not the reference code")
    ap.add_argument("--limit", type=int, default=30, help="rows to print (0 = none; the summary always covers all)")
    ap.add_argument("--csv", help="also write every selected row to this CSV (ids and codes only)")
    ap.add_argument("--top-pairs", type=int, default=10, help="how many frequent confusions to list in the summary")
    args = ap.parse_args()

    reqs, preds = load(args.run_dir, args.model, args.smoke)
    rows = [x for x in rows_for(reqs, preds)
            if (not args.source or x["source"] == args.source) and (not args.order or x["order"] == args.order)
            and (not args.request or x["request_id"] == args.request)]
    if not rows:
        sys.exit("no matching requests with an ok prediction")
    rows.sort(key=lambda x: x["request_id"])
    shown = [x for x in rows if not args.wrong_only or x["relation"] != "correct"]

    if args.limit:
        w = max(len(x["request_id"]) for x in shown[:args.limit]) if shown else 10
        print(f"{'request_id':<{w}}  {'gt':<7} {'gt_p':>6} {'gt_rank':>7} {'retr_rank':>9}  {'top1':<7} {'top1_p':>6}  relation")
        for x in shown[:args.limit]:
            print(f"{x['request_id']:<{w}}  {x['gt']:<7} {fmt(x['gt_p']):>6} {fmt(x['gt_rank']):>7} {fmt(x['retr_rank']):>9}  "
                  f"{x['top1']:<7} {x['top1_p']:.3f}  {x['relation']}")
        if len(shown) > args.limit:
            print(f"... {len(shown) - args.limit} more (raise --limit or use --csv)")

    summarize(rows, f"{args.model}" + "".join(f", {k}={v}" for k, v in (("source", args.source), ("order", args.order)) if v),
              args.top_pairs)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(shown)
        print(f"wrote {len(shown)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
