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
               "relation": relation(top1, gt, r["answerable"])}


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

    n = len(rows)
    c = Counter(x["relation"] for x in rows)
    wrong_in_list = sum(c[k] for k in ("parent/child", "sibling", "same chapter", "other chapter"))
    print(f"\nsummary: {args.model}, {n} requests" + "".join(f", {k}={v}" for k, v in
          (("source", args.source), ("order", args.order)) if v))
    for k in RELATIONS:
        share_wrong = f"  ({c[k] / wrong_in_list:.1%} of wrong picks with the reference in the list)" \
            if k not in ("correct", "not in list") and wrong_in_list else ""
        print(f"  {k:<14} {c[k]:>6}  {c[k] / n:.3f}{share_wrong}")
    ans = [x for x in rows if x["gt_rank"] is not None]
    if ans:
        ranks = sorted(x["gt_rank"] for x in ans)
        print(f"  reference in list: {len(ans)}; its model rank median {ranks[len(ranks) // 2]}, "
              f"rank 2: {sum(r == 2 for r in ranks)}, rank 3-5: {sum(3 <= r <= 5 for r in ranks)}, rank >5: {sum(r > 5 for r in ranks)}")
        ok = [x["top1_p"] for x in ans if x["relation"] == "correct"]
        bad = [x["top1_p"] for x in ans if x["relation"] != "correct"]
        if ok and bad:
            print(f"  mean top-1 probability: {sum(ok) / len(ok):.3f} when correct, {sum(bad) / len(bad):.3f} when wrong")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(shown)
        print(f"wrote {len(shown)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
