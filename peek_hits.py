#!/usr/bin/env python3
"""Print hit@k for finished runs to the terminal, from the saved predictions. Writes no files.

  python peek_hits.py --run-dir RESULTS_DIR/my_run --models "DREX MEDDECIDER31" --ks "1,5,6,10" \
      --sources "rrf_csv+fsn rrf_fsn+syn" --orders original

Same numbers as decision_eval.py report (same requests, same per-model option cut, N/A above a model's limit),
but nothing in the run folder is changed.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core  # noqa: E402
from decision_eval import NAMES, ORDERS, SOURCES, f3, hit_metrics, parse_ks  # noqa: E402
from run_decision_model import model_view, parse_sources, retriever_rankings, select_requests  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ks", type=parse_ks, required=True, metavar="'1,5,6,10'")
    ap.add_argument("--models", type=parse_sources, default=None, help="default: every model with predictions")
    ap.add_argument("--sources", type=parse_sources, default=None, help="default: all")
    ap.add_argument("--orders", type=parse_sources, default=None, help="default: all")
    ap.add_argument("--smoke", action="store_true", help="read the smoke-test predictions instead")
    args = ap.parse_args()

    out = Path(args.run_dir).resolve()
    req_path = out / "requests.jsonl"
    reqs = {r["request_id"]: r for r in map(json.loads, open(req_path, encoding="utf8"))}
    sha = hashlib.sha256(req_path.read_bytes()).hexdigest()
    for name, given, allowed in (("source", args.sources, SOURCES), ("order", args.orders, ORDERS)):
        bad = sorted(set(given or ()) - set(allowed))
        if bad:
            sys.exit(f"STOPPED: unknown {name}(s) {bad}; choose from {' '.join(allowed)}")
    models = [m.upper() for m in args.models] if args.models else list(NAMES)
    bad = [m for m in models if m not in NAMES]
    if bad:
        sys.exit(f"STOPPED: unknown model(s) {bad}; choose from {' '.join(NAMES)}")
    suffix = "_smoke" if args.smoke else ""
    longest_all = max(r["n_options"] for r in reqs.values())
    rankings = None
    KS = args.ks
    rows = []
    for m in models:
        path = out / f"predictions_{m}{suffix}.jsonl"
        meta = path.with_name(path.name + ".meta.json")
        if not path.exists():
            if args.models:
                print(f"WARNING: {path.name} not found; {m} skipped")
            continue
        info = json.loads(meta.read_text()) if meta.exists() else {}
        if info.get("requests_sha256") != sha:
            print(f"WARNING: {path.name} was not made from this requests.jsonl; {m} skipped")
            continue
        done = {p["request_id"]: p for p in lh_core.read_jsonl_readonly(path)[0] if p.get("status") == "ok"}
        selected = {r["request_id"] for r in select_requests(list(reqs.values()), info.get("stride", 1), args.sources, args.orders)}
        limit = lh_core.MODELS_INFO[m]["max_options"]
        capped = limit < longest_all
        if capped and rankings is None:
            rankings = retriever_rankings(reqs.values())
        for source in [s for s in SOURCES if not args.sources or s in args.sources]:
            for order in [o for o in ORDERS if not args.orders or o in args.orders]:
                mine = [i for i, r in reqs.items() if r["source"] == source and r["order"] == order and i in selected]
                if not mine:
                    continue
                view = (lambda i: model_view(reqs[i], limit, rankings)) if capped else (lambda i: reqs[i])
                items = [(view(i), done[i]) for i in mine if i in done]
                h = hit_metrics(items, KS, limit if capped else None)
                rows.append((source, order, NAMES[m], f"{h['n']}/{len(mine)}", limit, h))
    if not rows:
        sys.exit("no predictions found for that selection")

    head = ["source", "order", "model", "done", "max_opt"] + [f"hit@{k}" for k in KS]
    table = [head]
    for source, order, name, n, limit, h in rows:
        table.append([source, order, name, n, str(limit)] + [f3(h[f"model_hit@{k}"]) for k in KS])
    # the retriever on the same requests, once per source/order (it does not depend on the model)
    seen = set()
    for source, order, name, n, limit, h in rows:
        if (source, order) in seen:
            continue
        seen.add((source, order))
        table.append([source, order, "(retriever)", n, "-"] + [f3(h[f"retriever_hit@{k}"]) for k in KS])
    widths = [max(len(r[c]) for r in table) for c in range(len(head))]
    for j, r in enumerate(table):
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if j == 0:
            print("  ".join("-" * w for w in widths))
    if any(math.isnan(h[f"model_hit@{k}"]) for *_, h in rows for k in KS):
        print("\nN/A: k is above the number of options that model reads (max_opt).")


if __name__ == "__main__":
    main()
