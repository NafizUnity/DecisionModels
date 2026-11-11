"""No-notes evaluation on retrieval-ranking files (keys: entity_id, entity_text, query_text, ground_truth_codes, rankings{source: [codes]}).

  python decision_eval.py build --src FILE.json --run-dir results/NAME       # requests.jsonl (contains entity text; keep it with the data)
  python run_decision_model.py run ELEPHANT --gpu 1 --run-dir results/NAME   # one model at a time, resumable
  python decision_eval.py report --run-dir results/NAME [--smoke]            # tables in results/NAME

Each request gives the model EVERY code in one source's ranking (at most 255, the limit of all three models). Top-k is
then taken from the model's own probabilities: hit@k = the reference code is among the model's k most likely codes. The
retriever's hit@k on the same lists is the baseline. Each option's key is the code itself with no description, so the
models see e.g. "A. E11.9" (Elephant), "A: E11.9" (Drex), {"option_id": "E11.9"} (Clef, which sorts options by key, so
its input does not depend on the order). Original order is the pipeline as ranked; shuf1 removes position as a cue.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core  # noqa: E402
from run_decision_model import cell_metrics, load_requests, order_for  # noqa: E402

KS = (1, 5, 10, 20)          # cutoffs for hit@k, taken from the ranking over ALL options (not a cut of the input)
MAX_OPTIONS = 255            # Elephant (schema MAX_CHOICE_OPTIONS) and Drex (255 answer labels); Clef has no fixed limit
ORDERS = ("original", "shuf1")
SOURCES = ("csv_dense", "snomed_fsn", "snomed_syn", "rrf_csv+fsn", "rrf_csv+syn", "rrf_fsn+syn", "rrf_csv+fsn+syn")
NAMES = {"ELEPHANT": "Elephant-1", "CLEF": "Clef", "DREX": "Drex Max"}
QUESTION = "Which ICD-10-CA code best matches the diagnosis described above?"


def merged_context(entity, query):
    entity, query = entity.strip(), query.strip()
    return entity if entity.lower() == query.lower() else f"{entity} (expanded: {query})"


def validate(data):
    """Every problem in the file (entity ids and code values only; no entity or query text). Empty list = usable."""
    if not isinstance(data, list) or not data:
        return ["the file must hold a non-empty JSON list of entities"]
    problems, seen = [], set()
    for n, d in enumerate(data):
        eid = d.get("entity_id") if isinstance(d, dict) else None
        where = f"entity #{n} (entity_id {eid!r})"
        if not isinstance(d, dict):
            problems.append(f"entity #{n} is not an object"); continue
        missing = [k for k in ("entity_id", "entity_text", "ground_truth_codes", "rankings") if k not in d]
        if missing:
            problems.append(f"{where}: missing keys {missing}"); continue
        if not isinstance(eid, (str, int)) or str(eid) == "" or "|" in str(eid):
            problems.append(f"{where}: entity_id must be a non-empty string or number without '|'")
        elif str(eid) in seen:
            problems.append(f"{where}: duplicate entity_id")
        seen.add(str(eid))
        if not isinstance(d["entity_text"], str) or not d["entity_text"].strip():
            problems.append(f"{where}: empty entity_text")
        if d.get("query_text") is not None and not isinstance(d["query_text"], str):
            problems.append(f"{where}: query_text is not a string")
        gts = d["ground_truth_codes"]
        if not isinstance(gts, list) or len(set(map(str, gts))) != 1 or not all(isinstance(c, str) and c.strip() == c and c for c in gts):
            problems.append(f"{where}: needs exactly one ground-truth code (a non-empty string without spaces), found {gts!r}")
        if not isinstance(d["rankings"], dict):
            problems.append(f"{where}: rankings is not an object"); continue
        for source in SOURCES:
            ranking = d["rankings"].get(source)
            if ranking is None:
                problems.append(f"{where}: no ranking '{source}' (found: {sorted(d['rankings'])})"); continue
            if not isinstance(ranking, list) or not ranking:
                problems.append(f"{where}: {source} is not a non-empty list of codes"); continue
            if not all(isinstance(c, str) and c and c.strip() == c and " " not in c for c in ranking):
                problems.append(f"{where}: {source} has codes that are not non-empty strings without spaces")
            elif len(set(ranking)) < len(ranking):
                problems.append(f"{where}: {source} lists {len(ranking) - len(set(ranking))} code(s) more than once")
            elif len(ranking) > MAX_OPTIONS:
                problems.append(f"{where}: {source} has {len(ranking)} codes; the models take at most {MAX_OPTIONS}")
    return problems


def cmd_build(args):
    data = json.load(open(args.src, encoding="utf8"))
    out = Path(args.run_dir).resolve()
    if (out / "requests.jsonl").exists() or list(out.glob("predictions_*.jsonl")):
        sys.exit(f"{out} already has requests or predictions; use a new --run-dir")
    problems = validate(data)
    if problems:
        for line in problems[:30]:
            print("  " + line)
        sys.exit(f"STOPPED: {len(problems)} problem(s) in {args.src} (first {min(30, len(problems))} above); nothing was written")
    rows = []
    lengths = {source: [] for source in SOURCES}
    retr_rank = {source: [] for source in SOURCES}
    for d in data:
        gt = d["ground_truth_codes"][0]
        eid = str(d["entity_id"])
        state = merged_context(d["entity_text"], d.get("query_text") or "")
        for source in SOURCES:
            ranking = list(d["rankings"][source])
            rank = ranking.index(gt) + 1 if gt in ranking else None      # the retriever's rank of the reference code
            lengths[source].append(len(ranking))
            retr_rank[source].append(rank)
            for order in ORDERS:
                codes = order_for(ranking, order, f"{eid}|{source}", "all")
                rows.append({"request_id": f"{eid}|{source}|{order}", "entity_id": eid, "source": source, "order": order,
                             "n_options": len(codes), "gt_code": gt, "answerable": rank is not None,
                             "gt_letter": gt if rank is not None else None,     # the option key of the reference code
                             "gt_position": codes.index(gt) + 1 if rank is not None else None,
                             "retriever_rank": rank, "rank1_correct": rank == 1, "codes": codes,
                             "record": {"state": state, "questions": {lh_core.QUESTION_ID: {
                                 "type": "choice", "instructions": QUESTION, "criteria": {c: None for c in codes}}}}})
    print(f"{len(data)} entities. Every request holds a source's whole ranking. Retriever: share of entities with the "
          f"reference code within its top k:")
    print(f"  {'source':<16} {'codes min/median/max':>20}  " + "  ".join(f"@{k:<4}" for k in KS) + "  in list")
    for source in SOURCES:
        L, R = lengths[source], retr_rank[source]
        print(f"  {source:<16} {min(L):>6}/{statistics.median(L):>5g}/{max(L):<6}  "
              + "  ".join(f"{sum(r is not None and r <= k for r in R) / len(R):.3f}" for k in KS)
              + f"  {sum(r is not None for r in R) / len(R):.3f}")
    if not any(r is not None for R in retr_rank.values() for r in R):
        sys.exit("STOPPED: no reference code appears in any ranking. The code formats probably differ "
                 "(dots, case or spaces); run inspect_rankings.py and fix the file. Nothing was written.")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "requests.jsonl.tmp"
    with open(tmp, "w", encoding="utf8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out / "requests.jsonl")
    print(f"wrote {len(rows)} requests for {len(data)} entities ({sum(r['n_options'] for r in rows)} options in all) -> {out/'requests.jsonl'}")


def model_rank(r, p):
    """1-based rank of the reference code in the model's ordering of all options (None if it is not an option).
    Ties go to the earlier position, the same rule that picks the prediction."""
    if not r["answerable"]:
        return None
    probs, keys = p["probabilities"], list(r["record"]["questions"][lh_core.QUESTION_ID]["criteria"])
    g = probs[r["gt_letter"]]
    pos = keys.index(r["gt_letter"])
    return 1 + sum(v > g for v in probs.values()) + sum(probs[k] == g for k in keys[:pos])


def hit_metrics(items):
    """hit@k for the model and for the retriever on the same requests (a request without the reference code is a miss), MRR."""
    n = len(items)
    nan = float("nan")
    ranks = [model_rank(r, p) for r, p in items]
    out = {"n": n}
    for k in KS:
        out[f"model_hit@{k}"] = sum(x is not None and x <= k for x in ranks) / n if n else nan
        out[f"retriever_hit@{k}"] = sum(r["retriever_rank"] is not None and r["retriever_rank"] <= k for r, _ in items) / n if n else nan
    out["model_mrr"] = sum(1 / x for x in ranks if x) / n if n else nan
    out["retriever_mrr"] = sum(1 / r["retriever_rank"] for r, _ in items if r["retriever_rank"]) / n if n else nan
    out["in_list"] = sum(r["answerable"] for r, _ in items) / n if n else nan
    out["tied_top"] = sum(sorted(p["probabilities"].values())[-1] == sorted(p["probabilities"].values())[-2]
                          for _, p in items if len(p["probabilities"]) > 1)
    return out


def cmd_report(args):
    out = Path(args.run_dir).resolve()
    reqs = {r["request_id"]: r for r in map(json.loads, open(out / "requests.jsonl", encoding="utf8"))}
    if "n_options" not in next(iter(reqs.values())):
        sys.exit(f"STOPPED: {out} was built by the earlier top-k version (5/10/20 options per request); this report reads "
                 "runs where every request holds the whole ranking. Its exports in option_probs/ are still valid.")
    requests_sha = hashlib.sha256((out / "requests.jsonl").read_bytes()).hexdigest()
    preds, selected = {}, {}
    suffix = "_smoke" if args.smoke else ""
    for model in NAMES:
        path = out / f"predictions_{model}{suffix}.jsonl"
        if path.exists():
            meta = path.with_name(path.name + ".meta.json")
            recorded = json.loads(meta.read_text()).get("requests_sha256") if meta.exists() else None
            if recorded != requests_sha:
                print(f"WARNING: skipping {path.name}: it was not made from this requests.jsonl"
                      + ("" if meta.exists() else f" ({meta.name} is missing)"))
                continue
            preds[model] = {p["request_id"]: p for p in lh_core.read_jsonl_readonly(path)[0] if p.get("status") == "ok"}
            # a smoke run covers only the requests its --stride selected; score each model against its own selection
            selected[model] = {r["request_id"] for r in load_requests(out, json.loads(meta.read_text()).get("stride", 1))}
    if not preds:
        sys.exit(f"no {'smoke-test ' if args.smoke else ''}predictions found in {out}"
                 + ("" if args.smoke else " (for the smoke test, add --smoke)"))
    hit_cols = [f"model_hit@{k}" for k in KS] + ["model_mrr"] + [f"retriever_hit@{k}" for k in KS] + ["retriever_mrr"]
    rows = []
    for source in SOURCES:
        for order in ORDERS:
            for model, done in preds.items():
                mine = [i for i, r in reqs.items() if r["source"] == source and r["order"] == order and i in selected[model]]
                items = [(reqs[i], done[i]) for i in mine if i in done]
                h, c = hit_metrics(items), cell_metrics(items)
                rows.append({"source": source, "order": order, "model": NAMES[model], "n": h["n"], "n_expected": len(mine),
                             "median_options": statistics.median(reqs[i]["n_options"] for i in mine) if mine else float("nan"),
                             "in_list": h["in_list"], **{k: h[k] for k in hit_cols},
                             "acc_when_in_list": c["acc"], "brier_when_in_list": c["brier"], "ece_when_in_list": c["ece"],
                             "tied_top": h["tied_top"]})
    summary = []   # pooled over sources: a smoke run has only a few requests per source
    for model, done in preds.items():
        for order in ORDERS:
            items = [(reqs[i], done[i]) for i in sorted(selected[model]) if i in done and reqs[i]["order"] == order]
            summary.append((NAMES[model], order, hit_metrics(items), sum(reqs[i]["order"] == order for i in selected[model])))
    head = f"{'model':<11} {'order':<9} {'n ok':>11} " + " ".join(f"{'hit@' + str(k):>6}" for k in KS) + f" {'MRR':>6}  | retriever " \
           + " ".join(f"{'@' + str(k):>5}" for k in KS) + f" {'MRR':>6} | tied"
    print(head)
    for name, order, h, n_exp in summary:
        print(f"{name:<11} {order:<9} {h['n']:>5}/{n_exp:<5} " + " ".join(f"{h[f'model_hit@{k}']:>6.3f}" for k in KS)
              + f" {h['model_mrr']:>6.3f}  |           " + " ".join(f"{h[f'retriever_hit@{k}']:>5.3f}" for k in KS)
              + f" {h['retriever_mrr']:>6.3f} | {h['tied_top']}")
    with open(out / f"ranking_results{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["# Decision models re-ranking whole retrieval lists (no notes, bare codes)" + (" - SMOKE TEST (a small sample; not results)" if args.smoke else ""), "",
          "Each request gives the model every code in one source's ranking. hit@k: the reference code is among the model's k most "
          "likely codes (ties go to the earlier position); requests whose list lacks the reference code count as misses. "
          "Retriever columns: the same measure for the source's own ranking on the same requests. MRR: mean of 1/rank (0 if absent). "
          "Acc, Brier and ECE use only requests whose list holds the reference code. Clef sorts options by code, so its input "
          "is the same in both orders.", "",
          "## Summary (all sources pooled)", "",
          "| Model | Order | n ok | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR | " + " | ".join(f"retriever @{k}" for k in KS) + " | retriever MRR | Tied top |",
          "|---" * (5 + 2 * len(KS) + 2) + "|"]
    md += [f"| {name} | {order} | {h['n']}/{n_exp} | " + " | ".join(f"{h[f'model_hit@{k}']:.3f}" for k in KS) + f" | {h['model_mrr']:.3f} | "
           + " | ".join(f"{h[f'retriever_hit@{k}']:.3f}" for k in KS) + f" | {h['retriever_mrr']:.3f} | {h['tied_top']} |"
           for name, order, h, n_exp in summary] + [""]
    for order in ORDERS:
        md += [f"## Option order: {order}", "",
               "| Source | Model | n | Options (median) | In list | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR | "
               + " | ".join(f"retr @{k}" for k in KS) + " | retr MRR | Acc \\| in list | Brier | ECE |",
               "|---" * (8 + 2 * len(KS) + 3) + "|"]
        for r in (x for x in rows if x["order"] == order):
            flag = "" if r["n"] == r["n_expected"] else f" INCOMPLETE {r['n']}/{r['n_expected']}"
            md.append(f"| {r['source']} | {r['model']}{flag} | {r['n']} | {r['median_options']:g} | {r['in_list']:.3f} | "
                      + " | ".join(f"{r[f'model_hit@{k}']:.3f}" for k in KS) + f" | {r['model_mrr']:.3f} | "
                      + " | ".join(f"{r[f'retriever_hit@{k}']:.3f}" for k in KS) + f" | {r['retriever_mrr']:.3f} | "
                      f"{r['acc_when_in_list']:.3f} | {r['brier_when_in_list']:.3f} | {r['ece_when_in_list']:.3f} |")
        md.append("")
    (out / f"ranking_tables{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(f"wrote {out/f'ranking_results{suffix}.csv'} and {out/f'ranking_tables{suffix}.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--src", required=True); b.add_argument("--run-dir", required=True)
    r = sub.add_parser("report"); r.add_argument("--run-dir", required=True)
    r.add_argument("--smoke", action="store_true", help="report the smoke-test predictions (written as *_smoke.csv/.md)")
    args = p.parse_args()
    {"build": cmd_build, "report": cmd_report}[args.cmd](args)
