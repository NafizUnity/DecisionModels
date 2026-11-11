"""No-notes evaluation on retrieval-ranking files (keys: entity_text, query_text, ground_truth_codes, rankings{source: [codes]}).

  python decision_eval.py build --src FILE.json --run-dir results/NAME       # requests.jsonl (contains entity text; keep it with the data)
  python run_decision_model.py run ELEPHANT --gpu 1 --run-dir results/NAME          # existing runner, one model at a time, resumable
  python decision_eval.py report --run-dir results/NAME                       # tables in results/NAME

Options are bare codes, so scores measure whether a model can pick a code without its description.
Original order is the pipeline as ranked; shuf1 removes position as a cue.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core  # noqa: E402
from run_decision_model import LETTERS, cell_metrics, order_for  # noqa: E402

KS = (5, 10, 20)
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
            top = ranking[:max(KS)] if isinstance(ranking, list) else []
            if not all(isinstance(c, str) and c and c.strip() == c for c in top):
                problems.append(f"{where}: {source} has codes that are not non-empty strings without spaces")
            elif len(top) < max(KS) or len(set(top)) < max(KS):
                problems.append(f"{where}: {source} needs {max(KS)} unique codes in its top {max(KS)}, has {len(set(top))}")
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
    rows, hits = [], {(s, k): 0 for s in SOURCES for k in KS}
    for d in data:
        gt = d["ground_truth_codes"][0]
        eid = str(d["entity_id"])
        state = merged_context(d["entity_text"], d.get("query_text") or "")
        for source in SOURCES:
            ranking = d["rankings"][source]
            for k in KS:
                top = ranking[:k]
                hits[(source, k)] += gt in top
                for order in ORDERS:
                    codes = order_for(top, order, f"{eid}|{source}", k)
                    letters = LETTERS[:k]
                    rows.append({"request_id": f"{eid}|{source}|top_{k}|{order}", "entity_id": eid, "source": source,
                                 "k": k, "order": order, "gt_code": gt, "answerable": gt in codes,
                                 "gt_letter": letters[codes.index(gt)] if gt in codes else None, "rank1_correct": top[0] == gt, "codes": codes,
                                 "record": {"state": state, "questions": {lh_core.QUESTION_ID: {"type": "choice", "instructions": QUESTION,
                                                                                           "criteria": dict(zip(letters, codes))}}}})
    print("recall (reference code in the top k):")
    for source in SOURCES:
        print(f"  {source:16} " + "  ".join(f"top{k} {hits[(source, k)] / len(data):.3f}" for k in KS))
    if not any(hits[(s, max(KS))] for s in SOURCES):
        sys.exit(f"STOPPED: no reference code appears in any top-{max(KS)} list. The code formats probably differ "
                 "(dots, case or spaces); run inspect_rankings.py and fix the file. Nothing was written.")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "requests.jsonl.tmp"
    with open(tmp, "w", encoding="utf8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out / "requests.jsonl")
    print(f"wrote {len(rows)} requests for {len(data)} entities -> {out/'requests.jsonl'}")


def cmd_report(args):
    out = Path(args.run_dir).resolve()
    reqs = {r["request_id"]: r for r in map(json.loads, open(out / "requests.jsonl", encoding="utf8"))}
    requests_sha = hashlib.sha256((out / "requests.jsonl").read_bytes()).hexdigest()
    preds = {}
    for model in NAMES:
        path = out / f"predictions_{model}.jsonl"
        if path.exists():
            meta = path.with_name(path.name + ".meta.json")
            recorded = json.loads(meta.read_text()).get("requests_sha256") if meta.exists() else None
            if recorded != requests_sha:
                print(f"WARNING: skipping {path.name}: it was not made from this requests.jsonl"
                      + ("" if meta.exists() else f" ({meta.name} is missing)"))
                continue
            preds[model] = {p["request_id"]: p for p in lh_core.read_jsonl_readonly(path)[0] if p.get("status") == "ok"}
    if not preds:
        sys.exit("no predictions found")
    groups = {}
    for i, r in reqs.items():
        groups.setdefault((r["source"], r["k"], r["order"]), []).append(i)
    rows = []
    for (source, k, order), ids in sorted(groups.items(), key=lambda x: (SOURCES.index(x[0][0]), x[0][1], x[0][2])):
        for model, done in preds.items():
            m = cell_metrics([(reqs[i], done[i]) for i in ids if i in done])
            rows.append({"source": source, "k": k, "order": order, "model": NAMES[model], "complete": m["n"] == len(ids), "n": m["n"], "n_expected": len(ids),
                         "recall": sum(reqs[i]["answerable"] for i in ids) / len(ids),
                         "rank1_baseline": sum(reqs[i]["rank1_correct"] for i in ids) / len(ids),
                         "acc_overall": m["acc_all"], "acc_when_answerable": m["acc"], "brier_answerable": m["brier"], "ece_answerable": m["ece"]})
    with open(out / "ranking_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["# Decision models on retrieval rankings (no notes, bare codes)", "",
          "Evaluation design note: options are bare codes with no description. The context is the entity text plus its expansion.",
          "Acc overall counts items whose reference code is not in the list as wrong; Acc | present, Brier and ECE use only items where it is present.", ""]
    for order in ORDERS:
        md += [f"## Option order: {order}", "", "| Source | Top-k | Model | Recall | Always rank 1 | Acc overall | Acc \\| present | Brier | ECE |", "|---|---|---|---|---|---|---|---|---|"]
        for r in (x for x in rows if x["order"] == order):
            flag = "" if r["complete"] else f" INCOMPLETE {r['n']}/{r['n_expected']}"
            md.append(f"| {r['source']} | {r['k']} | {r['model']}{flag} | {r['recall']:.3f} | {r['rank1_baseline']:.3f} | {r['acc_overall']:.3f} | "
                      f"{r['acc_when_answerable']:.3f} | {r['brier_answerable']:.3f} | {r['ece_answerable']:.3f} |")
        md.append("")
    (out / "ranking_tables.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(f"wrote {out/'ranking_results.csv'} and {out/'ranking_tables.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--src", required=True); b.add_argument("--run-dir", required=True)
    r = sub.add_parser("report"); r.add_argument("--run-dir", required=True)
    args = p.parse_args()
    {"build": cmd_build, "report": cmd_report}[args.cmd](args)
