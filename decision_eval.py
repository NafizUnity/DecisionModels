"""No-notes evaluation on retrieval-ranking files (keys: entity_id, entity_text, query_text, ground_truth_codes, rankings{source: [codes]}).

  python decision_eval.py build --src FILE.json --run-dir results/NAME       # requests.jsonl (contains entity text; keep it with the data)
      [--descriptions icd_snomed_index.jsonl]                                 # options carry each code's long description
      [--top K]                                                               # only the retriever's first K codes per list
  python run_decision_model.py run ELEPHANT --gpu 1 --run-dir results/NAME   # one model at a time, resumable
  python decision_eval.py report --run-dir results/NAME [--smoke]            # tables in results/NAME

Each request gives the model EVERY code in one source's ranking (at most 255, the limit of all three models). Top-k is
then taken from the model's own probabilities: hit@k = the reference code is among the model's k most likely codes. The
retriever's hit@k on the same lists is the baseline. Each option's key is the code itself with no description, so the
models see e.g. "A. E11.9" (Elephant), "A: E11.9" (Drex), {"option_id": "E11.9"} (Clef, which sorts options by key, so
its input does not depend on the order). Original order is the pipeline as ranked; shuf1 removes position as a cue.
With --descriptions, each option's value is the code's long_description (short_description when long is empty), so the
models see "A. E119: Type 2 diabetes ..." / "A: E119: ..." / {"description": "...", "option_id": "E119"}; every model
adds the code itself. check_descriptions.py checks an index before such a build.
With --top K, each request holds only the first K codes of the source's ranking (shorter lists are kept whole); a
reference code ranked below K is then not an option and counts as a miss, so the ceiling is the retriever's hit@K.
Without --top, requests and option orders are exactly those of the whole-list runs.
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
from run_decision_model import cell_metrics, load_requests, n_tied_top, order_for, rank_options  # noqa: E402

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
            elif len(ranking) < 2:
                problems.append(f"{where}: {source} has {len(ranking)} code; a choice needs at least 2")
            elif len(ranking) > MAX_OPTIONS:
                problems.append(f"{where}: {source} has {len(ranking)} codes; the models take at most {MAX_OPTIONS}")
    return problems


def load_descriptions(path):
    """code -> long_description (short_description when long is empty), whitespace collapsed. Stops on a bad line,
    a record without a code or description, or a code listed twice with different text."""
    desc = {}
    with open(path, encoding="utf8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                sys.exit(f"STOPPED: {path} line {n} is not JSON; nothing was written")
            code = rec.get("code") if isinstance(rec, dict) else None
            text = next((" ".join(t.split()) for t in (rec.get("long_description"), rec.get("short_description"))
                         if isinstance(t, str) and t.strip()), None) if code else None
            if not isinstance(code, str) or not text:
                sys.exit(f"STOPPED: {path} line {n} has no code or no description; nothing was written")
            if desc.get(code, text) != text:
                sys.exit(f"STOPPED: {path} lists {code} twice with different descriptions; nothing was written")
            desc[code] = text
    return desc


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
    if args.top is not None and args.top < 2:
        sys.exit(f"STOPPED: --top must be at least 2 (a choice needs two options), got {args.top}; nothing was written")
    top = args.top
    desc = load_descriptions(args.descriptions) if args.descriptions else None
    option_text = "long_description" if desc else "code"
    if desc:
        missing = sorted({c for d in data for s in SOURCES for c in d["rankings"][s][:top] if c not in desc})
        if missing:
            sys.exit(f"STOPPED: {len(missing)} list code(s) have no description in {args.descriptions} "
                     f"(first: {missing[:20]}); run check_descriptions.py. Nothing was written")
    rows = []
    lengths = {source: [] for source in SOURCES}
    retr_rank = {source: [] for source in SOURCES}
    for d in data:
        gt = d["ground_truth_codes"][0]
        eid = str(d["entity_id"])
        state = merged_context(d["entity_text"], d.get("query_text") or "")
        for source in SOURCES:
            ranking = list(d["rankings"][source])[:top]                  # top=None keeps the whole ranking
            rank = ranking.index(gt) + 1 if gt in ranking else None      # the retriever's rank of the reference code
            lengths[source].append(len(ranking))
            retr_rank[source].append(rank)
            for order in ORDERS:
                # the shuffle seed keeps "all" for whole lists, so those runs reproduce the earlier option orders
                codes = order_for(ranking, order, f"{eid}|{source}", "all" if top is None else f"top{top}")
                rows.append({"request_id": f"{eid}|{source}|{order}", "entity_id": eid, "source": source, "order": order,
                             "n_options": len(codes), "gt_code": gt, "answerable": rank is not None,
                             "gt_letter": gt if rank is not None else None,     # the option key of the reference code
                             "gt_position": codes.index(gt) + 1 if rank is not None else None,
                             "retriever_rank": rank, "rank1_correct": rank == 1, "codes": codes, "option_text": option_text,
                             "top": top,
                             "record": {"state": state, "questions": {lh_core.QUESTION_ID: {
                                 "type": "choice", "instructions": QUESTION,
                                 "criteria": {c: desc[c] if desc else None for c in codes}}}}})
    held = "a source's whole ranking" if top is None else f"the first {top} codes of a source's ranking"
    print(f"{len(data)} entities. Every request holds {held}. Retriever: share of entities with the reference code "
          f"within its top k" + (f" (k above {top} cannot exceed @{top}):" if top else ":"))
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
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()   # noqa: E731
    lh_core.write_json_atomic(out / "build_meta.json", {
        "src": str(Path(args.src).resolve()), "src_sha256": sha(args.src), "option_text": option_text,
        "descriptions": str(Path(args.descriptions).resolve()) if desc else None,
        "descriptions_sha256": sha(args.descriptions) if desc else None, "top": top, "requests": len(rows)})
    print(f"options: {'code + ' + option_text if desc else 'bare codes'}; {'whole lists' if top is None else f'top {top} per list'}")
    print(f"wrote {len(rows)} requests for {len(data)} entities ({sum(r['n_options'] for r in rows)} options in all) -> {out/'requests.jsonl'}")


def model_rank(r, p):
    """1-based rank of the reference code in the model's ordering of all options (None if it is not an option).
    Ties go to the earlier position, the same rule that picks the prediction."""
    if not r["answerable"]:
        return None
    return rank_options(list(r["record"]["questions"][lh_core.QUESTION_ID]["criteria"]), p["probabilities"])[r["gt_letter"]]


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
    out["tied_top"] = sum(n_tied_top(p["probabilities"]) > 1 for _, p in items)
    return out


def separator(header):
    """The |---| row for a markdown header row (escaped pipes inside a cell do not count as columns)."""
    return "|---" * (header.replace("\\|", "").count("|") - 1) + "|"


def clef_order_check(reqs, done):
    """Clef sorts options by key, so its original and shuf1 inputs are identical and their probabilities should match.
    Returns (pairs compared, pairs whose probabilities differ by more than 1e-6 for some code)."""
    pairs = differ = 0
    for i, p in done.items():
        twin = done.get(i.rsplit("|", 1)[0] + "|shuf1") if reqs[i]["order"] == "original" else None
        if twin is None:
            continue
        pairs += 1
        differ += any(abs(v - twin["probabilities"][k]) > 1e-6 for k, v in p["probabilities"].items())
    return pairs, differ


def cmd_report(args):
    out = Path(args.run_dir).resolve()
    reqs = {r["request_id"]: r for r in map(json.loads, open(out / "requests.jsonl", encoding="utf8"))}
    if not reqs:
        sys.exit(f"STOPPED: {out / 'requests.jsonl'} is empty; build it again into a new --run-dir")
    if any("n_options" not in r for r in reqs.values()):
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
    if "CLEF" in preds:
        pairs, differ = clef_order_check(reqs, preds["CLEF"])
        clef_line = (f"Clef order check: {pairs} original/shuf1 pairs compared, {differ} with different probabilities"
                     + (" (expected 0: Clef sorts options by code; investigate)" if differ else " (as expected)")
                     + ("" if pairs else " (no pairs: the smoke sample held no matching requests)"))
        print(clef_line)
    print(head)
    for name, order, h, n_exp in summary:
        print(f"{name:<11} {order:<9} {h['n']:>5}/{n_exp:<5} " + " ".join(f"{h[f'model_hit@{k}']:>6.3f}" for k in KS)
              + f" {h['model_mrr']:>6.3f}  |           " + " ".join(f"{h[f'retriever_hit@{k}']:>5.3f}" for k in KS)
              + f" {h['retriever_mrr']:>6.3f} | {h['tied_top']}")
    with open(out / f"ranking_results{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = "codes with descriptions" if any(r.get("option_text", "code") != "code" for r in reqs.values()) else "bare codes"
    tops = {r.get("top") for r in reqs.values()}
    lists = "whole retrieval lists" if tops == {None} else f"the retriever's top {'/'.join(map(str, sorted(t or 0 for t in tops)))}"
    md = [f"# Decision models re-ranking {lists} (no notes, {text})" + (" - SMOKE TEST (a small sample; not results)" if args.smoke else ""), "",
          ("Each request gives the model every code in one source's ranking. " if tops == {None} else
           f"Each request gives the model the first {'/'.join(map(str, sorted(t or 0 for t in tops)))} codes of one source's ranking "
           "(a reference code ranked lower is not an option and counts as a miss). ") + "hit@k: the reference code is among the model's k most "
          "likely codes (ties go to the earlier position); requests whose list lacks the reference code count as misses. "
          "Retriever columns: the same measure for the source's own ranking on the same requests. MRR: mean of 1/rank (0 if absent). "
          "Acc, Brier and ECE use only requests whose list holds the reference code. Clef sorts options by code, so its input "
          "is the same in both orders.", "",
          *([clef_line, ""] if "CLEF" in preds else []),
          "## Summary (all sources pooled)", "",
          "| Model | Order | n ok | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR | " + " | ".join(f"retriever @{k}" for k in KS) + " | retriever MRR | Tied top |"]
    md.append(separator(md[-1]))
    md += [f"| {name} | {order} | {h['n']}/{n_exp} | " + " | ".join(f"{h[f'model_hit@{k}']:.3f}" for k in KS) + f" | {h['model_mrr']:.3f} | "
           + " | ".join(f"{h[f'retriever_hit@{k}']:.3f}" for k in KS) + f" | {h['retriever_mrr']:.3f} | {h['tied_top']} |"
           for name, order, h, n_exp in summary] + [""]
    for order in ORDERS:
        md += [f"## Option order: {order}", "",
               "| Source | Model | n | Options (median) | In list | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR | "
               + " | ".join(f"retr @{k}" for k in KS) + " | retr MRR | Acc \\| in list | Brier | ECE |"]
        md.append(separator(md[-1]))
        for r in (x for x in rows if x["order"] == order):
            flag = "" if r["n"] == r["n_expected"] else f" INCOMPLETE {r['n']}/{r['n_expected']}"
            md.append(f"| {r['source']} | {r['model']}{flag} | {r['n']} | {r['median_options']:g} | {r['in_list']:.3f} | "
                      + " | ".join(f"{r[f'model_hit@{k}']:.3f}" for k in KS) + f" | {r['model_mrr']:.3f} | "
                      + " | ".join(f"{r[f'retriever_hit@{k}']:.3f}" for k in KS) + f" | {r['retriever_mrr']:.3f} | "
                      f"{r['acc_when_in_list']:.3f} | {r['brier_when_in_list']:.3f} | {r['ece_when_in_list']:.3f} |")
        md.append("")
    (out / f"ranking_tables{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(f"wrote {out/f'ranking_results{suffix}.csv'} and {out/f'ranking_tables{suffix}.md'}")
    write_comparison(out, suffix)


def write_comparison(out, suffix=""):
    """One table per run: for every source, the retriever's own ranking (baseline) and each model x order, with hit@1/5/10/20
    and MRR and the difference from the baseline. Read from ranking_results.csv, so it needs no predictions."""
    with open(out / f"ranking_results{suffix}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"STOPPED: {out / f'ranking_results{suffix}.csv'} is empty")
    metrics = [f"hit@{k}" for k in KS] + ["mrr"]
    sources = [s for s in SOURCES if any(r["source"] == s for r in rows)]
    shown = [f"hit@{k}" for k in KS]
    md = ["# Model picks vs the retriever's own ranking (baseline)" + (" - SMOKE TEST (not results)" if suffix else ""), "",
          "top-k: share of entities whose reference code is among the picker's k highest-ranked codes. shuf1 = options "
          "shuffled (the model cannot use the retriever's order); original = retriever's order. MRR and the differences "
          "from the baseline are in ranking_comparison.csv.", "",
          "| Source | Picker | " + " | ".join(f"top-{k}" for k in KS) + " |"]
    md.append(separator(md[-1]))
    flat = []
    for source in sources:
        mine = [r for r in rows if r["source"] == source]
        base = mine[0]   # every model x order of a source covers the same entities, so the retriever values agree
        ref = {m: float(base[f"retriever_{m}"]) for m in metrics}
        md.append(f"| {source} | Retriever (baseline) | " + " | ".join(f"{ref[m]:.3f}" for m in shown) + " |")
        flat.append({"source": source, "ranker": "Retriever (baseline)", "order": "-", "n": base["n_expected"],
                     **{m: f"{ref[m]:.4f}" for m in metrics}, **{f"delta_{m}": "0" for m in metrics}})
        for order in ORDERS:
            for r in (x for x in mine if x["order"] == order):
                flag = "" if r["n"] == r["n_expected"] else f" (INCOMPLETE {r['n']}/{r['n_expected']})"
                md.append(f"| {source} | {r['model']} ({order}){flag} | "
                          + " | ".join(f"{float(r[f'model_{m}']):.3f}" for m in shown) + " |")
                flat.append({"source": source, "ranker": r["model"], "order": order, "n": r["n"],
                             **{m: f"{float(r[f'model_{m}']):.4f}" for m in metrics},
                             **{f"delta_{m}": f"{float(r[f'model_{m}']) - ref[m]:+.4f}" for m in metrics}})
    (out / f"ranking_comparison{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf8")
    with open(out / f"ranking_comparison{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    print(f"wrote {out/f'ranking_comparison{suffix}.md'} and {out/f'ranking_comparison{suffix}.csv'}")


def cmd_compare(args):
    write_comparison(Path(args.run_dir).resolve(), "_smoke" if args.smoke else "")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--src", required=True); b.add_argument("--run-dir", required=True)
    b.add_argument("--descriptions", help="icd_snomed_index.jsonl: give every option its code's long description")
    b.add_argument("--top", type=int, metavar="K", help="give the model only the retriever's first K codes of each list "
                   "(default: the whole list)")
    r = sub.add_parser("report"); r.add_argument("--run-dir", required=True)
    r.add_argument("--smoke", action="store_true", help="report the smoke-test predictions (written as *_smoke.csv/.md)")
    c = sub.add_parser("compare", help="detailed model-vs-retriever table from an existing ranking_results.csv (no rerun)")
    c.add_argument("--run-dir", required=True); c.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    {"build": cmd_build, "report": cmd_report, "compare": cmd_compare}[args.cmd](args)
