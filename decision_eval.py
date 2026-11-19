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
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core  # noqa: E402
from run_decision_model import cell_metrics, load_requests, n_tied_top, model_view, order_for, parse_sources, rank_options, retriever_rankings, select_requests  # noqa: E402

LADDER = (1, 3, 5, 10, 20, 50, 100, 200)   # hit@k cutoffs, taken from the ranking over ALL options of a request
MAX_OPTIONS = 255            # Elephant (schema MAX_CHOICE_OPTIONS) and Drex (255 answer labels); Clef has no fixed limit
ORDERS = ("original", "shuf1")
SOURCES = ("csv_dense", "snomed_fsn", "snomed_syn", "rrf_csv+fsn", "rrf_csv+syn", "rrf_fsn+syn", "rrf_csv+fsn+syn")
NAMES = {m: info["label"] for m, info in lh_core.MODELS_INFO.items()}
QUESTION = "Which ICD-10-CA code best matches the diagnosis described above?"


def cutoffs(top, longest):
    """hit@k cutoffs: the ladder up to K for a --top K build (K itself included: --top 10 -> 1,3,5,10; --top 20 ->
    1,3,5,10,20; --top 15 -> 1,3,5,10,15); for whole lists, the ladder up to the longest list."""
    k_max = top if top else longest
    ks = [k for k in LADDER if k <= k_max]
    return ks + [top] if top and top not in ks else ks


def parse_ks(text):
    """--ks '1,3,5,10' or '1 3 5 10' -> sorted unique whole numbers of 1 or more."""
    try:
        ks = sorted({int(x) for x in text.replace(",", " ").replace("[", " ").replace("]", " ").split()})
    except ValueError:
        raise argparse.ArgumentTypeError(f"--ks takes whole numbers, e.g. '1,3,5,10' (got {text!r})")
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError("--ks needs at least one number, all 1 or more")
    return ks


def merged_context(entity, query):
    entity, query = entity.strip(), query.strip()
    return entity if entity.lower() == query.lower() else f"{entity} (expanded: {query})"


def cap_rankings(data):
    """Keep the first MAX_OPTIONS codes of every ranking (the most the models take; a retriever may return more, e.g.
    500). Done right after loading, so validation, descriptions, retriever ranks and requests all see the same codes:
    a reference code ranked below MAX_OPTIONS is not an option and counts as a miss. Returns (lists cut, longest list)."""
    cut, longest = 0, 0
    for d in data:
        if not isinstance(d, dict) or not isinstance(d.get("rankings"), dict):
            continue   # validate() reports it
        for s, r in d["rankings"].items():
            if isinstance(r, list):
                longest = max(longest, len(r))
                if len(r) > MAX_OPTIONS:
                    d["rankings"][s] = r[:MAX_OPTIONS]
                    cut += 1
    return cut, longest


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
            elif len(ranking) > MAX_OPTIONS:   # cap_rankings() runs first; this guards a caller that skips it
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
    if args.top is not None and args.top > MAX_OPTIONS:
        sys.exit(f"STOPPED: --top {args.top} is above {MAX_OPTIONS}, the most options the models take; nothing was written")
    n_cut, longest = cap_rankings(data)
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
    if n_cut:
        held += (f" ({n_cut} lists held more than {MAX_OPTIONS} codes, up to {longest}: only their first {MAX_OPTIONS} "
                 "are read, the most the models take; a reference code ranked lower counts as a miss)")
    KS = cutoffs(top, max(max(L) for L in lengths.values()))
    print(f"{len(data)} entities. Every request holds {held}. Retriever: share of entities with the reference code "
          f"within its top k" + (f" (k above {top} cannot exceed @{top}):" if top else ":"))
    print(f"  {'source':<16} {'requests':>8} {'codes min/median/max':>20}  " + "  ".join(f"@{k:<4}" for k in KS)
          + "  in list (n)")
    for source in SOURCES:
        L, R = lengths[source], retr_rank[source]
        n_in = sum(r is not None for r in R)
        print(f"  {source:<16} {len(R) * len(ORDERS):>8} {min(L):>6}/{statistics.median(L):>5g}/{max(L):<6}  "
              + "  ".join(f"{sum(r is not None and r <= k for r in R) / len(R):.3f}" for k in KS)
              + f"  {n_in / len(R):.3f} ({n_in} of {len(R)} entities)")
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
        "descriptions_sha256": sha(args.descriptions) if desc else None, "top": top, "requests": len(rows),
        "max_options_read": MAX_OPTIONS, "lists_cut_to_max": n_cut, "longest_list_in_src": longest})
    print(f"options: {'code + ' + option_text if desc else 'bare codes'}; {'whole lists' if top is None else f'top {top} per list'}")
    print(f"wrote {len(rows)} requests for {len(data)} entities ({sum(r['n_options'] for r in rows)} options in all) -> {out/'requests.jsonl'}")


def model_rank(r, p):
    """1-based rank of the reference code in the model's ordering of all options (None if it is not an option).
    Ties go to the earlier position, the same rule that picks the prediction."""
    if not r["answerable"]:
        return None
    return rank_options(list(r["record"]["questions"][lh_core.QUESTION_ID]["criteria"]), p["probabilities"])[r["gt_letter"]]


def f3(x):
    """3 decimals, or N/A for a value that does not exist (nan: e.g. hit@k above the options a model can read)."""
    return "N/A" if isinstance(x, float) and math.isnan(x) else f"{float(x):.3f}"


def f4(x, sign=False):
    """4 decimals for ranking_comparison.csv (signed for the differences), N/A for nan."""
    return "N/A" if math.isnan(x) else (f"{x:+.4f}" if sign else f"{x:.4f}")


def num(text):
    """A number read back from ranking_results.csv ('N/A' / 'nan' / '' -> nan)."""
    return float("nan") if text in ("N/A", "nan", "") else float(text)


def hit_metrics(items, KS, limit=None):
    """hit@k for the model and for the retriever on the same requests (a request without the reference code is a miss), MRR.
    limit: the most options the model read (it was cut to that); its hit@k above the limit is N/A (nan)."""
    n = len(items)
    nan = float("nan")
    ranks = [model_rank(r, p) for r, p in items]
    out = {"n": n}
    for k in KS:
        out[f"model_hit@{k}"] = nan if (limit and k > limit) else sum(x is not None and x <= k for x in ranks) / n if n else nan
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
    chosen = args.sources                                    # None = all sources
    used = [s for s in SOURCES if not chosen or s in chosen]
    orders = [o for o in ORDERS if not args.orders or o in args.orders]   # --orders: None = both
    if args.orders and sorted(set(args.orders) - set(ORDERS)):
        sys.exit(f"STOPPED: unknown order(s) {sorted(set(args.orders) - set(ORDERS))}; choose from {' '.join(ORDERS)}")
    if chosen:
        bad = sorted(set(chosen) - set(SOURCES))
        if bad:
            sys.exit(f"STOPPED: unknown source(s) {bad}; choose from {' '.join(SOURCES)}")
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
            selected[model] = {r["request_id"] for r in select_requests(list(reqs.values()), json.loads(meta.read_text()).get("stride", 1), chosen, args.orders)}
    if not preds:
        sys.exit(f"no {'smoke-test ' if args.smoke else ''}predictions found in {out}"
                 + ("" if args.smoke else " (for the smoke test, add --smoke)"))
    # each model reads at most its own number of options (lh_core max_options): a longer list is cut to the retriever's
    # first `limit` codes, the question the worker gave it (rebuilt here and checked against the worker's record)
    longest_all = max(r["n_options"] for r in reqs.values())
    limits = {m: lh_core.MODELS_INFO[m]["max_options"] for m in preds}
    capped = {m: limits[m] for m in preds if limits[m] < longest_all}
    rankings = retriever_rankings(reqs.values()) if capped else {}
    views = {m: {i: model_view(reqs[i], capped[m], rankings) for i in selected[m]} for m in capped}
    for m in capped:
        for i, p in preds[m].items():
            if i in views[m] and p.get("shown_codes", views[m][i]["codes"]) != views[m][i]["codes"]:
                sys.exit(f"STOPPED: {i}: the options {m} was given differ from the ones rebuilt now; report it as a bug")
    R = lambda m, i: views[m][i] if m in views else reqs[i]   # noqa: E731  the request as model m read it
    tops = {r.get("top") for r in reqs.values()}
    KS = cutoffs(max(tops) if None not in tops else None, max(r["n_options"] for r in reqs.values()))
    if args.ks:   # --ks: the user's own cutoffs, up to the longest list (a larger k would repeat the last column)
        longest = max(r["n_options"] for r in reqs.values())
        dropped = [k for k in args.ks if k > longest]
        if dropped:
            print(f"WARNING: --ks {dropped} dropped: the lists hold at most {longest} options, so hit@k there equals hit@{longest}")
        KS = [k for k in args.ks if k <= longest]
        if not KS:
            sys.exit(f"STOPPED: every --ks value is above {longest}, the longest list")
    hit_cols = [f"model_hit@{k}" for k in KS] + ["model_mrr"] + [f"retriever_hit@{k}" for k in KS] + ["retriever_mrr"]
    rows = []
    for source in used:
        for order in orders:
            for model, done in preds.items():
                mine = [i for i, r in reqs.items() if r["source"] == source and r["order"] == order and i in selected[model]]
                items = [(R(model, i), done[i]) for i in mine if i in done]
                h, c = hit_metrics(items, KS, capped.get(model)), cell_metrics(items)
                rows.append({"source": source, "order": order, "model": NAMES[model], "n": h["n"], "n_expected": len(mine),
                             "model_max_options": limits[model],
                             "median_options": statistics.median(R(model, i)["n_options"] for i in mine) if mine else float("nan"),
                             "in_list": h["in_list"], **{k: h[k] for k in hit_cols},
                             "acc_when_in_list": c["acc"], "brier_when_in_list": c["brier"], "ece_when_in_list": c["ece"],
                             "tied_top": h["tied_top"]})
    summary = []   # pooled over sources: a smoke run has only a few requests per source
    for model, done in preds.items():
        for order in orders:
            items = [(R(model, i), done[i]) for i in sorted(selected[model]) if i in done and reqs[i]["order"] == order]
            summary.append((NAMES[model], order, hit_metrics(items, KS, capped.get(model)), sum(reqs[i]["order"] == order for i in selected[model])))
    head = f"{'model':<17} {'order':<9} {'n ok':>11} " + " ".join(f"{'hit@' + str(k):>6}" for k in KS) + f" {'MRR':>6}  | retriever " \
           + " ".join(f"{'@' + str(k):>5}" for k in KS) + f" {'MRR':>6} | tied"
    if "CLEF" in preds:
        pairs, differ = clef_order_check(reqs, preds["CLEF"])
        clef_line = (f"Clef order check: {pairs} original/shuf1 pairs compared, {differ} with different probabilities"
                     + (" (expected 0: Clef sorts options by code; investigate)" if differ else " (as expected)")
                     + ("" if pairs else " (no pairs: the smoke sample held no matching requests)"))
        print(clef_line)
    counts = []
    for source in SOURCES:
        mine = [r for r in reqs.values() if r["source"] == source]
        ents = {r["entity_id"] for r in mine}
        ents_in = {r["entity_id"] for r in mine if r["answerable"]}
        counts.append({"source": source, "entities": len(ents), "requests": len(mine),
                       "per_order": {o: sum(r["order"] == o for r in mine) for o in ORDERS},
                       "options_median": statistics.median(r["n_options"] for r in mine) if mine else 0,
                       "in_list": len(ents_in), "not_in_list": len(ents) - len(ents_in),
                       "results": {m: (sum(i in done for i in selected[m] if reqs[i]["source"] == source),
                                       sum(reqs[i]["source"] == source for i in selected[m])) if source in used else None
                                   for m, done in preds.items()}})
    cell = lambda x: "-" if x is None else f"{x[0]}/{x[1]}"   # noqa: E731  '-' = source left out by --sources
    if args.orders:
        print(f"Option orders in this report (--orders): {' '.join(orders)}")
    if chosen:
        print(f"Sources in this report (--sources): {' '.join(used)}; the counts table lists all of them, '-' = left out")
    print("Requests per retrieval source (one request = one entity x one option order):")
    print(f"  {'source':<16} {'entities':>8} {'requests':>8} {'options':>7} {'ref in list':>11} {'not in list':>11}  "
          + "  ".join(f"{NAMES[m]}" for m in preds))
    for c in counts:
        print(f"  {c['source']:<16} {c['entities']:>8} {c['requests']:>8} {c['options_median']:>7g} {c['in_list']:>11} "
              f"{c['not_in_list']:>11}  " + "  ".join(cell(c['results'][m]) for m in preds))
    print(f"  {'total':<16} {len({r['entity_id'] for r in reqs.values()}):>8} {len(reqs):>8}")
    print(head)
    for name, order, h, n_exp in summary:
        print(f"{name:<17} {order:<9} {h['n']:>5}/{n_exp:<5} " + " ".join(f"{f3(h[f'model_hit@{k}']):>6}" for k in KS)
              + f" {f3(h['model_mrr']):>6}  |           " + " ".join(f"{f3(h[f'retriever_hit@{k}']):>5}" for k in KS)
              + f" {f3(h['retriever_mrr']):>6} | {h['tied_top']}")
    with open(out / f"ranking_results{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: ("N/A" if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()} for row in rows])
    if capped:
        print("Read fewer options (cut to the retriever's first N codes; hit@k above N is N/A): "
              + ", ".join(f"{NAMES[m]} {n}" for m, n in capped.items()))
    cut = json.loads((out / "build_meta.json").read_text()) if (out / "build_meta.json").exists() else {}
    text = "codes with descriptions" if any(r.get("option_text", "code") != "code" for r in reqs.values()) else "bare codes"
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
          *([f"**Lists cut to {cut['max_options_read']} codes:** {cut['lists_cut_to_max']} source lists held more (up to "
             f"{cut['longest_list_in_src']}); only their first {cut['max_options_read']} codes are options, so a reference "
             "code ranked lower is a miss for the models and the retriever alike.", ""] if cut.get("lists_cut_to_max") else []),
          *([f"**{NAMES[m]} reads at most {n} options.** From each longer list it gets the retriever's first {n} codes "
             f"(the same question a --top {n} build gives), so a reference code ranked below {n} is a miss for it, its hit@k "
             f"above {n} is N/A, and its MRR and Acc are over those {n} codes." for m, n in capped.items()]),
          *([""] if capped else []),
          *([f"**Option order: only {', '.join(orders)}** (chosen with --orders); the other order is not in the tables.", ""]
            if args.orders else []),
          *([f"**Sources: only {', '.join(used)}** (chosen with --sources). The summary pools these sources only; the "
             "counts table lists every source, with '-' for those left out.", ""] if chosen else []),
          "## Requests per retrieval source", "",
          "One request = one entity x one option order. 'Ref in list': entities whose reference code is among the options "
          "(the others are misses for every model). Model columns: requests with a result / requests selected.", "",
          "| Source | Entities | Requests | Per order | Options (median) | Ref in list | Not in list | "
          + " | ".join(NAMES[m] for m in preds) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len(preds),
          *[f"| {c['source']} | {c['entities']} | {c['requests']} | " + " / ".join(f"{o} {c['per_order'][o]}" for o in ORDERS)
            + f" | {c['options_median']:g} | {c['in_list']} | {c['not_in_list']} | "
            + " | ".join(cell(c['results'][m]) for m in preds) + " |" for c in counts],
          f"| **total** | {len({r['entity_id'] for r in reqs.values()})} | {len(reqs)} | | | | | " + " | ".join("" for _ in preds) + " |", "",
          "## Summary (" + ("all sources pooled" if not chosen else "the chosen sources pooled") + ")", "",
          "| Model | Order | n ok | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR | " + " | ".join(f"retriever @{k}" for k in KS) + " | retriever MRR | Tied top |"]
    md.append(separator(md[-1]))
    md += [f"| {name} | {order} | {h['n']}/{n_exp} | " + " | ".join(f3(h[f'model_hit@{k}']) for k in KS) + f" | {f3(h['model_mrr'])} | "
           + " | ".join(f3(h[f'retriever_hit@{k}']) for k in KS) + f" | {f3(h['retriever_mrr'])} | {h['tied_top']} |"
           for name, order, h, n_exp in summary] + [""]
    for order in orders:
        md += [f"## Option order: {order}", "",
               "| Source | Model | n | Options (median) | In list | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR | "
               + " | ".join(f"retr @{k}" for k in KS) + " | retr MRR | Acc \\| in list | Brier | ECE |"]
        md.append(separator(md[-1]))
        for r in (x for x in rows if x["order"] == order):
            flag = "" if r["n"] == r["n_expected"] else f" INCOMPLETE {r['n']}/{r['n_expected']}"
            md.append(f"| {r['source']} | {r['model']}{flag} | {r['n']} | {r['median_options']:g} | {f3(r['in_list'])} | "
                      + " | ".join(f3(r[f'model_hit@{k}']) for k in KS) + f" | {f3(r['model_mrr'])} | "
                      + " | ".join(f3(r[f'retriever_hit@{k}']) for k in KS) + f" | {f3(r['retriever_mrr'])} | "
                      f"{f3(r['acc_when_in_list'])} | {f3(r['brier_when_in_list'])} | {f3(r['ece_when_in_list'])} |")
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
    KS = sorted(int(c.split("@")[1]) for c in rows[0] if c.startswith("model_hit@"))   # the cutoffs the report used
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
        ref = {m: num(base[f"retriever_{m}"]) for m in metrics}
        md.append(f"| {source} | Retriever (baseline) | " + " | ".join(f3(ref[m]) for m in shown) + " |")
        flat.append({"source": source, "ranker": "Retriever (baseline)", "order": "-", "n": base["n_expected"],
                     **{m: f4(ref[m]) for m in metrics}, **{f"delta_{m}": "0" for m in metrics}})
        for order in ORDERS:
            for r in (x for x in mine if x["order"] == order):
                flag = "" if r["n"] == r["n_expected"] else f" (INCOMPLETE {r['n']}/{r['n_expected']})"
                md.append(f"| {source} | {r['model']} ({order}){flag} | "
                          + " | ".join(f3(num(r[f'model_{m}'])) for m in shown) + " |")
                flat.append({"source": source, "ranker": r["model"], "order": order, "n": r["n"],
                             **{m: f4(num(r[f'model_{m}'])) for m in metrics},
                             **{f"delta_{m}": f4(num(r[f'model_{m}']) - ref[m], sign=True) for m in metrics}})
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
    r.add_argument("--sources", type=parse_sources, default=None, metavar="'A B'",
                   help="report only these retrieval sources (space- or comma-separated; default: all)")
    r.add_argument("--orders", type=parse_sources, default=None, metavar="'original'",
                   help="report only these option orders (original, shuf1; default: both)")
    r.add_argument("--ks", type=parse_ks, default=None, metavar="'1,3,5,10'",
                   help="hit@k cutoffs to report (comma- or space-separated; default: 1,3,5,10,20,50,... up to the list length)")
    c = sub.add_parser("compare", help="detailed model-vs-retriever table from an existing ranking_results.csv (no rerun)")
    c.add_argument("--run-dir", required=True); c.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    {"build": cmd_build, "report": cmd_report, "compare": cmd_compare}[args.cmd](args)
