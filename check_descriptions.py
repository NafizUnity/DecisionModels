"""Checks, before any model run, whether ICD-10-CA code descriptions from an index file fit the decision models.

    $PY check_descriptions.py --data "$DATA" --index /path/icd_snomed_index.jsonl \
        [--weights-dir $MODELS] [--bare-run $RUN] [--out FILE.json] [--show-examples]

Reads only; writes nothing except --out (counts only). No GPU. Never prints or saves entity or query text:
text is used inside this process (and the model tokenizers it starts) only to count tokens and compare strings.
Index descriptions are public reference text; they are printed only with --show-examples, for codes taken from
the index itself (not from the data).

  A. the index on its own: parse, duplicate codes, empty descriptions, code format, odd characters, lengths
  B. data codes vs index: coverage exactly, after normalising (dots, case, spaces), and via the hierarchy
  C. options a model cannot tell apart: two codes of one list with the same description
  D. how each model shows an option with a description (--show-examples)
  E. tokens per request with each model's own prompt code and tokenizer, bare vs with descriptions, vs its limit
     (--weights-dir; starts each model's conda env on CPU). Without it: a characters / 4 estimate.
  F. the data matches the bare-code run (--bare-run): same entities, rankings, ground truth and option order;
     plus a rough run-time estimate from that run's latencies
  G. entities whose text equals their reference code's description (count only)

Exit code 1 if any check FAILs. Planned use of the descriptions in the build: criteria = {code: description}, the
description being long_description (short_description when long is empty), whitespace collapsed to single spaces.
Every model adds the code itself (Elephant "A. E119: ...", Drex "A: E119: ...", Clef {"option_id": "E119", ...}).
"""
import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTION_ID = "icd10ca_code"   # lh_core.QUESTION_ID; repeated so the model-env child needs no project imports

# Token limits the runs apply (sources: run_decision_model.cmd_worker max_tokens=16000, passed to Drex predict and
# checked by the Clef adapter; qev/engine.py DecisionEngine max_branch_tokens=32768, max_request_tokens=65536).
LIMITS = {"ELEPHANT": 32768, "DREX": 16000, "CLEF": 16000}
ENVS = {"ELEPHANT": "realdata-elephant", "DREX": "realdata-drex", "CLEF": "realdata-clef"}
EXAMPLE_STATE = "type 2 diabetes mellitus (example text, not from the data)"

status_lines, summary = [], {}


def report(level, check, message, **counts):
    status_lines.append((level, check))
    print(f"[{level:<4}] {check}: {message}")
    summary[check] = {"status": level, **counts}


def norm(code):
    return code.replace(".", "").replace(" ", "").upper()


def clean(text):
    return " ".join(text.split()) if isinstance(text, str) else ""


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.2f}%)" if b else f"{a}/0"


def dist(values):
    if not values:
        return "n/a"
    s = sorted(values)
    return f"min {s[0]}, median {statistics.median(s):g}, p95 {s[int(0.95 * (len(s) - 1))]}, max {s[-1]}"


# --- A: the index -----------------------------------------------------------------------------------------------------
def load_index(path):
    main, hier, parse_errors, no_code = {}, {}, 0, 0
    dup_same, dup_diff, empty_long, empty_both, odd_chars = 0, 0, 0, 0, 0
    long_eq_short, n = 0, 0
    with open(path, encoding="utf8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            code = rec.get("code") if isinstance(rec, dict) else None
            if not isinstance(code, str) or not code.strip():
                no_code += 1
                continue
            long_d, short_d = rec.get("long_description"), rec.get("short_description")
            raw = long_d if isinstance(long_d, str) and long_d.strip() else short_d
            if not (isinstance(long_d, str) and long_d.strip()):
                empty_long += 1
                if not (isinstance(short_d, str) and short_d.strip()):
                    empty_both += 1
            if isinstance(raw, str) and any(c in raw for c in "\n\r\t") or isinstance(raw, str) and any(ord(c) < 32 for c in raw):
                odd_chars += 1
            if clean(long_d) and clean(long_d) == clean(short_d):
                long_eq_short += 1
            desc = clean(raw) or None
            if code in main:
                if main[code] == desc:
                    dup_same += 1
                else:
                    dup_diff += 1
                continue
            main[code] = desc
            for level, node in (rec.get("hierarchy") or {}).items():   # parents (cat3, cat4, block, ...) as a fallback
                if isinstance(node, dict) and isinstance(node.get("code"), str) and node["code"].strip():
                    d = clean(node.get("long_description")) or clean(node.get("short_description"))
                    if d:
                        hier.setdefault(node["code"].strip(), (level, d))
    codes = list(main)
    report("FAIL" if parse_errors or not codes else "PASS", "A1 index parses",
           f"{n} lines, {len(codes)} distinct codes, {parse_errors} unparsable lines, {no_code} records without a code",
           lines=n, codes=len(codes), parse_errors=parse_errors, no_code=no_code)
    report("FAIL" if dup_diff else ("WARN" if dup_same else "PASS"), "A2 duplicate codes",
           f"{dup_diff} repeated with a DIFFERENT description, {dup_same} repeated with the same one",
           different=dup_diff, same=dup_same)
    report("WARN" if empty_long else "PASS", "A3 empty long_description",
           f"{empty_long} codes (fall back to short_description); {empty_both} have neither",
           empty_long=empty_long, empty_both=empty_both)
    fmt = {"with dot": sum("." in c for c in codes), "lowercase": sum(c != c.upper() for c in codes),
           "with space": sum(" " in c for c in codes), "length": f"{min(map(len, codes))}-{max(map(len, codes))}" if codes else "n/a"}
    report("INFO", "A4 index code format", ", ".join(f"{k} {v}" for k, v in fmt.items()), **fmt)
    lengths = [len(d) for d in main.values() if d]
    report("WARN" if odd_chars else "PASS", "A5 description text",
           f"{odd_chars} with newlines/tabs/control characters (collapsed to spaces); long == short in {long_eq_short}; "
           f"characters: {dist(lengths)}", odd_chars=odd_chars, long_eq_short=long_eq_short)
    return main, hier


# --- B, C, G: the data against the index -------------------------------------------------------------------------------
def check_data(data, main, hier):
    import decision_eval as de
    list_codes = {c for d in data for s in de.SOURCES for c in d["rankings"][s]}
    gt_codes = {d["ground_truth_codes"][0] for d in data}
    by_norm = collections.defaultdict(set)
    for c in main:
        by_norm[norm(c)].add(c)
    sample_codes = sorted(list_codes)[:5]
    report("INFO", "B0 data code format", f"{len(list_codes)} distinct codes in the lists, {len(gt_codes)} reference codes; "
           f"with dot {sum('.' in c for c in list_codes)}, lowercase {sum(c != c.upper() for c in list_codes)}; "
           f"examples {sample_codes}", list_codes=len(list_codes), gt_codes=len(gt_codes))

    def coverage(codes):
        exact = {c for c in codes if c in main and main[c]}
        normed = {c for c in codes - exact if len(by_norm.get(norm(c), ())) == 1 and main[next(iter(by_norm[norm(c)]))]}
        ambiguous = {c for c in codes - exact if len(by_norm.get(norm(c), ())) > 1}
        via_hier = {c for c in codes - exact - normed if c in hier}
        prefix = {c for c in codes - exact - normed - via_hier if any(k.startswith(norm(c)) for k in by_norm)}
        return exact, normed, ambiguous, via_hier, prefix

    for check, codes, name in (("B1", list_codes, "list codes"), ("B2", gt_codes, "reference codes")):
        exact, normed, ambiguous, via_hier, prefix = coverage(codes)
        missing = sorted(codes - exact - normed - via_hier)
        level = "PASS" if len(exact) == len(codes) else ("WARN" if len(exact) + len(normed) + len(via_hier) == len(codes) else "FAIL")
        report(level, f"{check} {name} with a description",
               f"exact {pct(len(exact), len(codes))}; only after normalising (dots/case/spaces) {len(normed)}; "
               f"ambiguous after normalising {len(ambiguous)}; only via a hierarchy parent {len(via_hier)}; "
               f"still missing {len(missing)}, of which {len(prefix)} are a prefix of an index code (e.g. E11 vs E119)"
               + (f"; first missing: {missing[:20]}" if missing else ""),
               total=len(codes), exact=len(exact), normalised=len(normed), ambiguous=len(ambiguous),
               via_hierarchy=len(via_hier), missing=len(missing), missing_prefix=len(prefix), first_missing=missing[:20])
        if check == "B1":
            exact_list = exact
    per_source, lists_with_gap = {}, {}
    for s in de.SOURCES:
        codes = [c for d in data for c in d["rankings"][s]]
        per_source[s] = sum(c in exact_list for c in codes) / len(codes)
        lists_with_gap[s] = sum(any(c not in exact_list for c in d["rankings"][s]) for d in data)
    report("PASS" if all(v == 1 for v in per_source.values()) else "WARN", "B3 coverage per source (exact)",
           "; ".join(f"{s} {per_source[s]:.4f} ({lists_with_gap[s]} lists with a code lacking one)" for s in de.SOURCES),
           per_source=per_source, lists_with_gap=lists_with_gap)

    desc = {c: main[c] for c in exact_list}
    lists_dup, pairs_dup, gt_involved = 0, 0, 0
    for d in data:
        gt = d["ground_truth_codes"][0]
        for s in de.SOURCES:
            seen = collections.Counter(desc[c].lower() for c in d["rankings"][s] if c in desc)
            dups = {t: k for t, k in seen.items() if k > 1}
            if dups:
                lists_dup += 1
                pairs_dup += sum(k * (k - 1) // 2 for k in dups.values())
                gt_involved += gt in desc and desc[gt].lower() in dups
    total_lists = len(data) * len(de.SOURCES)
    report("WARN" if lists_dup else "PASS", "C1 same description for two codes in one list",
           f"{pct(lists_dup, total_lists)} lists, {pairs_dup} code pairs; the reference code is one of them in {gt_involved} "
           "lists (the code shown with each option tells them apart)",
           lists=lists_dup, pairs=pairs_dup, gt_involved=gt_involved)

    eq_long = sum(clean(d["entity_text"]).lower() == (desc.get(d["ground_truth_codes"][0]) or "").lower() for d in data)
    eq_query = sum(clean(d.get("query_text") or "").lower() == (desc.get(d["ground_truth_codes"][0]) or "").lower() for d in data)
    report("INFO", "G1 text equals the reference description",
           f"entity text in {pct(eq_long, len(data))} entities; query text in {pct(eq_query, len(data))}",
           entity_text=eq_long, query_text=eq_query)
    return desc


def build_items(data):
    """The requests the build would make, in the same order and with the same option order as decision_eval build."""
    import decision_eval as de
    items = []
    for d in data:
        eid = str(d["entity_id"])
        state = de.merged_context(d["entity_text"], d.get("query_text") or "")
        for s in de.SOURCES:
            for order in de.ORDERS:
                items.append({"request_id": f"{eid}|{s}|{order}", "state": state, "gt": d["ground_truth_codes"][0],
                              "codes": de.order_for(list(d["rankings"][s]), order, f"{eid}|{s}", "all")})
    return items


# --- E: tokens per request (child side, runs inside a model's conda env) ------------------------------------------------
def child(model, weights):
    """Reads a header line {descs, question, example} then one {i, state, codes} per line; writes {i, bare, desc}."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    weights = Path(weights)
    if model == "ELEPHANT":
        from transformers import AutoTokenizer
        from qev.prompt import DEFAULT_STYLE, build_messages, render_question
        from qev.schema import SystemOneRequest
        ckpt = weights / "elephant-1"
        tok = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True, trust_remote_code=False)
        kw = {"enable_thinking": False} if "qwen" in json.loads((ckpt / "config.json").read_text()).get("model_type", "") else {}

        def render(state, question):
            req = SystemOneRequest(state=state, questions={QUESTION_ID: question})
            r = render_question(QUESTION_ID, req.questions[QUESTION_ID], style=DEFAULT_STYLE)
            text = tok.apply_chat_template(build_messages(req.state, r.suffix, DEFAULT_STYLE), tokenize=False,
                                           add_generation_prompt=True, **kw)
            return len(tok.encode(text, add_special_tokens=False)), text
    elif model == "DREX":
        sys.path.insert(0, str(weights / "drex-decision-models"))
        from transformers import AutoProcessor
        from drex_max.model import _prompt
        ckpt = weights / "drex-max-v1"
        proc = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=False)
        codes = json.loads((ckpt / "decision_config.json").read_text())["codes"]
        checked = []

        def render(state, question):
            text = proc.apply_chat_template(_prompt({"state": state, "question": question}, codes), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            n = len(proc.tokenizer(text)["input_ids"])
            if not checked:   # the model counts with the processor; confirm once that the tokenizer agrees
                assert n == proc(text=[text], padding=True, return_tensors="pt")["input_ids"].shape[1], "Drex count mismatch"
                checked.append(1)
            return n, text
    elif model == "CLEF":
        sys.path.insert(0, str(weights / "clef"))
        sys.dont_write_bytecode = True
        from transformers import AutoProcessor
        from joint_schema_model import encode_record
        proc = AutoProcessor.from_pretrained(str(weights / "clef"))

        def render(state, question):
            enc = encode_record(proc.tokenizer, {"state": state, "questions": {QUESTION_ID: question}},
                                max_length=10 ** 9, processor=proc)
            return len(enc.input_ids), proc.tokenizer.decode(enc.input_ids)
    else:
        sys.exit(f"unknown model {model}")
    header = json.loads(sys.stdin.readline())
    descs, q = header["descs"], header["question"]
    out = sys.stdout
    if header.get("example"):
        ex = header["example"]
        _, text = render(EXAMPLE_STATE, {"type": "choice", "instructions": q, "criteria": {c: ex[c] for c in ex}})
        out.write(json.dumps({"example": text}) + "\n")
    for line in sys.stdin:
        r = json.loads(line)
        row = {"i": r["i"]}
        try:
            row["bare"] = render(r["state"], {"type": "choice", "instructions": q, "criteria": {c: None for c in r["codes"]}})[0]
            row["desc"] = render(r["state"], {"type": "choice", "instructions": q,
                                              "criteria": {c: descs.get(c) for c in r["codes"]}})[0]
        except Exception as exc:  # noqa: BLE001 - type only: messages (e.g. pydantic's) can echo the input text
            row["error"] = type(exc).__name__
        out.write(json.dumps(row) + "\n")
    out.flush()


def env_python(model, overrides):
    if model in overrides:
        return overrides[model]
    import main as lhmain
    return lhmain.resolve_env_python(ENVS[model])


def check_tokens(items, desc, main, args):
    if not args.weights_dir:
        bare = [sum(len(c) + 4 for c in it["codes"]) + len(it["state"]) + 300 for it in items]
        withd = [b + sum(len(desc.get(c) or "") + 2 for c in it["codes"]) for b, it in zip(bare, items)]
        b, w = [x // 4 for x in bare], [x // 4 for x in withd]
        report("INFO", "E0 tokens (ESTIMATE: characters / 4; pass --weights-dir for exact counts)",
               f"bare {dist(b)}; with descriptions {dist(w)}; ratio of means {statistics.mean(w) / statistics.mean(b):.2f}")
        return {}
    overrides = dict(x.split("=", 1) for x in args.env_python)
    sel = items[::args.stride]
    used = {c for it in sel for c in it["codes"]}
    example_codes = [c for c in main if main[c]][:3]
    header = {"descs": {c: desc[c] for c in used if c in desc}, "question": __import__("decision_eval").QUESTION,
              "example": {c: main[c] for c in example_codes} if args.show_examples else None}
    payload = json.dumps(header, ensure_ascii=False) + "\n" + "".join(
        json.dumps({"i": i, "state": it["state"], "codes": it["codes"]}, ensure_ascii=False) + "\n" for i, it in enumerate(sel))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "MPLBACKEND")}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
    ratios = {}
    for model in args.models:
        python = env_python(model, overrides)
        if not python:
            report("FAIL", f"E {model}", f"conda env {ENVS[model]} not found (or pass --env-python {model}=PATH)")
            continue
        print(f"       counting tokens for {model} ({len(sel)} requests, CPU) ...", flush=True)
        proc = subprocess.run([python, "-u", str(Path(__file__).resolve()), "--count-tokens", model, "--weights-dir",
                               str(Path(args.weights_dir).resolve())], input=payload, capture_output=True, text=True,
                              env=env, cwd=str(HERE))
        rows = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
        if proc.returncode != 0 or not rows:
            tail = "\n".join(proc.stderr.strip().splitlines()[-8:])   # library errors; requests are never echoed there
            report("FAIL", f"E {model}", f"token counter exited with {proc.returncode}:\n{tail}")
            continue
        if args.show_examples and "example" in rows[0]:
            print(f"\n----- D: {model} input for a made-up request with 3 index codes -----\n{rows[0]['example']}\n" + "-" * 60)
        res = [r for r in rows if "i" in r]
        errors = collections.Counter(r["error"] for r in res if "error" in r)
        ok = [r for r in res if "error" not in r]
        bare, withd = [r["bare"] for r in ok], [r["desc"] for r in ok]
        over_b = sum(x > LIMITS[model] for x in bare)
        over_d = sum(x > LIMITS[model] for x in withd)
        ratios[model] = statistics.mean(withd) / statistics.mean(bare) if ok else None
        level = "FAIL" if errors or over_d or len(res) != len(sel) else "PASS"
        report(level, f"E {model} tokens (limit {LIMITS[model]})",
               f"{len(ok)}/{len(sel)} counted{'' if args.stride == 1 else f' (every {args.stride}th request)'}; "
               f"bare: {dist(bare)}; with descriptions: {dist(withd)}; ratio of means {ratios[model] or 0:.2f}; "
               f"over the limit: bare {over_b}, with descriptions {over_d}" + (f"; errors {dict(errors)}" if errors else ""),
               counted=len(ok), requests=len(sel), over_limit_bare=over_b, over_limit_desc=over_d,
               max_desc=max(withd) if withd else None, median_desc=statistics.median(withd) if withd else None,
               ratio=ratios[model], errors=dict(errors))
    return ratios


# --- F: same data as the bare-code run ------------------------------------------------------------------------------------
def check_bare_run(items, run, ratios):
    run = Path(run)
    want = {it["request_id"]: it for it in items}
    have, mismatch = 0, collections.Counter()
    with open(run / "requests.jsonl", encoding="utf8") as fh:
        for line in fh:
            r = json.loads(line)
            have += 1
            it = want.get(r["request_id"])
            if it is None:
                mismatch["request not in data"] += 1
            elif r.get("codes") != it["codes"]:
                mismatch["different codes or order"] += 1
            elif r.get("gt_code") != it["gt"]:
                mismatch["different reference code"] += 1
            elif r["record"]["state"] != it["state"]:
                mismatch["different state"] += 1
    level = "PASS" if have == len(want) and not mismatch else "FAIL"
    report(level, "F1 data matches the bare-code run", f"{have} requests in {run.name}, {len(want)} from the data; "
           f"mismatches {dict(mismatch) or 0}", requests_run=have, requests_data=len(want), mismatches=dict(mismatch))
    for model in ("ELEPHANT", "DREX", "CLEF"):
        p = run / f"predictions_{model}.jsonl"
        if not p.exists():
            continue
        lat = [json.loads(l).get("latency_s") for l in open(p, encoding="utf8") if l.strip()]
        lat = [x for x in lat if isinstance(x, (int, float))]
        if not lat:
            continue
        hours = sum(lat) / 3600
        r = ratios.get(model)
        report("INFO", f"F2 {model} run time", f"bare run {hours:.1f} h ({statistics.mean(lat):.2f} s/request)"
               + (f"; with descriptions roughly {hours * r:.1f} h if time grows with tokens (x{r:.2f}); the smoke test "
                  "gives the measured figure" if r else ""), bare_hours=hours, ratio=r)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="the entity file given to decision_eval.py build --src")
    p.add_argument("--index", help="icd_snomed_index.jsonl")
    p.add_argument("--weights-dir", help="$MODELS: exact token counts with each model's own prompt code (CPU)")
    p.add_argument("--bare-run", help="the bare-code run folder ($RUN): data consistency and a run-time estimate")
    p.add_argument("--models", nargs="+", default=["ELEPHANT", "DREX", "CLEF"], type=str.upper,
                   choices=["ELEPHANT", "DREX", "CLEF"])
    p.add_argument("--stride", type=int, default=1, help="count tokens for every Nth request only (faster)")
    p.add_argument("--show-examples", action="store_true", help="print each model's input for a made-up request")
    p.add_argument("--env-python", action="append", default=[], metavar="MODEL=PATH", help="override a model env's python")
    p.add_argument("--out", help="write the counts (no text) as JSON")
    p.add_argument("--count-tokens", help=argparse.SUPPRESS)   # child mode, inside a model env
    args = p.parse_args()
    if args.count_tokens:
        return child(args.count_tokens, args.weights_dir)
    if not (args.data and args.index):
        p.error("--data and --index are required")
    sys.path.insert(0, str(HERE))
    import decision_eval as de
    main_idx, hier = load_index(args.index)
    data = json.load(open(args.data, encoding="utf8"))
    problems = de.validate(data)
    report("FAIL" if problems else "PASS", "B data file valid", f"{len(data)} entities, {len(problems)} problem(s)"
           + (f"; first: {problems[0]}" if problems else ""), entities=len(data), problems=len(problems))
    if problems:
        sys.exit(1)
    desc = check_data(data, main_idx, hier)
    items = build_items(data)
    ratios = check_tokens(items, desc, main_idx, args)
    if args.bare_run:
        check_bare_run(items, args.bare_run, ratios)
    n_fail = sum(level == "FAIL" for level, _ in status_lines)
    n_warn = sum(level == "WARN" for level, _ in status_lines)
    print(f"\n{n_fail} FAIL, {n_warn} WARN, {len(status_lines) - n_fail - n_warn} PASS/INFO")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out} (counts only)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
