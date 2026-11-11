"""Evaluate the three decision models as ICD-10-CA code pickers on RAG_OUTPUTS.json (synthetic data).

Commands (run from GeminiOps/ClinicalRag with the base interpreter; every run gets its own --run-dir):
  python run_decision_model.py build --run-dir results/NAME [--src data/FILE.json]  # requests + dataset diagnostics (CPU)
  python run_decision_model.py run ELEPHANT --gpu 1 --run-dir results/NAME          # one model in its own env; resumable
  python run_decision_model.py run ELEPHANT --gpu 1 --run-dir results/NAME --stride 40   # smoke test: every 40th request
  python run_decision_model.py report --run-dir results/NAME                        # tables -> results/NAME/

Design: each entity becomes one 'choice' request (state = the note, question = which code fits the entity text, options =
"CODE: description" for the top-k candidates). Candidate order is varied because the source ranking puts the reference code
first whenever it is present: `original` keeps that order, `reversed` puts it last, `shuf1`/`shuf2` are fixed random orders.
Only the shuffled/reversed orders can show whether a model reads the text or just follows position.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core  # noqa: E402

DEFAULT_SRC = HERE / "data" / "RAG_OUTPUTS.json"
KS = ("top_5", "top_10", "top_20")
ORDERS = ("original", "reversed", "shuf1", "shuf2")
LETTERS = "ABCDEFGHIJKLMNOPQRST"


# --- build ------------------------------------------------------------------------------------------------------------
def order_for(codes, order, entity_id, k):
    if order == "original":
        return list(codes)
    if order == "reversed":
        return list(reversed(codes))
    seed = int(hashlib.sha256(f"{entity_id}|{k}|{order}".encode()).hexdigest()[:12], 16)
    shuffled = list(codes)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def build_requests(src, with_notes=True):
    data = json.load(open(src, encoding="utf8"))
    desc = data["code_descriptions"]
    requests = []
    for note in data["patient_notes"]:
        for ent in note["entities"]:
            gt = ent["ground_truth_code"]
            for k in KS:
                cands = ent["rag_outputs"][k]["candidate_codes"]
                for order in ORDERS:
                    codes = order_for(cands, order, ent["entity_id"], k)
                    letters = LETTERS[:len(codes)]
                    criteria = {l: f"{c}: {desc[c]}" for l, c in zip(letters, codes)}
                    if with_notes:
                        state = note["note_text"]
                        question = (f'Which ICD-10-CA code best matches the diagnosis "{ent["entity_text"]}" '
                                    f"as documented in this note?")
                    else:   # entity text only; the model's context field is not left empty
                        state = ent["entity_text"]
                        question = "Which ICD-10-CA code best matches the diagnosis described above?"
                    requests.append({
                        "request_id": f"{ent['entity_id']}|{k}|{order}", "entity_id": ent["entity_id"], "note_id": note["note_id"],
                        "k": k, "order": order, "gt_code": gt, "answerable": gt in codes,
                        "gt_letter": letters[codes.index(gt)] if gt in codes else None,
                        "codes": codes, "source_pick": ent["rag_outputs"][k].get("llm_icd_code"),
                        "record": {"state": state,
                                   "questions": {lh_core.QUESTION_ID: {"type": "choice", "instructions": question, "criteria": criteria}}},
                    })
    return requests, data


def diagnostics(data):
    desc = data["code_descriptions"]
    ents = [e for n in data["patient_notes"] for e in n["entities"]]
    out = {"n_notes": len(data["patient_notes"]), "n_entities": len(ents), "n_codes_in_pool": len(desc),
           "n_distinct_reference_codes": len({e["ground_truth_code"] for e in ents})}
    for k in KS:
        ranks, fam, chap, dist = [], 0, 0, 0
        for e in ents:
            c = e["rag_outputs"][k]["candidate_codes"]
            gt = e["ground_truth_code"]
            if gt in c:
                ranks.append(c.index(gt) + 1)
            for x in c:
                if x != gt:
                    dist += 1
                    fam += x[:3] == gt[:3]
                    chap += x[0] == gt[0]
        out[k] = {"recall": len(ranks) / len(ents), "reference_at_rank1_when_present": sum(r == 1 for r in ranks) / max(1, len(ranks)),
                  "rank_distribution": dict(sorted(collections.Counter(ranks).items())),
                  "distractors_same_3char_family": fam / dist, "distractors_same_first_letter": chap / dist,
                  "first_candidate_baseline_accuracy": sum(e["rag_outputs"][k]["candidate_codes"][0] == e["ground_truth_code"]
                                                           for e in ents) / len(ents)}
    out["entity_text_equals_reference_description"] = sum(
        e["entity_text"].strip().lower() == desc[e["ground_truth_code"]].strip().lower() for e in ents) / len(ents)
    return out


def cmd_build(args):
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    requests, data = build_requests(args.src, with_notes=not args.no_notes)
    with open(out / "requests.jsonl", "w", encoding="utf8") as fh:
        for r in requests:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    diag = diagnostics(data)
    (out / "dataset_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(f"wrote {len(requests)} requests -> {out/'requests.jsonl'}")
    print(json.dumps(diag, indent=2))


# --- run --------------------------------------------------------------------------------------------------------------
def load_requests(run_dir, stride):
    path = Path(run_dir) / "requests.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf8")]
    if stride == 1:
        return rows
    # about 1 in `stride`, chosen by a hash of the request id: every order, source and k gets its share (taking every
    # stride-th line would follow the file's fixed pattern, e.g. only 'original' requests when the stride is even)
    return [r for r in rows if int(hashlib.sha256(r["request_id"].encode()).hexdigest()[:8], 16) % stride == 0]


EXPORT_COLUMNS = ("request_id", "entity_id", "source", "k", "order", "model", "position", "option_letter", "option_code",
                  "probability", "prob_rank", "is_prediction", "is_gt", "answerable", "n_tied_top")


def export_options(run_dir, model, smoke):
    """Write two timestamped files to run_dir/option_probs/ (name <MODEL>[_smoke]_<run>_<UTC time>), plus a .json sidecar:
      .jsonl  one self-contained record per selected request: the model input (state, question, options letter -> code),
              the ground truth, the prediction and every option's probability (sorted high to low). Requests without a
              result are included with their status. HOLDS ENTITY TEXT: keep it with the data, like requests.jsonl.
      .csv    one row per (request, option): ids, codes and probabilities only, no entity text.
    Reads requests.jsonl and the predictions file. Standard library only, so it runs in the model envs and the base python."""
    run_dir = Path(run_dir).resolve()
    pred_path = run_dir / f"predictions_{model}{'_smoke' if smoke else ''}.jsonl"
    meta_path = pred_path.with_name(pred_path.name + ".meta.json")
    if not pred_path.exists() or not meta_path.exists():
        sys.exit(f"STOPPED: {pred_path.name} (or its .meta.json) not found in {run_dir}")
    meta = json.loads(meta_path.read_text())
    requests_sha = hashlib.sha256((run_dir / "requests.jsonl").read_bytes()).hexdigest()
    if meta.get("requests_sha256") != requests_sha:
        sys.exit(f"STOPPED: {pred_path.name} was made for a different requests.jsonl; not exporting")
    requests = load_requests(run_dir, meta.get("stride", 1))
    latest, last_error = {}, {}
    for row in lh_core.read_jsonl_readonly(pred_path)[0]:
        if row.get("status") == "ok":
            latest[row["request_id"]] = row          # a later ok row for the same request wins
        else:
            last_error[row["request_id"]] = row
    revision = json.loads((HERE / "revisions.lock.json").read_text())["models"].get(model, {}).get("revision")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = run_dir / "option_probs" / f"{model}{'_smoke' if smoke else ''}_{run_dir.name}_{stamp}.csv"
    out_jsonl = out.with_suffix(".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp, tmp_jsonl = out.with_name(out.name + ".tmp"), out_jsonl.with_name(out_jsonl.name + ".tmp")
    n_rows = n_req = n_tied = 0
    with open(tmp, "w", newline="", encoding="utf8") as f, open(tmp_jsonl, "w", encoding="utf8") as fj:
        w = csv.writer(f)
        w.writerow(EXPORT_COLUMNS)
        for r in requests:
            question = next(iter(r["record"]["questions"].values()))   # the one question (older runs named it "longhealth")
            letters = list(question["criteria"])
            entity_id = r.get("entity_id", r["request_id"].split("|")[0])
            record = {"request_id": r["request_id"], "entity_id": entity_id, "source": r.get("source", ""), "k": r["k"],
                      "order": r["order"], "model": model, "model_revision": revision,
                      "input": {"state": r["record"]["state"], "instructions": question["instructions"],
                                "options": dict(question["criteria"])},
                      "ground_truth": {"code": r["gt_code"], "letter": r["gt_letter"], "answerable": bool(r["answerable"]),
                                       "retriever_rank1_correct": r.get("rank1_correct")}}
            p = latest.get(r["request_id"])
            if p is None:
                err = last_error.get(r["request_id"])
                record.update(status=err["status"] if err else "not_run",
                              error={k: err.get(k) for k in ("error_type", "failure_class")} if err else None)
                fj.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            probs = p["probabilities"]
            top = max(probs.values())
            tied = sum(v == top for v in probs.values())
            n_req += 1
            n_tied += tied > 1
            ranked = sorted(letters, key=lambda l: (-probs[l], letters.index(l)))
            rank = {l: i for i, l in enumerate(ranked, 1)}
            record.update(status="ok",
                          prediction={"letter": p["prediction_letter"], "code": p["prediction_code"],
                                      "correct": p["prediction_code"] == r["gt_code"]},
                          probabilities=[{"rank": rank[l], "letter": l, "code": question["criteria"][l], "probability": probs[l]}
                                         for l in ranked],
                          n_tied_top=tied, latency_s=p.get("latency_s"))
            fj.write(json.dumps(record, ensure_ascii=False) + "\n")
            for pos, (letter, code) in enumerate(zip(letters, r["codes"]), 1):
                w.writerow((r["request_id"], entity_id, r.get("source", ""), r["k"],
                            r["order"], model, pos, letter, code, repr(probs[letter]), rank[letter],
                            int(letter == p["prediction_letter"]), int(code == r["gt_code"]), int(bool(r["answerable"])), tied))
                n_rows += 1
    os.replace(tmp, out)
    os.replace(tmp_jsonl, out_jsonl)
    lh_core.write_json_atomic(out.with_suffix(".json"), {
        "files": {"jsonl": out_jsonl.name, "csv": out.name}, "model": model, "model_revision": revision,
        "run_dir": run_dir.name, "smoke": smoke, "stride": meta.get("stride", 1), "created_utc": stamp,
        "requests_sha256": requests_sha, "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        "requests_selected": len(requests), "requests_exported": n_req, "requests_without_result": len(requests) - n_req,
        "requests_with_tied_top": n_tied, "csv_rows": n_rows, "csv_columns": list(EXPORT_COLUMNS),
        "jsonl_contains_entity_text": True,
        "notes": {"position": "1-based place the option was shown in (A=1)",
                  "probability": "normalized probability as given to the scorer (full float precision)",
                  "prob_rank / rank": "1 = highest probability; ties broken by position, the same rule that picks the prediction",
                  "is_prediction": "the option the runner picked (highest probability, earliest position on a tie)",
                  "n_tied_top": "how many options share the highest probability (1 = no tie)",
                  "jsonl status": "ok, error (last failure kept in 'error') or not_run"}})
    lh_core.log(f"{model}: exported {n_req}/{len(requests)} requests ({n_tied} with a tie for the top) -> {out_jsonl.name} "
                f"(with input text) and {out.name} (codes only) in {out.parent}")
    return out


def cmd_export(args):
    export_options(args.run_dir, args.model.upper(), args.smoke)


def cmd_worker(args):
    model = args.model.upper()
    run_dir = Path(args.run_dir).resolve()
    spec = {"model": model, "device": "cuda:0", "project_dir": str(Path(args.weights_dir).resolve()),
            "config": {"seed": 2026, "max_tokens": 16000, "drex_int8": False}}
    info = lh_core.MODELS_INFO[model]
    mspec = importlib.util.spec_from_file_location(f"model_{model.lower()}", HERE / info["file"])
    module = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(module)
    out_path = run_dir / f"predictions_{model}{'_smoke' if args.stride > 1 else ''}.jsonl"
    lock = lh_core.RunLock(out_path.with_name(out_path.name + ".lock"))
    try:
        lock.acquire()   # held until the process exits; a second worker on the same file stops here
    except lh_core.RunAborted:
        sys.exit(f"{out_path.name} is being written by another {model} worker; wait for it or stop it first.")
    # the predictions belong to one requests.jsonl: refuse to resume against a different one
    requests_sha = hashlib.sha256((run_dir / "requests.jsonl").read_bytes()).hexdigest()
    meta_path = out_path.with_name(out_path.name + ".meta.json")
    if meta_path.exists():
        recorded = json.loads(meta_path.read_text()).get("requests_sha256")
        if recorded != requests_sha:
            sys.exit(f"requests.jsonl changed since {out_path.name} was started (sha {str(recorded)[:12]} -> {requests_sha[:12]}); "
                     "use a new --run-dir.")
    elif out_path.exists():
        sys.exit(f"{out_path.name} exists without {meta_path.name}, so its requests cannot be checked; move it away or use a new --run-dir.")
    else:
        lh_core.write_json_atomic(meta_path, {"requests_sha256": requests_sha, "model": model, "stride": args.stride})
    requests = load_requests(run_dir, args.stride)
    old = [r["request_id"] for r in requests if lh_core.QUESTION_ID not in r["record"]["questions"]]
    if old:   # only the models read the question; report and export work on older run folders too
        sys.exit(f"STOPPED: {len(old)} request(s) in {run_dir / 'requests.jsonl'} were built by an older version (question id "
                 f"is not '{lh_core.QUESTION_ID}'); build again into a new --run-dir")
    adapter = module.Adapter(spec)
    adapter.check_environment()
    adapter.load()
    done = {r["request_id"] for r in lh_core.read_jsonl_recovering(out_path) if r.get("status") == "ok"} if out_path.exists() else set()
    todo = [r for r in requests if r["request_id"] not in done]
    lh_core.log(f"{model}: {len(requests)} requests, {len(done)} already done, {len(todo)} to run -> {out_path.name}")
    consecutive, t_start = 0, time.perf_counter()
    with lh_core.JsonlAppender(out_path) as sink:
        for i, r in enumerate(todo, 1):
            choices = list(r["record"]["questions"][lh_core.QUESTION_ID]["criteria"])
            t0 = time.perf_counter()
            row = {"request_id": r["request_id"], "model": model}
            failure = None
            try:
                probs = lh_core.normalize_probabilities(adapter.predict(r["record"]), choices)
                letter = max(probs, key=probs.get)
                row.update(status="ok", prediction_letter=letter, prediction_code=r["codes"][choices.index(letter)],
                           probabilities=probs, latency_s=round(time.perf_counter() - t0, 4))
                consecutive = 0
            except Exception as exc:  # noqa: BLE001
                failure = exc
                row.update(status="error", error_type=type(exc).__name__, failure_class=lh_core.classify_failure(exc))
                consecutive += 1
            if failure is not None:
                adapter.cleanup_after_failure()   # outside the except block so the traceback does not pin tensors
            sink.write(row)
            if consecutive >= 5:
                lh_core.log(f"{model}: 5 consecutive errors, aborting (last: {row.get('error_type')}/{row.get('failure_class')})")
                sys.exit(3)
            if i % 200 == 0 or i == len(todo):
                rate = (time.perf_counter() - t_start) / i
                lh_core.log(f"{model}: {i}/{len(todo)} done, {rate:.2f}s/request, ETA {rate*(len(todo)-i)/60:.1f} min")
    rows = lh_core.read_jsonl_readonly(out_path)[0] if out_path.exists() else []
    ok = {r["request_id"] for r in rows if r.get("status") == "ok"}
    wanted = {r["request_id"] for r in requests}
    missing = wanted - ok
    if missing:
        last = {r["request_id"]: r for r in rows if r["request_id"] in missing}
        classes = collections.Counter(last[i].get("failure_class", "not run") if i in last else "not run" for i in missing)
        lh_core.log(f"{model}: finished with {len(wanted) - len(missing)}/{len(wanted)} ok; {len(missing)} without a result "
                    f"({dict(classes)}). Rerun the same command to retry them.")
    else:
        lh_core.log(f"{model}: finished, {len(wanted)}/{len(wanted)} ok")
    if ok & wanted:
        export_options(run_dir, model, args.stride > 1)


def cmd_run(args):
    import main as lhmain
    model = args.model.upper()
    env_name = lh_core.MODELS_INFO[model]["env"]
    python = lhmain.resolve_env_python(env_name)
    if python is None:
        sys.exit(f"conda env {env_name} not found: run `python main.py --project-dir {args.weights_dir} setup-envs --models {model.lower()}` first.")
    args.weights_dir = str(Path(args.weights_dir).expanduser().resolve())   # the worker starts in this folder, not the caller's
    if not Path(args.weights_dir).is_dir():
        sys.exit(f"--weights-dir {args.weights_dir} is not a folder (pass the folder given to setup-envs as --project-dir).")
    if not (Path(args.run_dir) / "requests.jsonl").exists():
        sys.exit(f"{args.run_dir} has no requests.jsonl; run decision_eval.py build first.")
    free, need = lhmain.gpu_free_gb(args.gpu), lh_core.MODELS_INFO[model]["min_vram_gb"]
    if free < need:
        sys.exit(f"Only {free:.1f} GB free on GPU {args.gpu}; {model} needs {need}.")
    env = lhmain.clean_env(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONNOUSERSITE="1",
                           PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
    lh_core.log(f"{model}: {python} on physical GPU {args.gpu} ({free:.1f} GB free)")
    rc = subprocess.run([python, str(HERE / "run_decision_model.py"), "worker", "--model", model, "--run-dir", str(Path(args.run_dir).resolve()),
                         "--weights-dir", args.weights_dir, "--stride", str(args.stride)], cwd=str(HERE), env=env).returncode
    sys.exit(rc)


# --- report -----------------------------------------------------------------------------------------------------------
def auroc(pos, neg):
    """P(score_pos > score_neg), ties count half."""
    if not pos or not neg:
        return float("nan")
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2          # mean 1-based rank of the tied block
        rank_sum += avg * sum(lbl for _, lbl in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cell_metrics(items):
    """Accuracy, Brier (summed over options) and 10-bin ECE on answerable items, as in the LongHealth report; acc_all counts misses as wrong."""
    answerable = [(r, p) for r, p in items if r["answerable"]]
    out = {"acc_all": sum(p["prediction_code"] == r["gt_code"] for r, p in items) / len(items) if items else float("nan"),
           "n": len(items), "n_answerable": len(answerable)}
    if not answerable:
        return {**out, "acc": float("nan"), "brier": float("nan"), "ece": float("nan")}
    bins = collections.defaultdict(list)
    brier = 0.0
    for r, p in answerable:
        brier += sum((v - (c == r["gt_letter"])) ** 2 for c, v in p["probabilities"].items())
        conf = max(p["probabilities"].values())
        bins[min(int(conf * 10), 9)].append((conf, p["prediction_code"] == r["gt_code"]))
    n = len(answerable)
    ece = sum(len(b) / n * abs(sum(x[1] for x in b) / len(b) - sum(x[0] for x in b) / len(b)) for b in bins.values())
    return {**out, "acc": sum(p["prediction_code"] == r["gt_code"] for r, p in answerable) / n, "brier": brier / n, "ece": ece}


def comparison_tables(reqs, all_preds, out_dir):
    """Model x top-k tables: headline = the two shuffled orders pooled; original and reversed reported separately."""
    views = (("shuffled", "Shuffled option order (headline; two shuffles pooled)", ("shuf1", "shuf2")),
             ("original", "Original rank order (position favours a model that picks the first option)", ("original",)),
             ("reversed", "Reversed rank order", ("reversed",)))
    names = {"ELEPHANT": "Elephant-1", "CLEF": "Clef", "DREX": "Drex Max"}
    md = ["# Decision models as ICD-10-CA code pickers", "",
          "Acc, Brier and ECE use answerable items only (reference code in the candidate list). "
          "Acc (all) counts items with no correct option as wrong.", "",
          "Evaluation design note: the dataset gives candidates as bare codes. run_decision_model.py adds each code's text from "
          "`code_descriptions`, so options read `CODE: description`. These scores measure matching a diagnosis to a code "
          "description, not knowledge of ICD codes from the code alone.", ""]
    for key, title, orders in views:
        expected = {k: sum(r["k"] == k and r["order"] in orders for r in reqs.values()) for k in KS}
        csv_rows, warnings = [], []
        for model, preds in all_preds.items():
            row = {"model": names[model]}
            for k in KS:
                items = [(reqs[i], p) for i, p in preds.items() if reqs[i]["k"] == k and reqs[i]["order"] in orders]
                m = cell_metrics(items)
                if m["n"] != expected[k]:
                    warnings.append(f"{names[model]} {k}: only {m['n']} of {expected[k]} requests")
                short = k.replace("top_", "top")
                row.update({f"{short}_acc": m["acc"], f"{short}_brier": m["brier"], f"{short}_ece": m["ece"], f"{short}_acc_all": m["acc_all"]})
                row[f"{short}_n_answerable"] = m["n_answerable"]
            csv_rows.append(row)
        with open(out_dir / f"comparison_{key}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
        head = ["Model"] + [f"Top-{k[4:]} {c}" for k in KS for c in ("Acc", "Brier", "ECE", "Acc (all)")]
        md += [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for row in csv_rows:
            cells = [f"{row[k.replace('top_', 'top') + '_' + c]:.3f}" for k in KS for c in ("acc", "brier", "ece", "acc_all")]
            md.append("| " + " | ".join([row["model"]] + cells) + " |")
        counts = ", ".join(f"Top-{k[4:]}: {csv_rows[0][k.replace('top_', 'top') + '_n_answerable']} of {expected[k]}" for k in KS)
        md += ["", f"Answerable requests: {counts}."]
        md += [f"**INCOMPLETE:** {w}" for w in warnings] + [""]
    text = "\n".join(md) + "\n"
    (out_dir / "comparison_tables.md").write_text(text, encoding="utf8")
    print("\n" + text)


def cmd_report(args):
    project = Path(args.run_dir).resolve()
    reqs = {r["request_id"]: r for r in map(json.loads, open(project / "requests.jsonl", encoding="utf8"))}
    rows, summary, all_preds = [], [], {}
    for model in ("ELEPHANT", "CLEF", "DREX"):
        path = project / f"predictions_{model}.jsonl"
        if not path.exists():
            continue
        preds = {p["request_id"]: p for p in lh_core.read_jsonl_recovering(path) if p.get("status") == "ok"}
        all_preds[model] = preds
        errors = sum(1 for p in lh_core.read_jsonl_recovering(path) if p.get("status") != "ok")
        for k in KS:
            for order in ORDERS:
                cell = [(reqs[i], p) for i, p in preds.items() if reqs[i]["k"] == k and reqs[i]["order"] == order]
                if not cell:
                    continue
                n = len(cell)
                ans = [(r, p) for r, p in cell if r["answerable"]]
                un = [(r, p) for r, p in cell if not r["answerable"]]
                correct = lambda r, p: p["prediction_code"] == r["gt_code"]
                conf = lambda p: max(p["probabilities"].values())
                rows.append({
                    "model": model, "k": k, "order": order, "n": n, "n_answerable": len(ans),
                    "acc_overall": sum(correct(r, p) for r, p in cell) / n,
                    "acc_answerable": sum(correct(r, p) for r, p in ans) / max(1, len(ans)),
                    "family3_overall": sum(p["prediction_code"][:3] == r["gt_code"][:3] for r, p in cell) / n,
                    "pred_first_option": sum(p["prediction_letter"] == "A" for r, p in cell) / n,
                    "pred_last_option": sum(p["prediction_letter"] == LETTERS[len(r["codes"]) - 1] for r, p in cell) / n,
                    "mean_p_gold_answerable": sum(p["probabilities"][r["gt_letter"]] for r, p in ans) / max(1, len(ans)),
                    "median_conf_answerable": sorted(conf(p) for r, p in ans)[len(ans) // 2] if ans else float("nan"),
                    "median_conf_unanswerable": sorted(conf(p) for r, p in un)[len(un) // 2] if un else float("nan"),
                    "auroc_conf_answerable_vs_not": auroc([conf(p) for r, p in ans], [conf(p) for r, p in un]),
                    "median_latency_s": sorted(p["latency_s"] for r, p in cell)[n // 2], "errors_total_model": errors})
            # stability between the two random orders, and agreement with the file's simulated picker
            pairs = [(preds.get(f"{e}|{k}|shuf1"), preds.get(f"{e}|{k}|shuf2")) for e in {r["entity_id"] for r in reqs.values()}]
            pairs = [(a, b) for a, b in pairs if a and b]
            if pairs:
                summary.append({"model": model, "k": k, "pairs": len(pairs),
                                "same_code_across_two_shuffles": sum(a["prediction_code"] == b["prediction_code"] for a, b in pairs) / len(pairs)})
    if not rows:
        sys.exit("no predictions found")
    fields = list(rows[0])
    with open(project / "rag_eval_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # the file's own simulated picker, for reference
    ref = collections.defaultdict(lambda: [0, 0, 0])
    for r in reqs.values():
        if r["order"] == "original":
            ref[r["k"]][0] += r["source_pick"] == r["gt_code"]
            ref[r["k"]][1] += 1
    if any(r["source_pick"] is not None for r in reqs.values()):
        print("Reference: simulated picker in the file (overall accuracy):", {k: round(a / n, 3) for k, (a, n, _) in ref.items()})
    cols = ["model", "k", "order", "n", "acc_overall", "acc_answerable", "family3_overall", "pred_first_option", "pred_last_option",
            "mean_p_gold_answerable", "auroc_conf_answerable_vs_not", "median_latency_s"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    print("\nStability across the two shuffled orders:")
    for s in summary:
        print(s)
    (project / "rag_eval_stability.json").write_text(json.dumps(summary, indent=2))
    comparison_tables(reqs, all_preds, project)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run", "worker", "report", "export"):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", required=True, help="output folder for this run, e.g. results/original_full")
        if name in ("run", "worker"):
            s.add_argument("--weights-dir", required=True, help="the models folder given to main.py setup-envs as --project-dir")
        if name == "build":
            s.add_argument("--src", default=str(DEFAULT_SRC))
            s.add_argument("--no-notes", action="store_true", help="context is the entity text only, no clinical note")
        if name in ("run", "worker"):
            models = [m for m in lh_core.MODELS_INFO] + [m.lower() for m in lh_core.MODELS_INFO]
            s.add_argument("model", choices=models) if name == "run" else s.add_argument("--model", required=True, choices=models)
            s.add_argument("--stride", type=positive_int, default=1)
        if name == "run":
            s.add_argument("--gpu", required=True)
        if name == "export":
            s.add_argument("model", choices=[m for m in lh_core.MODELS_INFO] + [m.lower() for m in lh_core.MODELS_INFO])
            s.add_argument("--smoke", action="store_true", help="export the smoke-test predictions instead of the full run")
    args = p.parse_args()
    {"build": cmd_build, "run": cmd_run, "worker": cmd_worker, "report": cmd_report, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
