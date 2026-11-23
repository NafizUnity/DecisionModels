#!/usr/bin/env python3
"""Build the LLM picker's inputs from a finished decision-model (DM) run: the DM's reranking of each list, cut to K.

  python picker_build.py --dm-run RESULTS_DIR/real_whole_list_desc --out PICKER_RESULTS_DIR/my_picker \
      --llm-model /path/to/gemma-4-31b-it --dms "DREX MEDDECIDER31" --sources "rrf_csv+fsn rrf_fsn+syn" --ks "3,5,10,20"

For every source x DM it reads the DM's probabilities for the codes that DM read (its first max_options codes of the
retriever's list), sorts them high to low (ties: the earlier code in the shown order, as in the DM report), and writes:
  --format native (default): the picker's own input, as the RAG pipeline gives it
        native/<source>__<DM>.input.json     consolidated JSON: code_descriptions + one empty note per entity, whose
                                             rag_outputs hold the codes the DM read, in retriever order
        native/<source>__<DM>.reranked.json  entity_id -> the same codes in DM order (the picker shows the first K)
  --format requests: requests/<source>__<DM>.jsonl, one line per entity with its candidates already in DM order
  configs/<source>__<DM>__k<K>.yaml   one llm_picker.py config per K (paths in it are relative to the config file)
  truth.jsonl                         ids, codes and ranks only (no entity text), for picker_report.py
  build_meta.json                     the inputs (with sha256) and the settings

A K above what a DM read (MedDecider-31B reads 10: K=20) is not built and reads N/A in the report. An entity the DM
has no result for is left out of that DM's inputs and counted. A list shorter than K goes to the LLM whole and is
counted. The entity text is the DM's own input (entity_text, with "(expanded: query_text)" when they differ).
Output folders hold entity text: keep them on the server, like the DM run folders. Standard library only.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core  # noqa: E402
from decision_eval import ORDERS, SOURCES, load_descriptions, parse_ks  # noqa: E402
from run_decision_model import model_view, parse_sources, rank_options, retriever_rankings  # noqa: E402

DM_NAMES = tuple(lh_core.MODELS_INFO)
NOTE_MODES = ("none",)   # "none": no clinical note (the only mode the DM data supports so far)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stem(source, dm):
    return f"{source}__{dm}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dm-run", required=True, help="the DM run folder (requests.jsonl, predictions_<DM>.jsonl)")
    ap.add_argument("--out", required=True, help="the picker folder to create (or reuse, when built with the same settings)")
    ap.add_argument("--llm-model", required=True, help="the LLM's folder (written into the configs)")
    ap.add_argument("--dms", type=parse_sources, required=True, metavar="'DREX MEDDECIDER31'")
    ap.add_argument("--sources", type=parse_sources, default=None, help="default: all sources")
    ap.add_argument("--ks", type=parse_ks, default=parse_ks("3,5,10,20"), metavar="'3,5,10,20'",
                    help="how many of the DM's top codes the LLM sees (default 3,5,10,20)")
    ap.add_argument("--order", default="original", choices=ORDERS,
                    help="which DM request to take the probabilities from (default original)")
    ap.add_argument("--format", default="native", choices=("native", "requests"))
    ap.add_argument("--note", default="none", choices=NOTE_MODES, help="clinical note in the prompt (only none so far)")
    ap.add_argument("--stop-tokens", default="generation_config", choices=("generation_config", "tokenizer_eos"),
                    help="generation_config: stop at the model's end tokens; tokenizer_eos: the original script's behaviour")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-note-chars", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None, help="only the first N entities (a smoke test)")
    args = ap.parse_args()

    dm_run, out = Path(args.dm_run).resolve(), Path(args.out).resolve()
    dms = [d.upper() for d in args.dms]
    bad = [d for d in dms if d not in DM_NAMES]
    if bad:
        sys.exit(f"STOPPED: unknown DM(s) {bad}; choose from {' '.join(DM_NAMES)}")
    sources = list(args.sources) if args.sources else list(SOURCES)
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        sys.exit(f"STOPPED: unknown source(s) {bad}; choose from {' '.join(SOURCES)}")
    if args.limit is not None and args.limit < 1:
        sys.exit("STOPPED: --limit must be 1 or more")
    if not Path(args.llm_model).is_dir():
        sys.exit(f"STOPPED: --llm-model {args.llm_model} is not a folder")

    # ---- the DM run: requests, descriptions, predictions (each checked against its record) ----
    req_path = dm_run / "requests.jsonl"
    if not req_path.is_file():
        sys.exit(f"STOPPED: {req_path} not found (is --dm-run a DM run folder?)")
    build = json.loads((dm_run / "build_meta.json").read_text())
    if not build.get("descriptions"):
        sys.exit("STOPPED: that DM run was built without descriptions (bare codes); the picker needs code descriptions. "
                 "Use a DM run built with realdata_run.sh --desc")
    desc_path = Path(build["descriptions"])
    if not desc_path.is_file() or sha256(desc_path) != build["descriptions_sha256"]:
        sys.exit(f"STOPPED: {desc_path} is missing or has changed since the DM build (sha256 differs)")
    desc = load_descriptions(desc_path)
    reqs = [json.loads(line) for line in open(req_path, encoding="utf8")]
    requests_sha = sha256(req_path)
    preds, pred_sha = {}, {}
    for dm in dms:
        path = dm_run / f"predictions_{dm}.jsonl"
        meta = path.with_name(path.name + ".meta.json")
        if not path.is_file() or not meta.is_file():
            sys.exit(f"STOPPED: {path.name} (or its .meta.json) not found in {dm_run}: run that DM first")
        info = json.loads(meta.read_text())
        if info.get("requests_sha256") != requests_sha:
            sys.exit(f"STOPPED: {path.name} was not made from this requests.jsonl")
        if info.get("stride", 1) != 1:
            sys.exit(f"STOPPED: {path.name} is a smoke-test sample (stride {info['stride']})")
        preds[dm] = {p["request_id"]: p for p in lh_core.read_jsonl_readonly(path)[0] if p.get("status") == "ok"}
        pred_sha[dm] = sha256(path)

    entity_order = list(dict.fromkeys(r["entity_id"] for r in reqs))
    if args.limit:
        entity_order = entity_order[:args.limit]
    keep = set(entity_order)
    longest = max(r["n_options"] for r in reqs)
    rankings = retriever_rankings(reqs)

    settings = {"dm_run": str(dm_run), "requests_sha256": requests_sha, "predictions_sha256": pred_sha,
                "descriptions": str(desc_path), "descriptions_sha256": build["descriptions_sha256"],
                "dms": dms, "sources": sources, "ks": args.ks, "order": args.order, "format": args.format,
                "note": args.note, "llm_model": str(Path(args.llm_model).resolve()), "stop_tokens": args.stop_tokens,
                "max_new_tokens": args.max_new_tokens, "max_note_chars": args.max_note_chars, "limit": args.limit}
    meta_path = out / "build_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("settings") == settings:
            print(f"build unchanged: {out} (reused)")
            return
        diff = sorted(k for k in settings if old.get("settings", {}).get(k) != settings[k])
        sys.exit(f"STOPPED: {out} was built with other settings ({', '.join(diff)}); use a new folder (--name)")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"STOPPED: {out} is not empty and has no build_meta.json; use a new folder")

    # ---- per source x DM: the DM's order of the codes it read ----
    counts, truth, files = [], [], {}
    for source in sources:
        for dm in dms:
            limit = lh_core.MODELS_INFO[dm]["max_options"]
            ks = [k for k in args.ks if k <= limit]
            na = [k for k in args.ks if k > limit]
            entities, skipped, short = [], 0, {k: 0 for k in ks}
            for r in reqs:
                if r["source"] != source or r["order"] != args.order or r["entity_id"] not in keep:
                    continue
                p = preds[dm].get(r["request_id"])
                if p is None:
                    skipped += 1
                    continue
                v = model_view(r, limit, rankings) if r["n_options"] > limit else r   # the question the DM read
                shown = list(v["record"]["questions"][lh_core.QUESTION_ID]["criteria"])
                if p.get("shown_codes", v["codes"]) != v["codes"] or set(p["probabilities"]) != set(shown):
                    sys.exit(f"STOPPED: {r['request_id']}: the DM's codes differ from the ones rebuilt now; report it as a bug")
                rank = rank_options(shown, p["probabilities"])
                dm_order = sorted(shown, key=rank.get)
                read = set(shown)
                retr_order = [c for c in rankings[(r["entity_id"], source)] if c in read]   # retriever order of the same codes
                for k in ks:
                    short[k] += len(dm_order) < k
                gt = r["gt_code"]
                entities.append({"r": r, "dm_order": dm_order, "retr_order": retr_order})
                truth.append({"source": source, "dm": dm, "entity_id": r["entity_id"], "gt_code": gt,
                              "n_read": len(dm_order), "retriever_rank": r.get("retriever_rank"),
                              "dm_rank": rank.get(gt), "dm_top1": dm_order[0]})
            counts.append({"source": source, "dm": dm, "entities": len(entities), "skipped_no_dm_result": skipped,
                           "ks": ks, "ks_na": na, "short_lists": short})
            if not entities:
                continue
            pool = max(len(e["dm_order"]) for e in entities)
            name = stem(source, dm)
            if args.format == "native":
                doc = {"code_descriptions": {c: desc[c] for e in entities for c in e["dm_order"]},
                       "patient_notes": [{"note_id": e["r"]["entity_id"], "note_text": "", "entities": [{
                           "entity_id": e["r"]["entity_id"], "entity_text": e["r"]["record"]["state"],
                           "start": 0, "end": 0, "ground_truth_code": e["r"]["gt_code"],
                           "rag_outputs": {f"top_{pool}": {"candidate_codes": e["retr_order"]}}}]} for e in entities]}
                files[f"native/{name}.input.json"] = json.dumps(doc, ensure_ascii=False, indent=1)
                files[f"native/{name}.reranked.json"] = json.dumps(
                    {e["r"]["entity_id"]: e["dm_order"] for e in entities}, ensure_ascii=False, indent=1)
                picker_paths = {"input": f"../native/{name}.input.json", "reranked": f"../native/{name}.reranked.json"}
            else:
                files[f"requests/{name}.jsonl"] = "".join(json.dumps({
                    "entity_id": e["r"]["entity_id"], "source": source, "dm": dm,
                    "entity_text": e["r"]["record"]["state"], "note_text": "", "ground_truth_code": e["r"]["gt_code"],
                    "candidates": [{"code": c, "description": desc[c]} for c in e["dm_order"]]},
                    ensure_ascii=False) + "\n" for e in entities)
                picker_paths = {"input": f"../requests/{name}.jsonl"}
            for k in ks:
                cfg = {"picker": {"format": args.format, **picker_paths,
                                  "output": f"../predictions/{name}__k{k}.jsonl",
                                  "run": f"k{k}", "model_key": "llm", "candidate_pool": pool},
                       "rag": {"runs": {f"k{k}": {"k": k, "max_note_chars": args.max_note_chars,
                                                   "use_note": args.note != "none"}},
                               "models": {"llm": {"backend": "transformers", "model_name": settings["llm_model"],
                                                  "device": "auto", "max_new_tokens": args.max_new_tokens,
                                                  "stop_tokens": args.stop_tokens}}}}
                files[f"configs/{name}__k{k}.yaml"] = json.dumps(cfg, indent=1)   # JSON is valid YAML

    if not truth:
        sys.exit("STOPPED: no entity has a DM result for these sources and DMs; nothing was written")
    out.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf8")
    (out / "predictions").mkdir(exist_ok=True)
    (out / "truth.jsonl").write_text("".join(json.dumps(t) + "\n" for t in truth), encoding="utf8")
    lh_core.write_json_atomic(meta_path, {"settings": settings, "counts": counts, "longest_dm_list": longest,
                                          "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})

    print(f"built {out} ({args.format} format, note: {args.note}, DM order from the {args.order} requests)")
    print(f"  {'source':<16} {'DM':<13} {'entities':>8} {'no DM result':>12}  K built (N/A)       short lists (< K)")
    for c in counts:
        print(f"  {c['source']:<16} {c['dm']:<13} {c['entities']:>8} {c['skipped_no_dm_result']:>12}  "
              f"{','.join(map(str, c['ks'])) or '-':<10} {('(' + ','.join(map(str, c['ks_na'])) + ')') if c['ks_na'] else '':<8} "
              + " ".join(f"k{k}:{n}" for k, n in c["short_lists"].items()))
    print(f"  {sum(1 for f in files if f.startswith('configs/'))} picker configs in {out / 'configs'}")


if __name__ == "__main__":
    main()
