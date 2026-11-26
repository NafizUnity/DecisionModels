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

--candidates retriever: no DM; the LLM sees the retriever's own top K (the fused ranking file's list, as kept in the DM
run's requests.jsonl), in retriever order, as the original RAG picker does (no reranked JSON). Its rows are named
RETRIEVER, and the report's dm_* columns then describe the retriever. Use it to reproduce the RAG picker's result.

A K above what a DM read (MedDecider-31B reads 10: K=20) is not built and reads N/A in the report. An entity the DM
has no result for is left out of that DM's inputs and counted. A list shorter than K goes to the LLM whole and is
counted. The entity text is the DM's own input (entity_text, with "(expanded: query_text)" when they differ); with
--entity-text mention it is entity_text alone, read from the DM's input file (the fused ranking file, sha256-checked),
as the RAG pipeline's picker gives it.

Clinical note (--note first --notes notes.jsonl): the LLM also sees the first --max-note-chars characters of the
entity's note, as the original picker does (the original prompt with its note section). Each entity's note_id, start
and end come from requests.jsonl (DM runs built with them), else from the DM's input file (the fused ranking file in
the DM build_meta.json, checked by sha256). notes.jsonl lines hold note_id and note_text. It stops if an entity has no
note_id or its note is missing. It counts mentions that lie past the cut (the LLM does not see them in the note) and
mentions whose note text at start:end does not match the entity text.
Output folders hold entity text and notes: keep them on the server, like the DM run folders. Standard library only.
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
from decision_eval import ORDERS, SOURCES, load_descriptions, load_notes, parse_ks, squash  # noqa: E402
from run_decision_model import model_view, parse_sources, rank_options, retriever_rankings  # noqa: E402

DM_NAMES = tuple(lh_core.MODELS_INFO)
RETRIEVER = "RETRIEVER"   # --candidates retriever: the row name in place of a DM
NOTE_MODES = ("none", "first")   # none: no clinical note; first: the note's first --max-note-chars characters


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dm_input(build, why):
    """The DM's input file (the fused ranking JSON in the DM build_meta.json), sha256-checked against the DM build."""
    src = Path(build.get("src") or "")
    if not src.is_file():
        sys.exit(f"STOPPED: {why}: the DM's input file {src} is not found")
    if sha256(src) != build.get("src_sha256"):
        sys.exit(f"STOPPED: the DM's input file {src} has changed since the DM build (sha256 differs)")
    return src, json.load(open(src, encoding="utf8"))


def mention_texts(build):
    """entity_id -> entity_text as written in the DM's input file (the mention, without the query_text expansion)."""
    src, data = dm_input(build, "--entity-text mention reads the mentions")
    lacking = [str(d.get("entity_id")) for d in data if not isinstance(d.get("entity_text"), str)]
    if lacking:
        sys.exit(f"STOPPED: {len(lacking)} entities in {src} have no entity_text (first: {lacking[:5]})")
    return {str(d["entity_id"]): d["entity_text"] for d in data}


def note_links(reqs, build):
    """entity_id -> (note_id, start, end), and where it came from: requests.jsonl when the DM build kept the note link,
    else the DM's input file (sha256-checked against the DM build)."""
    if all("note_id" in r for r in reqs):
        return {r["entity_id"]: (r["note_id"], r["start"], r["end"]) for r in reqs}, "requests.jsonl"
    src, data = dm_input(build, "requests.jsonl has no note_id")
    lacking = [str(d.get("entity_id")) for d in data if not all(k in d for k in ("note_id", "start", "end"))]
    if lacking:
        sys.exit(f"STOPPED: {len(lacking)} entities in {src} have no note_id/start/end (first: {lacking[:5]})")
    return {str(d["entity_id"]): (str(d["note_id"]), d["start"], d["end"]) for d in data}, f"DM input file {src}"


def stem(source, dm):
    return f"{source}__{dm}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dm-run", required=True, help="the DM run folder (requests.jsonl, predictions_<DM>.jsonl)")
    ap.add_argument("--out", required=True, help="the picker folder to create (or reuse, when built with the same settings)")
    ap.add_argument("--llm-model", required=True, help="the LLM's folder (written into the configs)")
    ap.add_argument("--dms", type=parse_sources, default=None, metavar="'DREX MEDDECIDER31'",
                    help="the DMs whose reranking the LLM sees (needed with --candidates dm)")
    ap.add_argument("--candidates", default="dm", choices=("dm", "retriever"),
                    help="dm: each DM's top K in DM order (default); retriever: the retriever's top K in its own order")
    ap.add_argument("--sources", type=parse_sources, default=None, help="default: all sources")
    ap.add_argument("--ks", type=parse_ks, default=parse_ks("3,5,10,20"), metavar="'3,5,10,20'",
                    help="how many of the DM's top codes the LLM sees (default 3,5,10,20)")
    ap.add_argument("--order", default="original", choices=ORDERS,
                    help="which DM request to take the probabilities from (default original)")
    ap.add_argument("--format", default="native", choices=("native", "requests"))
    ap.add_argument("--note", default="none", choices=NOTE_MODES,
                    help="none: no clinical note (default); first: the note's first --max-note-chars characters")
    ap.add_argument("--notes", default=None, help="notes.jsonl (note_id, note_text); needed with --note first")
    ap.add_argument("--stop-tokens", default="generation_config", choices=("generation_config", "tokenizer_eos"),
                    help="generation_config: stop at the model's end tokens; tokenizer_eos: the original script's behaviour")
    ap.add_argument("--entity-text", default="expanded", choices=("expanded", "mention"),
                    help="expanded: the DM's input, entity_text (expanded: query_text) (default); mention: entity_text "
                         "only, as written in the fused ranking file")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-note-chars", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None, help="only the first N entities (a smoke test)")
    args = ap.parse_args()

    dm_run, out = Path(args.dm_run).resolve(), Path(args.out).resolve()
    if args.candidates == "retriever":
        if args.dms:
            sys.exit("STOPPED: --candidates retriever uses no DM; leave out --dms")
        dms = [RETRIEVER]
    else:
        if not args.dms:
            sys.exit("STOPPED: --dms is needed (or --candidates retriever)")
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
    if (args.note != "none") != bool(args.notes):
        sys.exit("STOPPED: --note first needs --notes notes.jsonl, and --notes is only used with --note first")
    if args.notes and not Path(args.notes).is_file():
        sys.exit(f"STOPPED: --notes {args.notes} not found")
    if args.max_note_chars < 1:
        sys.exit("STOPPED: --max-note-chars must be 1 or more")
    if args.max_new_tokens < 1:
        sys.exit("STOPPED: --max-new-tokens must be 1 or more")

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
    for dm in (d for d in dms if d != RETRIEVER):
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

    # ---- the clinical notes (--note first): entity -> note_id/start/end -> note text ----
    links, notes, link_from = {}, {}, None
    if args.note != "none":
        links, link_from = note_links(reqs, build)
        notes = load_notes(args.notes)
        unlinked = sorted(e for e in keep if e not in links)
        if unlinked:
            sys.exit(f"STOPPED: {len(unlinked)} entities have no note link (first: {unlinked[:5]})")
        missing = sorted({links[e][0] for e in keep} - set(notes))
        if missing:
            sys.exit(f"STOPPED: {len(missing)} note_id(s) of these entities are not in {args.notes} (first: {missing[:5]})")

    # ---- the entity text the LLM sees: the DM's input (default) or the mention alone ----
    # a DM run built with --notes has the note in its context; its entity text alone is entity_state
    state = {r["entity_id"]: r.get("entity_state", r["record"]["state"]) for r in reqs if r["entity_id"] in keep}
    if args.entity_text == "mention":
        mentions = mention_texts(build)
        absent = sorted(e for e in keep if e not in mentions)
        if absent:
            sys.exit(f"STOPPED: {len(absent)} entities are not in the DM's input file (first: {absent[:5]})")
        # the DM's input starts with the mention; anything else means the file does not belong to this DM run
        odd = sorted(e for e in keep if not squash(state[e]).startswith(squash(mentions[e]) or "\0"))
        if odd:
            sys.exit(f"STOPPED: {len(odd)} mentions in the DM's input file do not start the DM's entity text "
                     f"(first: {odd[:5]}); report it as a bug")
        shown_text = {e: mentions[e] for e in keep}
    else:
        shown_text = state
    n_expanded = sum(shown_text[e] != state[e] for e in keep) if args.entity_text == "mention" else 0

    settings = {"dm_run": str(dm_run), "requests_sha256": requests_sha, "predictions_sha256": pred_sha,
                "descriptions": str(desc_path), "descriptions_sha256": build["descriptions_sha256"],
                "dms": dms, "sources": sources, "ks": args.ks, "order": args.order, "format": args.format,
                "note": args.note, "llm_model": str(Path(args.llm_model).resolve()), "stop_tokens": args.stop_tokens,
                "max_new_tokens": args.max_new_tokens, "max_note_chars": args.max_note_chars, "limit": args.limit}
    if args.candidates != "dm":   # added only when set, so folders built before this keep matching
        settings["candidates"] = args.candidates
    if args.entity_text != "expanded":   # likewise
        settings["entity_text"] = args.entity_text
    if args.note != "none":   # added only with notes, so no-note folders built before this keep matching
        settings.update(notes=str(Path(args.notes).resolve()), notes_sha256=sha256(args.notes), note_link_from=link_from)
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
            limit = lh_core.MODELS_INFO[dm]["max_options"] if dm != RETRIEVER else longest
            ks = [k for k in args.ks if k <= limit]
            na = [k for k in args.ks if k > limit]
            entities, skipped, short = [], 0, {k: 0 for k in ks}
            for r in reqs:
                if r["source"] != source or r["order"] != args.order or r["entity_id"] not in keep:
                    continue
                if dm == RETRIEVER:   # the retriever's whole list in its own order; the LLM sees its first K
                    retr_order = dm_order = list(rankings[(r["entity_id"], source)])
                    rank = {c: i for i, c in enumerate(dm_order, 1)}
                else:
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
            # the note of each entity: without --note an empty note of its own (note_id = entity_id), as before
            for e in entities:
                eid = e["r"]["entity_id"]
                nid, start, end = links[eid] if links else (eid, 0, 0)
                e.update(note_id=nid, start=start, end=end, note_text=notes.get(nid, ""), text=shown_text[eid])
            cnt = {"source": source, "dm": dm, "entities": len(entities), "skipped_no_dm_result": skipped,
                   "ks": ks, "ks_na": na, "short_lists": short}
            if args.entity_text == "mention":   # entities whose shown text lost an expansion
                cnt["expansion_dropped"] = sum(e["text"] != state[e["r"]["entity_id"]] for e in entities)
            if notes:
                cnt["notes"] = len({e["note_id"] for e in entities})
                # the mention ends past the cut: the LLM does not see it in the note it is shown
                cnt["mention_past_cut"] = sum(e["end"] > args.max_note_chars for e in entities)
                # note_text[start:end] is not the entity text (the DM's input starts with the entity text)
                cnt["span_mismatch"] = sum(not squash(e["text"]).startswith(
                    squash(e["note_text"][e["start"]:e["end"]]) or "\0") for e in entities)
            counts.append(cnt)
            if not entities:
                continue
            pool = max(len(e["dm_order"]) for e in entities)
            name = stem(source, dm)
            if args.format == "native":
                by_note = {}
                for e in entities:   # one note per note_id holding its entities, as the picker's own input
                    by_note.setdefault(e["note_id"], {"note_id": e["note_id"], "note_text": e["note_text"],
                                                      "entities": []})["entities"].append({
                        "entity_id": e["r"]["entity_id"], "entity_text": e["text"],
                        "start": e["start"], "end": e["end"], "ground_truth_code": e["r"]["gt_code"],
                        "rag_outputs": {f"top_{pool}": {"candidate_codes": e["retr_order"]}}})
                doc = {"code_descriptions": {c: desc[c] for e in entities for c in e["dm_order"]},
                       "patient_notes": list(by_note.values())}
                files[f"native/{name}.input.json"] = json.dumps(doc, ensure_ascii=False, indent=1)
                picker_paths = {"input": f"../native/{name}.input.json"}
                if dm != RETRIEVER:   # the retriever run is the original pipeline: no reranked JSON
                    files[f"native/{name}.reranked.json"] = json.dumps(
                        {e["r"]["entity_id"]: e["dm_order"] for e in entities}, ensure_ascii=False, indent=1)
                    picker_paths["reranked"] = f"../native/{name}.reranked.json"
            else:
                files[f"requests/{name}.jsonl"] = "".join(json.dumps({
                    "entity_id": e["r"]["entity_id"], "source": source, "dm": dm, "note_id": e["note_id"],
                    "start": e["start"], "end": e["end"], "entity_text": e["text"],
                    "note_text": e["note_text"], "ground_truth_code": e["r"]["gt_code"],
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

    print(f"built {out} ({args.format} format, note: {args.note}, " + (
        "candidates: the retriever's own top K, retriever order)" if args.candidates == "retriever"
        else f"DM order from the {args.order} requests)"))
    print("  entity text: " + ("the DM's input, entity_text (expanded: query_text) when they differ" if
          args.entity_text == "expanded" else f"the mention alone (entity_text); {n_expanded} of {len(keep)} entities "
          "lose an expansion") + f"; max_new_tokens {args.max_new_tokens}")
    print(f"  {'source':<16} {'DM':<13} {'entities':>8} {'no DM result':>12}  K built (N/A)       short lists (< K)")
    for c in counts:
        print(f"  {c['source']:<16} {c['dm']:<13} {c['entities']:>8} {c['skipped_no_dm_result']:>12}  "
              f"{','.join(map(str, c['ks'])) or '-':<10} {('(' + ','.join(map(str, c['ks_na'])) + ')') if c['ks_na'] else '':<8} "
              + " ".join(f"k{k}:{n}" for k, n in c["short_lists"].items()))
    if notes:
        print(f"  notes from {args.notes} (note link from {link_from}); the LLM sees each note's first "
              f"{args.max_note_chars} characters")
        for c in counts:
            print(f"  {c['source']:<16} {c['dm']:<13} {c.get('notes', 0):>5} notes  mention past the cut: "
                  f"{c.get('mention_past_cut', 0)}/{c['entities']}  span does not match the entity text: "
                  f"{c.get('span_mismatch', 0)}")
    print(f"  {sum(1 for f in files if f.startswith('configs/'))} picker configs in {out / 'configs'}")


if __name__ == "__main__":
    main()
