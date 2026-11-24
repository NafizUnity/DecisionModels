#!/usr/bin/env python3
"""Report an LLM picker folder (picker_build.py + llm_picker.py): the LLM's top-1 against the DM it reranked from.

  python picker_report.py --picker-dir PICKER_RESULTS_DIR/my_picker            # print only
  python picker_report.py --picker-dir PICKER_RESULTS_DIR/my_picker --save     # also report_<UTC time>.md/.csv

One row per source x DM x K, on the entities the DM has a result for:
  dm_hit@K       the reference code is in the DM's top K: the most the LLM can reach at this K
  dm_top1        the DM's own first code is right (the result without the LLM)
  llm_top1       the LLM's pick is right; delta = llm_top1 - dm_top1
  llm_acc_in     the LLM is right, among entities whose reference code it was shown
  fixed / broke  the DM's first code was wrong and the LLM is right / the DM's was right and the LLM is wrong
  no_answer, not_in_list   LLM replies without a readable code / with a code outside the list (both count as wrong)
  retr_top1      the retriever's own first code is right
A K above what that DM read is N/A; a run without predictions yet is "not run". Codes and ids only, no entity text.
--save writes new timestamped files; it never overwrites.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def dotless(code):
    return (code or "").upper().replace(".", "")


def f3(x):
    return "N/A" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picker-dir", required=True)
    ap.add_argument("--save", action="store_true", help="also write report_<UTC time>.md and .csv into the folder")
    args = ap.parse_args()
    out = Path(args.picker_dir).resolve()
    meta = json.loads((out / "build_meta.json").read_text())
    s = meta["settings"]
    truth = {}
    for line in open(out / "truth.jsonl", encoding="utf8"):
        t = json.loads(line)
        truth.setdefault((t["source"], t["dm"]), {})[t["entity_id"]] = t

    rows = []
    for c in meta["counts"]:
        source, dm = c["source"], c["dm"]
        ents = truth.get((source, dm), {})
        n = len(ents)
        retr1 = sum(t["retriever_rank"] == 1 for t in ents.values()) / n if n else None
        dm1 = sum(t["dm_rank"] == 1 for t in ents.values()) / n if n else None
        for k in s["ks"]:
            row = {"source": source, "dm": dm, "k": k, "entities": n, "skipped_no_dm_result": c["skipped_no_dm_result"],
                   "short_lists": c["short_lists"].get(str(k), c["short_lists"].get(k)), "retr_top1": retr1, "dm_top1": dm1}
            if k in c["ks_na"]:
                rows.append({**row, "status": "N/A"})
                continue
            hit = sum(t["dm_rank"] is not None and t["dm_rank"] <= k for t in ents.values())
            row["dm_hit@K"] = hit / n if n else None
            path = out / "predictions" / f"{source}__{dm}__k{k}.jsonl"
            if not path.exists():
                rows.append({**row, "status": "not run"})
                continue
            preds = {p["entity_id"]: p for p in map(json.loads, open(path, encoding="utf8"))}
            pmeta = path.with_name(path.name + ".meta.json")
            row["batch_size"] = (json.loads(pmeta.read_text())["model"].get("batch_size", 1)
                                 if pmeta.exists() else "unknown")
            missing = set(ents) - set(preds)
            if missing or set(preds) - set(ents):
                sys.exit(f"STOPPED: {path.name} does not cover the built entities ({len(missing)} missing); rebuild or rerun")
            right = {e: dotless(p["llm_icd_code"]) == dotless(ents[e]["gt_code"]) for e, p in preds.items()}
            shown = {e: ents[e]["dm_rank"] is not None and ents[e]["dm_rank"] <= k for e in ents}
            row.update(status="ok", done=len(preds),
                       llm_top1=sum(right.values()) / n,
                       llm_acc_in=(sum(right[e] for e in ents if shown[e]) / hit) if hit else None,
                       fixed=sum(right[e] and ents[e]["dm_rank"] != 1 for e in ents),
                       broke=sum(not right[e] and ents[e]["dm_rank"] == 1 for e in ents),
                       no_answer=sum(p["status"] == "no_answer" for p in preds.values()),
                       not_in_list=sum(p["status"] == "not_in_list" for p in preds.values()))
            row["delta"] = row["llm_top1"] - dm1
            rows.append(row)

    head = ["source", "dm", "k", "entities", "dm_hit@K", "dm_top1", "llm_top1", "delta", "llm_acc_in", "fixed", "broke",
            "no_answer", "not_in_list", "retr_top1"]
    table = [head]
    for r in rows:
        if r["status"] != "ok":
            table.append([r["source"], r["dm"], str(r["k"]), str(r["entities"]),
                          f3(r.get("dm_hit@K")), f3(r["dm_top1"])] + [r["status"]] + [""] * 6 + [f3(r["retr_top1"])])
            continue
        table.append([r["source"], r["dm"], str(r["k"]), str(r["entities"]), f3(r["dm_hit@K"]), f3(r["dm_top1"]),
                      f3(r["llm_top1"]), f"{r['delta']:+.3f}", f3(r["llm_acc_in"]), str(r["fixed"]), str(r["broke"]),
                      str(r["no_answer"]), str(r["not_in_list"]), f3(r["retr_top1"])])
    widths = [max(len(t[i]) for t in table) for i in range(len(head))]
    text = ["  ".join(c.ljust(w) for c, w in zip(t, widths)) for t in table]
    text.insert(1, "  ".join("-" * w for w in widths))
    note = "none" if s["note"] == "none" else f"the first {s['max_note_chars']} characters of each entity's note"
    notes = [f"DM run: {s['dm_run']} (DM order from the {s['order']} requests); format {s['format']}; note: {note}; "
             f"stop tokens: {s['stop_tokens']}" + (f"; SMOKE: first {s['limit']} entities only" if s["limit"] else "")]
    if s.get("candidates") == "retriever":
        notes.append("Candidates: the retriever's own top K in retriever order (no DM). In the RETRIEVER rows the dm_* "
                     "columns describe the retriever: dm_hit@K = the retriever's recall@K, dm_top1 = its first code")
    if s["note"] != "none":
        notes.append("Mention past the note cut (the LLM sees the note without it): " + ", ".join(
            f"{c['source']}/{c['dm']}: {c.get('mention_past_cut', 0)}/{c['entities']}" for c in meta["counts"]))
        bad = [c for c in meta["counts"] if c.get("span_mismatch")]
        if bad:
            notes.append("WARNING: note text at start:end differs from the entity text: " + ", ".join(
                f"{c['source']}/{c['dm']}: {c['span_mismatch']}" for c in bad) + " (check the note ids)")
    skipped = [c for c in meta["counts"] if c["skipped_no_dm_result"]]
    if skipped:
        notes.append("Left out (no DM result): " + ", ".join(f"{c['source']}/{c['dm']}: {c['skipped_no_dm_result']}" for c in skipped))
    sizes = sorted({str(r["batch_size"]) for r in rows if "batch_size" in r})
    if sizes:
        notes.append(f"LLM batch size: {', '.join(sizes)}" + (
            "  WARNING: these runs used different batch sizes; bf16 picks can differ between batch sizes, so compare "
            "rows of one batch size" if len(sizes) > 1 else ""))
    shorts = [f"{r['source']}/{r['dm']}/k{r['k']}: {r['short_lists']}" for r in rows if r.get("short_lists")]
    if shorts:
        notes.append("Lists shorter than K (given whole): " + ", ".join(shorts))
    print("\n".join(text + [""] + notes))
    if args.save:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        md, cs = out / f"report_{stamp}.md", out / f"report_{stamp}.csv"
        md.write_text("```\n" + "\n".join(text) + "\n```\n\n" + "\n\n".join(notes) + "\n", encoding="utf8")
        keys = ["source", "dm", "k", "status", "entities", "done", "dm_hit@K", "dm_top1", "llm_top1", "delta", "llm_acc_in",
                "fixed", "broke", "no_answer", "not_in_list", "retr_top1", "skipped_no_dm_result", "short_lists",
                "batch_size"]
        with open(cs, "x", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("N/A" if r.get(k) is None and k in r else r.get(k, "")) for k in keys})
        print(f"\nsaved {md.name} and {cs.name} in {out}")


if __name__ == "__main__":
    main()
