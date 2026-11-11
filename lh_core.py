"""Shared library for the LongHealth runner. Standard library only: it is imported by main.py and by every model worker,
which run in different Python environments (3.10 / 3.11 / 3.12). Never import torch or any third-party package here.

Benchmark text must never be printed or logged from this module.
"""
import argparse
import collections
import fcntl
import gzip
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------
# Static model table, shared by main.py and the workers
# ----------------------------------------------------------------------------------------------------------------------
MODELS_INFO = {
    "ELEPHANT": {"env": "realdata-elephant", "file": "model_elephant.py", "repo": "baselabs/elephant-1",
                 "weights_dir": "elephant-1", "min_vram_gb": 22, "python": "3.11"},
    "CLEF": {"env": "realdata-clef", "file": "model_clef.py", "repo": "Cloudflare/clef",
             "weights_dir": "clef", "min_vram_gb": 60, "python": "3.11"},
    "DREX": {"env": "realdata-drex", "file": "model_drex.py", "repo": "nace-ai/drex-max-v1",
             "weights_dir": "drex-max-v1", "min_vram_gb": 58, "min_vram_gb_int8": 30, "python": "3.12"},
}
TASKS = ("task1", "task23")
EXIT_OK, EXIT_GATE, EXIT_ABORT, EXIT_ENV, EXIT_INTERRUPT, EXIT_CRASH = 0, 2, 3, 4, 130, 1


class OverlongInput(ValueError):
    """The request exceeds the model's token limit; recorded as 'overlong', never as a wrong answer."""


class EnvironmentMismatch(RuntimeError):
    """The worker's environment does not match what the adapter requires."""


class RunAborted(RuntimeError):
    """A guard stopped the run. Rows already written stay on disk."""


class GateFailed(RuntimeError):
    """A pre-run gate failed."""


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Small pure helpers (ported from the notebook)
# ----------------------------------------------------------------------------------------------------------------------
def normalize_probabilities(p, choices):
    if not isinstance(p, dict):
        raise TypeError("Expected keyed probabilities")
    if set(p) != set(choices):
        raise ValueError(f"Choices mismatch: returned {set(p)} expected {set(choices)}")
    result = {k: float(p[k]) for k in choices}
    if not all(math.isfinite(x) and x >= 0 for x in result.values()):
        raise ValueError("Nonfinite/negative probability")
    z = sum(result.values())
    if not 0.98 <= z <= 1.02:
        raise ValueError(f"Probability sum {z} invalid")
    return {k: x / z for k, x in result.items()}


def record_sha256(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def classify_failure(exc):
    """overlong only via the explicit OverlongInput type (adapters convert their model's own signal). No text matching."""
    if isinstance(exc, OverlongInput):
        return "overlong"
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return "oom"
    return "error"


def select_subset(examples, max_questions, repeats):
    """Pilot subset: the first `max_questions` questions of the frozen order and repeats below `repeats`."""
    order, seen = [], set()
    for e in examples:
        q = (e["patient_id"], e["question_idx"])
        if q not in seen:
            seen.add(q)
            order.append(q)
    keep = set(order if max_questions is None else order[:max_questions])
    return [e for e in examples if (e["patient_id"], e["question_idx"]) in keep and (repeats is None or e["repeat"] < repeats)]


def load_frozen(data_path, meta_path, expected_revision=None, expected_dataset_sha=None):
    data_path, meta_path = Path(data_path), Path(meta_path)
    if not (meta_path.exists() and data_path.exists()):
        return None, None
    meta = json.loads(meta_path.read_text())
    payload = gzip.decompress(data_path.read_bytes())
    if hashlib.sha256(payload).hexdigest() != meta["examples_sha256"]:
        raise ValueError(f"Frozen example file {data_path} does not match its recorded SHA-256.")
    if expected_revision is not None and meta.get("longhealth_revision") != expected_revision:
        raise ValueError("Frozen examples were built from a different LongHealth revision.")
    if expected_dataset_sha is not None and meta.get("dataset_sha256") != expected_dataset_sha:
        raise ValueError("Frozen examples were built from a different dataset file.")
    examples = [json.loads(line) for line in payload.decode("utf8").splitlines() if line.strip()]
    return examples, meta


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------------------------
# Crash-safe JSONL
# ----------------------------------------------------------------------------------------------------------------------
def read_jsonl_readonly(path):
    """Rows of a JSONL file without modifying it. A final line without a newline (a write cut short) is ignored.
    Returns (rows, dropped_bytes). Invalid JSON in a complete line raises ValueError (real corruption)."""
    data = Path(path).read_bytes()
    dropped = 0
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        dropped = len(data) - cut
        data = data[:cut]
    rows = []
    for number, line in enumerate(data.decode("utf8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {number} is not valid JSON ({exc.msg}); the file is corrupted") from exc
    return rows, dropped


def read_jsonl_recovering(path):
    """Like read_jsonl_readonly, but truncates an unterminated final line away. Call only while holding the run lock."""
    rows, dropped = read_jsonl_readonly(path)
    if dropped:
        with open(path, "r+b") as handle:
            handle.truncate(handle.seek(0, os.SEEK_END) - dropped)
        log(f"WARNING: dropped {dropped} bytes of an unterminated last line in {path}")
    return rows


class JsonlAppender:
    """Append one JSON row per write, flushed and fsynced so a kill loses at most the row being written."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.handle = open(self.path, "a", encoding="utf8")
        return self

    def write(self, row):
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def __exit__(self, *exc):
        self.handle.close()


class RunLock:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise RunAborted(f"Run directory is locked by another process: {self.path.parent}")
        return self

    def release(self):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


# ----------------------------------------------------------------------------------------------------------------------
# Run configuration / manifest
# ----------------------------------------------------------------------------------------------------------------------
def build_config_record(spec, job, frozen_meta, adapter):
    cfg, rev = spec["config"], spec["revisions"]
    info = MODELS_INFO[spec["model"]]
    return {
        "schema_version": 4, "protocol": "longhealth_decision_adapted_native", "setting": "zero_shot",
        "model": spec["model"], "task": job["task"], "max_tokens": cfg["max_tokens"],
        "max_questions": job["max_questions"], "repeats": job["repeats"], "seed": cfg["seed"],
        "longhealth_revision": rev["longhealth"], "dataset_sha256": rev["dataset_sha256"],
        "examples_sha256": frozen_meta["examples_sha256"], "state_token_budget": cfg["state_token_budget"],
        "context_policy": "frozen_shared_examples_v4", "checkpoint": info["repo"],
        "quantization": getattr(adapter, "quantization", "native"),
        "checkpoint_revision": rev.get("checkpoint"), "runtime_revision": rev.get("runtime"),
    }


def run_id_for(config_record):
    return hashlib.sha256(json.dumps(config_record, sort_keys=True).encode()).hexdigest()[:12]


def package_snapshot():
    versions = {}
    for dist in metadata.distributions():
        try:
            versions[dist.metadata["Name"].lower()] = dist.version
        except Exception:
            continue
    freeze = "\n".join(f"{k}=={v}" for k, v in sorted(versions.items()))
    wanted = ("torch", "transformers", "accelerate", "safetensors", "huggingface-hub", "tokenizers", "bitsandbytes",
              "flash-linear-attention", "numpy", "pandas")
    return {k: versions.get(k) for k in wanted}, hashlib.sha256(freeze.encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Run loop
# ----------------------------------------------------------------------------------------------------------------------
def run_job(adapter, job, examples, config_record, cfg, runs_root, stop_flag, gates=None, gpu_name="unknown"):
    """Score `examples` with `adapter`, appending one row per request. Returns a result dict; raises RunAborted on a guard."""
    run_id = run_id_for(config_record)
    run_dir = Path(runs_root) / f"{config_record['model']}_{config_record['task']}_{run_id}"
    lock = RunLock(run_dir / ".lock").acquire()
    try:
        results_path, manifest_path = run_dir / "predictions.jsonl", run_dir / "manifest.json"
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            if prior["config"] != config_record:
                raise RunAborted(f"Run ID collision or changed configuration in {run_dir}")
        else:
            packages, freeze_sha = package_snapshot()
            write_json_atomic(manifest_path, {"run_id": run_id, "config": config_record,
                                              "created_utc": datetime.now(timezone.utc).isoformat(),
                                              "expected_requests": len(examples), "gpu_name": gpu_name,
                                              "packages": packages, "pip_freeze_sha256": freeze_sha,
                                              "python": platform.python_version()})
        if gates is not None:
            write_json_atomic(run_dir / "gates.json", {"gates": gates, "peak_vram_gb": None})
        ok_rows, last_row, attempts = {}, {}, collections.Counter()
        if results_path.exists():
            for old in read_jsonl_recovering(results_path):
                key = old["example_id"]
                if old.get("status") in ("oom", "error"):
                    attempts[key] += 1
                if old.get("status") == "ok":
                    if key in ok_rows:
                        raise RunAborted(f"Duplicate successful prediction: {key}")
                    ok_rows[key] = old
                last_row[key] = old
        shas = {e["example_id"]: record_sha256(e["record"]) for e in examples}

        def needs_request(ex):
            key = ex["example_id"]
            prior_row = last_row.get(key)
            if prior_row is not None and prior_row.get("input_sha256") != shas[key]:
                raise RunAborted(f"Input changed on resumed run: {key}")
            if key in ok_rows:
                return False
            if prior_row is not None and prior_row["status"] == "overlong":
                return False
            if prior_row is not None and not cfg["retry_failed"]:
                return False
            if attempts[key] >= cfg["max_item_attempts"]:
                return False   # deterministic failures (an item too big for the GPU) are not retried forever
            return True

        todo = [ex for ex in examples if needs_request(ex)]
        log(f"{run_dir.name}: {len(ok_rows)} already successful; {len(todo)} requests to run.")
        consecutive, processed, bad = 0, 0, collections.Counter()
        started, latencies, outcome = time.perf_counter(), [], "completed"
        with JsonlAppender(results_path) as out:
            for ex in todo:
                if stop_flag.is_set():
                    outcome = "interrupted"
                    break
                key = ex["example_id"]
                base = {"schema_version": 4, "run_id": run_id, "example_id": key, "patient_id": ex["patient_id"],
                        "question_idx": ex["question_idx"], "repeat": ex["repeat"], "task": ex["task"],
                        "model": config_record["model"], "setting": "zero_shot", "input_sha256": shas[key],
                        "gold": ex["gold"], "gold_set": ex["gold_set"], "answerable": ex["answerable"],
                        "answer_position": ex["answer_position"], "position_bucket": ex["position_bucket"],
                        "state_chars": len(ex["record"]["state"])}
                t0 = time.perf_counter()
                failure = None
                try:
                    choices = list(ex["record"]["questions"]["longhealth"]["criteria"])
                    p = normalize_probabilities(adapter.predict(ex["record"]), choices)
                    prediction = max(p, key=p.get)
                    row = {**base, "prediction": prediction, "probabilities": p, "correct": prediction in ex["gold_set"],
                           "latency_s": time.perf_counter() - t0, "status": "ok"}
                    consecutive = 0
                except Exception as exc:   # KeyboardInterrupt is not an Exception: signals only set stop_flag anyway
                    failure = classify_failure(exc)
                    row = {**base, "status": failure, "error": repr(exc)[:2000]}
                # Cleanup runs OUTSIDE the except block: inside it the traceback still references the model's tensors.
                if failure == "oom":
                    adapter.cleanup_after_failure()
                    if not ok_rows:   # an OOM before any success means the setup does not fit; later it is that item's problem
                        consecutive += 1
                elif failure == "error":
                    consecutive += 1
                elif failure == "overlong":
                    consecutive = 0
                out.write(row)
                last_row[key] = row
                if row["status"] == "ok":
                    ok_rows[key] = row
                    latencies.append(row["latency_s"])
                processed += 1
                bad[row["status"]] += 1
                if processed % 50 == 0:
                    rate = (time.perf_counter() - started) / processed
                    log(f"{run_dir.name}: {processed}/{len(todo)} done, median latency "
                        f"{statistics.median(latencies) if latencies else float('nan'):.2f}s, ETA {rate * (len(todo) - processed) / 60:.1f} min")
                if processed >= 20:
                    if bad["overlong"] / processed > cfg["max_overlong_fraction"]:
                        raise RunAborted(f"Stopped: {bad['overlong']} of {processed} requests were 'overlong'. Frozen inputs should fit "
                                         f"every model, so check the error text and the token counting. Last: {row.get('error')}")
                    if (bad["oom"] + bad["error"]) / processed > cfg["max_failure_fraction"]:
                        raise RunAborted(f"Stopped: {bad['oom']} oom + {bad['error']} error of {processed} requests. "
                                         f"Last at {key}: {row.get('error')}")
                if consecutive >= cfg["max_consecutive_errors"]:
                    raise RunAborted(f"Stopped after {consecutive} consecutive failures; last at {key}: {row.get('error')}. "
                                     f"Resuming retries oom/error rows up to max_item_attempts={cfg['max_item_attempts']} times each.")
        counts = collections.Counter(
            "ok" if e["example_id"] in ok_rows else last_row.get(e["example_id"], {}).get("status", "not_run") for e in examples)
        if gates is not None:
            write_json_atomic(run_dir / "gates.json", {"gates": gates, "peak_vram_gb": _peak(adapter)})
        return {"run_id": run_id, "run_dir": str(run_dir), "task": job["task"], "stage": job["stage"],
                "expected": len(examples), "counts": dict(counts), "outcome": outcome,
                "median_latency_s": statistics.median(latencies) if latencies else None}
    finally:
        lock.release()


def _peak(adapter):
    try:
        return adapter.peak_vram_gb()
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------------------------------------------------
_KNOWN = [  # (state, instruction, gold, distractors, gold_index). Synthetic: no benchmark content.
    ("The patient was prescribed metformin.", "Which drug was prescribed?", "metformin",
     ["lisinopril", "atorvastatin", "warfarin", "omeprazole"], 2),
    ("The tumor is located in the left lung.", "Where is the tumor?", "left lung",
     ["right lung", "liver", "pancreas", "left kidney"], 0),
    ("Her blood type is AB negative.", "What is her blood type?", "AB negative",
     ["O positive", "A positive", "B negative", "O negative"], 4),
    ("He was admitted on Tuesday.", "On which day was he admitted?", "Tuesday",
     ["Monday", "Wednesday", "Thursday", "Friday"], 1),
    ("The biopsy result was benign.", "What was the biopsy result?", "benign",
     ["malignant", "inconclusive", "necrotic", "hemorrhagic"], 3),
    ("The patient is allergic to penicillin.", "Which allergy does the patient have?", "penicillin",
     ["sulfa", "latex", "peanuts", "aspirin"], 2),
    ("The patient takes insulin.", "Which medication does the patient take?", "insulin",
     ["aspirin", "warfarin", "digoxin", "furosemide", "prednisone"], 5),
]


def known_answer_records():
    out = []
    for state, instruction, gold, distractors, gold_index in _KNOWN:
        options = list(distractors)
        options.insert(gold_index, gold)
        letters = "ABCDEF"[:len(options)]
        out.append(({"state": state, "questions": {"longhealth": {"type": "choice", "instructions": instruction,
                                                                      "criteria": dict(zip(letters, options))}}},
                    letters[gold_index]))
    return out


def gate_known_answer(adapter, min_correct=5):
    """Wrong letter mapping would give plausible but wrong scores; this catches it with obvious questions."""
    per_record, correct, first_p = [], 0, None
    records = known_answer_records()
    for record, gold in records:
        choices = list(record["questions"]["longhealth"]["criteria"])
        p = normalize_probabilities(adapter.predict(record), choices)
        first_p = p if first_p is None else first_p
        prediction = max(p, key=p.get)
        correct += prediction == gold
        per_record.append({"gold": gold, "prediction": prediction, "p_gold": round(p[gold], 4),
                           "probabilities": {k: round(v, 4) for k, v in p.items()}})
    first = records[0][0]
    again = normalize_probabilities(adapter.predict(first), list(first["questions"]["longhealth"]["criteria"]))
    deterministic = all(abs(first_p[k] - again[k]) <= 1e-3 for k in first_p)
    result = {"passed": correct >= min_correct, "n_correct": correct, "n": len(records), "min_correct": min_correct,
              "deterministic": deterministic, "per_record": per_record}
    if not deterministic:
        log("WARNING: repeating one known-answer record changed its probabilities by more than 1e-3.")
    if not result["passed"]:
        raise GateFailed(f"known-answer gate: {correct}/{len(records)} correct (need {min_correct}). Per record: {json.dumps(per_record)}")
    return result


def gate_probe(adapter, examples, allow_probe_oom=False, n=3):
    """Run the n longest frozen inputs first (untimed, unrecorded) so a memory problem shows up in minutes, not hours."""
    longest = sorted(examples, key=lambda e: len(e["record"]["state"]), reverse=True)[:n]
    reset = getattr(adapter, "reset_peak_vram", None)
    if reset:
        reset()
    outcomes = []
    for ex in longest:
        status, detail = "ok", ""
        try:
            choices = list(ex["record"]["questions"]["longhealth"]["criteria"])
            normalize_probabilities(adapter.predict(ex["record"]), choices)
        except Exception as exc:
            status, detail = classify_failure(exc), repr(exc)[:500]
        outcomes.append(status)
        if status == "oom":
            adapter.cleanup_after_failure()   # outside the except block, so the traceback no longer pins tensors
        elif status != "ok":
            raise GateFailed(f"probe gate: longest input failed with {status}: {detail}")
    result = {"passed": True, "outcomes": outcomes, "state_chars": [len(e["record"]["state"]) for e in longest],
              "peak_vram_gb": _peak(adapter), "allow_probe_oom": allow_probe_oom}
    if "oom" in outcomes and not allow_probe_oom:
        result["passed"] = False
        raise GateFailed(f"probe_oom: the longest inputs ran out of memory ({outcomes}); peak {result['peak_vram_gb']} GB. "
                         "Use --allow-probe-oom to accept partial coverage.")
    return result


def gate_pilot(run_result, runs_root, remaining_requests):
    run_dir = Path(run_result["run_dir"])
    rows = [r for r in read_jsonl_readonly(run_dir / "predictions.jsonl")[0] if r.get("status") == "ok"]
    by_sha = {}
    for r in rows:
        by_sha.setdefault(r["input_sha256"], r["probabilities"])
    vectors = list(by_sha.values())
    all_ok = run_result["counts"].get("ok", 0) == run_result["expected"]
    distinct = False
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if sum(abs(vectors[i][k] - vectors[j][k]) for k in vectors[i]) >= 1e-3:
                distinct = True
    near_uniform = bool(rows) and all(max(abs(v - 1 / len(r["probabilities"])) for v in r["probabilities"].values()) < 1e-3 for r in rows)
    median = run_result.get("median_latency_s")
    projected_min = None if median is None else median * remaining_requests / 60
    result = {"passed": bool(all_ok and distinct and not near_uniform), "all_ok": all_ok, "distinct_outputs": distinct,
              "near_uniform": near_uniform, "median_latency_s": median, "remaining_requests": remaining_requests,
              "projected_minutes": projected_min}
    log(f"pilot gate: {result}")
    if not result["passed"]:
        raise GateFailed(f"pilot gate failed: {result}")
    return result


# ----------------------------------------------------------------------------------------------------------------------
# Worker entry point
# ----------------------------------------------------------------------------------------------------------------------
def worker_main(AdapterCls, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    cfg = spec["config"]
    stop_flag = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_flag.set())   # the current request finishes; then the loop stops cleanly
    status = {"model": spec["model"], "outcome": "crashed", "gates": {}, "jobs": [], "message": ""}
    code = EXIT_CRASH
    try:
        adapter = AdapterCls(spec)
        adapter.check_environment()
        adapter.load()
        gpu_name = getattr(adapter, "gpu_name", lambda: "unknown")()
        gates = status["gates"]
        log("gate A: known-answer check")
        gates["known_answer"] = gate_known_answer(adapter)
        frozen, selected = {}, []
        for job in spec["jobs"]:
            task = job["task"]
            if task not in frozen:
                paths = spec["frozen"][task]
                frozen[task] = load_frozen(paths["data"], paths["meta"], spec["revisions"]["longhealth"], spec["revisions"]["dataset_sha256"])
                if frozen[task][0] is None:
                    raise EnvironmentMismatch(f"Frozen example set missing for {task}")
            selected.append(select_subset(frozen[task][0], job["max_questions"], job["repeats"]))
        log("gate B: longest-input probe")
        gates["probe"] = gate_probe(adapter, [e for sub in selected for e in sub], cfg.get("allow_probe_oom", False))
        interrupted = False
        for index, job in enumerate(spec["jobs"]):
            examples = selected[index]
            record = build_config_record(spec, job, frozen[job["task"]][1], adapter)
            result = run_job(adapter, job, examples, record, cfg, spec["runs_root"], stop_flag, gates=gates, gpu_name=gpu_name)
            status["jobs"].append(result)
            log(f"{job['task']} {job['stage']}: {result['counts']} ({result['outcome']})")
            if result["outcome"] == "interrupted":
                interrupted = True
                break
            if job["stage"] == "pilot":
                remaining = sum(len(selected[j]) for j in range(index + 1, len(spec["jobs"])) if spec["jobs"][j]["stage"] != "pilot")
                gates["pilot"] = gate_pilot(result, spec["runs_root"], remaining)
                write_json_atomic(Path(result["run_dir"]) / "gates.json", {"gates": gates, "peak_vram_gb": _peak(adapter)})
        status["outcome"], code = ("interrupted", EXIT_INTERRUPT) if interrupted else ("completed", EXIT_OK)
    except EnvironmentMismatch as exc:
        status.update(outcome="aborted", message=f"environment: {exc}")
        code = EXIT_ENV
    except GateFailed as exc:
        status.update(outcome="gate_failed", message=str(exc))
        code = EXIT_GATE
    except RunAborted as exc:
        status.update(outcome="aborted", message=str(exc))
        code = EXIT_ABORT
    except Exception as exc:
        traceback.print_exc()
        status.update(outcome="crashed", message=repr(exc)[:2000])
        code = EXIT_CRASH
    finally:
        try:
            write_json_atomic(spec["status_path"], status)
        except Exception:
            traceback.print_exc()
    if status["message"]:
        log(f"{status['outcome']}: {status['message']}")
    sys.exit(code)
