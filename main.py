#!/usr/bin/env python3
"""LongHealth runner: orchestrator, frozen-set builder and comparison report.

Runs under the base interpreter (no torch). Model scoring happens in model_*.py workers, one per conda env, launched
one at a time and exchanging data with this file only through files on disk (see PLAN.md).

  python main.py lock | setup-envs | build-frozen | run | status | compare | all

Benchmark text is never printed or written to logs/reports by this program.
"""
import argparse
import copy
import fcntl
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core  # noqa: E402
from lh_core import MODELS_INFO, TASKS  # noqa: E402

LONGHEALTH_REPO = HERE.parent / "LongHealth"
LOCK_FILE = HERE / "revisions.lock.json"
MODELS = ("ELEPHANT", "CLEF", "DREX")
EVAL_TASKS = ("task1", "task23")
BOOTSTRAP_DRAWS = 1200
SEED = 2026                       # module global used by the ported example builder; set from --seed by each command
REFERENCE_TOKENIZER = "ELEPHANT"  # drives upstream document selection; the result is then checked against all three
create_prompt = None              # set by load_upstream(); only this file ever imports LongHealth's `utils`
sample_distractions = None


class Paths:
    def __init__(self, project_dir):
        self.project = Path(project_dir).expanduser().resolve()
        self.results = self.project / "results_v4"
        self.examples, self.runs, self.reports = (self.results / s for s in ("examples", "runs", "reports"))
        self.run_logs, self.setup_logs = self.project / "run_logs", self.project / "setup_logs"


def fail(message, code=1):
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(code)


def load_upstream(repo=LONGHEALTH_REPO):
    """Import LongHealth's create_prompt / sample_distractions (the only place upstream `utils` is imported)."""
    global create_prompt, sample_distractions
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import utils
    create_prompt, sample_distractions = utils.create_prompt, utils.sample_distractions


# ----------------------------------------------------------------------------------------------------------------------
# lock: pinned revisions
# ----------------------------------------------------------------------------------------------------------------------
def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def load_lock(path=None):
    path = Path(path or LOCK_FILE)
    if not path.exists():
        fail(f"{path.name} not found. Run `python main.py lock` first.")
    lock = json.loads(path.read_text())
    shas = [m["revision"] for m in lock["models"].values()] + [lock["models"]["DREX"]["runtime_revision"], lock["longhealth"]["revision"]]
    if not all(isinstance(s, str) and len(s) == 40 for s in shas):
        fail(f"{path.name} does not hold full 40-character revisions.")
    return lock


def cmd_lock(args):
    if LOCK_FILE.exists() and not args.update:
        fail(f"{LOCK_FILE.name} exists. Pass --update to overwrite it deliberately (this changes the run_id of future runs).")
    from huggingface_hub import HfApi
    api = HfApi()
    models = {}
    for name, info in MODELS_INFO.items():
        models[name] = {"repo": info["repo"], "revision": api.model_info(info["repo"]).sha}
    drex_url = "https://github.com/nace-ai/drex-decision-models.git"
    head = subprocess.check_output(["git", "ls-remote", drex_url, "HEAD"], text=True, timeout=60).split()[0]
    models["DREX"]["runtime_url"], models["DREX"]["runtime_revision"] = drex_url, head
    dataset = LONGHEALTH_REPO / "data" / "benchmark_v5.json"
    lock = {"created_utc": datetime.now(timezone.utc).isoformat(), "models": models,
            "longhealth": {"revision": git_head(LONGHEALTH_REPO), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()},
            "note": "Elephant's OneJev runtime is pinned by the checkpoint's own setup.sh; Clef's code ships inside its checkpoint."}
    LOCK_FILE.write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))


# ----------------------------------------------------------------------------------------------------------------------
# conda / environments
# ----------------------------------------------------------------------------------------------------------------------
def find_conda():
    candidates = [shutil.which("conda"), os.environ.get("CONDA_EXE")]
    candidates += [str(Path.home() / n / "bin" / "conda") for n in ("miniforge3", "mambaforge", "miniconda3", "anaconda3")]
    return next((c for c in candidates if c and Path(c).exists()), None)


def resolve_env_python(env_name):
    conda = find_conda()
    if conda is None:
        fail("conda was not found on PATH or in ~/miniforge3 etc.")
    envs = json.loads(subprocess.check_output([conda, "env", "list", "--json"], text=True))["envs"]
    for env in envs:
        if Path(env).name == env_name and (Path(env) / "bin" / "python").exists():
            return str(Path(env) / "bin" / "python")
    return None


def hf_token_note():
    token = Path.home() / ".cache" / "huggingface" / "token"
    print(f"Hugging Face token file: {'present' if token.exists() else 'MISSING (gated repos will fail)'}", flush=True)


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "MPLBACKEND")}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def cmd_setup_envs(args):
    if args.project_dir is None:
        fail("Pass --project-dir MODELS_DIR (where the checkpoints, git clones and setup logs go), e.g. main.py --project-dir /data/models setup-envs")
    paths, lock = Paths(args.project_dir), load_lock()
    hf_token_note()
    conda = find_conda()
    if conda is None:
        fail("conda not found")
    paths.setup_logs.mkdir(parents=True, exist_ok=True)
    hf_cache = None
    if args.hf_cache:   # a shared Hugging Face cache holding models--ORG--NAME/snapshots/...: verified and linked, not downloaded
        hf_cache = Path(args.hf_cache).expanduser().resolve()
        if not hf_cache.is_dir():
            fail(f"--hf-cache {hf_cache} is not a folder")
        found = sorted(d.name for d in hf_cache.glob("models--*"))
        print(f"Shared Hugging Face cache {hf_cache}: {len(found)} model folders", flush=True)
    for model in args.models:
        model = model.upper()
        script = HERE / "setup" / f"setup_{model.lower()}.sh"
        env = clean_env(CHECKPOINT_REVISION=lock["models"][model]["revision"], PYTHONUNBUFFERED="1", ENV_NAME=MODELS_INFO[model]["env"],
                        PATH=str(Path(conda).parent) + os.pathsep + os.environ.get("PATH", ""))
        if model == "DREX":
            env["RUNTIME_REVISION"] = lock["models"]["DREX"]["runtime_revision"]
        if os.environ.get("HF_HOME"):
            env["HF_HOME"] = os.environ["HF_HOME"]
        env.pop("HF_SHARED_CACHE", None)
        if hf_cache:
            env["HF_SHARED_CACHE"] = str(hf_cache)
        print(f"===== {model}: bash {script} {paths.project} =====", flush=True)
        rc = subprocess.run(["bash", str(script), str(paths.project)], env=env, stdin=subprocess.DEVNULL).returncode
        if rc != 0:
            fail(f"{model} setup stopped at its first failing step (exit {rc}); see {paths.setup_logs}. Later models were not run.")
        python = resolve_env_python(MODELS_INFO[model]["env"])
        freeze = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        (paths.setup_logs / f"freeze_{model.lower()}.txt").write_text(freeze)
        print(f"{model}: environment ready ({len(freeze.splitlines())} packages frozen to setup_logs/freeze_{model.lower()}.txt)", flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Frozen example sets (ported from the notebook; globals replaced by arguments)
# ----------------------------------------------------------------------------------------------------------------------
BEGIN = '--------------BEGIN DOCUMENTS--------------'
END = '--------------END DOCUMENTS--------------'
UNANSWERABLE_TEXT = 'Question cannot be answered with provided documents'
def documents_from_prompt(prompt):
    if BEGIN not in prompt or END not in prompt:
        raise ValueError('Official LongHealth prompt boundaries changed. Review upstream utils.py.')
    return prompt.split(BEGIN,1)[1].split(END,1)[0].strip()

def gold_letters(correct, criteria):
    """Every option letter whose text is the gold answer.

    Upstream scores by checking whether the gold TEXT appears in the response, so when two options carry the
    same text (2 of 400 questions in benchmark_v5) either letter is correct. Accuracy accepts any letter in
    this set and Brier/NLL treat the duplicates as one merged option.
    """
    value = str(correct).strip()
    if value.upper() in criteria: return [value.upper()]
    matches=[k for k,v in criteria.items() if str(v).strip().casefold()==value.casefold()]
    if matches: return matches
    raise ValueError(f'Cannot match gold label {correct!r} to answer options')

def answer_position(locations):
    """Mean midpoint of upstream answer spans ({doc: [{'start','end'}, ...]}); None when no answer document is present."""
    spans=[]
    for entries in locations.values():
        for loc in entries:
            a=float(loc['start']); b=float(loc['end'])
            if 0<=a<=1 and 0<=b<=1: spans.append((a+b)/2)
    return float(sum(spans)/len(spans)) if spans else None

def location_bucket(value):
    if value is None: return 'unanswerable'
    return 'early' if value<1/3 else ('middle' if value<2/3 else 'late')

def make_example_id(task, pid, qi, rep):
    return f'{task}|{pid}|{qi}|{rep}'

def create_examples(dataset, tokenizer, task, max_tokens, max_questions=None, repeats_override=None, seed=SEED):
    rng_state=random.getstate()
    random.seed(seed)  # upstream utilities use random.shuffle / random.sample
    try:
        n=(5 if task=='task1' else 10) if repeats_override is None else repeats_override
        assert isinstance(n,int) and n>=1
        out=[]; qcount=0
        for pid,patient in dataset.items():
            for qi,base in enumerate(patient['questions']):
                if max_questions is not None and qcount>=max_questions: return out
                qcount+=1
                q=copy.deepcopy(base)
                if task=='task23': q['answer_f']=UNANSWERABLE_TEXT
                labels='abcde' if task=='task1' else 'abcdef'
                criteria={ch.upper():str(q[f'answer_{ch}']) for ch in labels}
                answer_docs={did:patient['texts'][did] for did in q['answer_location']}
                others=[v for did,v in patient['texts'].items() if did not in answer_docs]
                for rep in range(n):
                    if task=='task1':
                        docs=answer_docs; distractors=others; gold_set=gold_letters(q['correct'],criteria)
                    else:
                        distractors=sample_distractions(pid,dataset,n=10)
                        docs={} if rep%2==0 else answer_docs
                        gold_set=['F'] if rep%2==0 else gold_letters(q['correct'],criteria)
                    gold=gold_set[0]
                    official_prompt,locations=create_prompt(docs,distractors,q,option_labels=labels,
                                                            max_len=max_tokens,tokenizer=tokenizer)
                    state=documents_from_prompt(official_prompt)
                    if not state: raise ValueError(f'Empty clinical documents: {pid}/{qi}/{rep}')
                    record={'state':state,'questions':{'longhealth':{
                        'type':'choice','instructions':str(q['question']),'criteria':criteria}}}
                    pos=answer_position(locations)
                    out.append({'example_id':make_example_id(task,pid,qi,rep),'patient_id':str(pid),
                                'question_idx':qi,'repeat':rep,'task':task,'gold':gold,'gold_set':gold_set,
                                'answerable':gold!='F','answer_position':pos,'position_bucket':location_bucket(pos),
                                'answer_locations':locations,'record':record})
        return out
    finally:
        random.setstate(rng_state)

class TokenizerJSON:
    """A model's own tokenizer.json, counting raw text tokens (no special tokens, no chat template)."""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.path=str(path); self._tok=Tokenizer.from_file(self.path)
        self.sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def encode(self, text, **kw): return self._tok.encode(text, add_special_tokens=False).ids
    def decode(self, ids, **kw): return self._tok.decode(list(ids), skip_special_tokens=False)
    def count_many(self, texts, chunk=64):
        # Chunked: encoding thousands of long states at once keeps every token's offsets in memory.
        texts=list(texts); counts=[]
        for i in range(0,len(texts),chunk):
            counts.extend(len(e.ids) for e in self._tok.encode_batch(texts[i:i+chunk], add_special_tokens=False))
        return counts

def count_tokens(tokenizer, texts):
    return tokenizer.count_many(texts) if hasattr(tokenizer,'count_many') else [len(tokenizer.encode(t)) for t in texts]

def build_frozen_examples(task, tokenizers, reference, budget, dataset=None, max_rounds=6):
    """Build with the reference tokenizer; shrink the build budget until every tokenizer fits within `budget`.

    The shrink is proportional (the hungriest tokenizer's token count scales with the reference's), so the
    documents end close to the budget instead of far below it.
    """
    dataset=benchmark if dataset is None else dataset
    build_budget=budget; rounds=[]
    for _ in range(max_rounds):
        examples=create_examples(dataset,tokenizers[reference],task,build_budget,None,None,SEED)
        states=[e['record']['state'] for e in examples]
        worst={m:max(count_tokens(tok,states)) for m,tok in tokenizers.items()}
        rounds.append({'build_budget':build_budget,'max_state_tokens':worst})
        hungriest=max(worst.values())
        if hungriest<=budget:
            return examples,{'build_budget':build_budget,'max_state_tokens':worst,'rounds':rounds}
        build_budget=int(build_budget*budget/hungriest)-32
        if build_budget<1024:
            break
    raise RuntimeError(f'Could not fit {task} states within {budget} tokens for every tokenizer: {rounds}')



def load_model_tokenizer(model, lock):
    from huggingface_hub import hf_hub_download
    return TokenizerJSON(hf_hub_download(MODELS_INFO[model]["repo"], "tokenizer.json", revision=lock["models"][model]["revision"]))


def frozen_paths(paths, task, rev, seed, budget):
    stem = f"{task}_seed{seed}_budget{budget}_rev{rev[:8]}"
    return paths.examples / f"{stem}.jsonl.gz", paths.examples / f"{stem}.meta.json"


def write_frozen(paths, task, examples, meta, rev, seed, budget):
    payload = "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in examples).encode("utf8")
    sha = hashlib.sha256(payload).hexdigest()
    data_path, meta_path = frozen_paths(paths, task, rev, seed, budget)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        prior = json.loads(meta_path.read_text())
        if prior["examples_sha256"] != sha:
            raise FileExistsError(f"A different frozen set already exists at {meta_path}. Runs may already depend on it; refusing to "
                                  "overwrite. Move it aside deliberately or change --seed/--state-token-budget.")
        print("Identical frozen set already present; kept:", data_path.name)
        return prior
    with open(data_path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(payload)
    meta = {**meta, "task": task, "examples_sha256": sha, "n_examples": len(examples),
            "n_questions": len({(e["patient_id"], e["question_idx"]) for e in examples}), "file": data_path.name}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def cmd_build_frozen(args):
    global SEED
    SEED = args.seed
    lock, paths = load_lock(), Paths(args.project_dir)
    load_upstream()
    dataset_path = LONGHEALTH_REPO / "data" / "benchmark_v5.json"
    if git_head(LONGHEALTH_REPO) != lock["longhealth"]["revision"] or hashlib.sha256(dataset_path.read_bytes()).hexdigest() != lock["longhealth"]["dataset_sha256"]:
        fail("The LongHealth clone or dataset differs from revisions.lock.json.")
    benchmark = json.loads(dataset_path.read_text())
    tokenizers = {m: load_model_tokenizer(m, lock) for m in MODELS}
    expected_questions = sum(len(p["questions"]) for p in benchmark.values())
    for task in TASKS:
        t0 = time.perf_counter()
        frozen, info = build_frozen_examples(task, tokenizers, REFERENCE_TOKENIZER, args.state_token_budget, dataset=benchmark)
        n_questions = len({(e["patient_id"], e["question_idx"]) for e in frozen})
        assert n_questions == expected_questions == 400, (n_questions, expected_questions)
        assert len(frozen) == (2000 if task == "task1" else 4000), len(frozen)
        assert max(info["max_state_tokens"].values()) <= args.state_token_budget
        meta = write_frozen(paths, task, frozen, {
            "longhealth_revision": lock["longhealth"]["revision"], "dataset_sha256": lock["longhealth"]["dataset_sha256"], "seed": SEED,
            "state_token_budget": args.state_token_budget, "reference_tokenizer": REFERENCE_TOKENIZER,
            "tokenizer_repos": {m: MODELS_INFO[m]["repo"] for m in MODELS},
            "tokenizer_revisions": {m: lock["models"][m]["revision"] for m in MODELS},
            "tokenizer_sha256": {m: t.sha256 for m, t in tokenizers.items()}, **info,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            lock["longhealth"]["revision"], SEED, args.state_token_budget)
        print(f"{task}: {meta['n_examples']} examples, {meta['n_questions']} questions, sha256 {meta['examples_sha256'][:12]}, "
              f"build budget {meta['build_budget']}, max state tokens {meta['max_state_tokens']} ({time.perf_counter() - t0:.0f}s)", flush=True)


def frozen_files(paths, lock, seed, budget):
    out = {}
    for task in TASKS:
        data, meta = frozen_paths(paths, task, lock["longhealth"]["revision"], seed, budget)
        if not (data.exists() and meta.exists()):
            return None
        out[task] = {"data": str(data), "meta": str(meta)}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# run: launch one worker per model, sequentially
# ----------------------------------------------------------------------------------------------------------------------
def gpu_free_gb(gpu):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)], text=True)
    return float(out.strip().splitlines()[0]) / 1024


def build_jobs(args, tasks):
    jobs = []
    if not args.no_pilot:
        jobs.append({"task": tasks[0], "stage": "pilot", "max_questions": 2, "repeats": 2})
    if not args.pilot_only:
        stage = "full" if args.max_questions is None and args.repeats is None else "custom"
        for task in tasks:
            if task == "task23" and args.repeats is not None and args.repeats < 2:
                fail("Task 2/3 repeats alternate unanswerable/answerable; use --repeats >= 2.")
            jobs.append({"task": task, "stage": stage, "max_questions": args.max_questions, "repeats": args.repeats})
    return jobs


class Child:
    proc = None
    stop = False


def _forward(signum, _frame):
    Child.stop = True
    if Child.proc is not None and Child.proc.poll() is None:
        Child.proc.send_signal(signal.SIGTERM)


def cmd_run(args):
    paths, lock = Paths(args.project_dir), load_lock(args.lock_file)
    models = list(MODELS) if args.models == ["all"] else [m.upper() for m in args.models]
    tasks = list(args.tasks)
    frozen = frozen_files(paths, lock, args.seed, args.state_token_budget)
    if frozen is None:
        fail("Frozen example sets not found. Run `python main.py build-frozen` first.")
    paths.runs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log_dir = paths.run_logs / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    hf_token_note()
    gpu_lock = open(paths.project / f".gpu{args.gpu}.lock", "a+")
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fail(f"Another run is already using GPU {args.gpu} (lock {gpu_lock.name}).")
    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)
    summary = []
    for model in models:
        if Child.stop:
            break
        info = MODELS_INFO[model]
        override = args.worker_override.get(model)
        int8 = bool(args.drex_int8 and model == "DREX")
        minimum = info.get("min_vram_gb_int8") if int8 else info["min_vram_gb"]
        started = time.perf_counter()
        python = sys.executable if override else resolve_env_python(info["env"])
        if python is None:
            summary.append((model, "no_env", f"conda env {info['env']} missing; run setup-envs --models {model.lower()}", 0))
            print(f"{model}: environment {info['env']} not found. Run `python main.py setup-envs --models {model.lower()}`.", flush=True)
            continue
        if not override:
            free = gpu_free_gb(args.gpu)
            if free < minimum:
                summary.append((model, "preflight", f"only {free:.1f} GB free on GPU {args.gpu}; needs {minimum}", 0))
                print(f"{model}: only {free:.1f} GB free on GPU {args.gpu} (needs {minimum}); skipped.", flush=True)
                continue
        spec = {"model": model, "device": "cuda:0", "project_dir": str(paths.project), "runs_root": str(paths.runs),
                "status_path": str(log_dir / f"{model}.status.json"),
                "config": {"seed": args.seed, "max_tokens": args.max_tokens, "state_token_budget": args.state_token_budget,
                           "max_consecutive_errors": 5, "max_item_attempts": 2, "max_overlong_fraction": 0.2,
                           "max_failure_fraction": 0.5, "retry_failed": True, "drex_int8": int8,
                           "allow_probe_oom": bool(args.allow_probe_oom)},
                "revisions": {"checkpoint": lock["models"][model]["revision"], "runtime": lock["models"][model].get("runtime_revision"),
                              "longhealth": lock["longhealth"]["revision"], "dataset_sha256": lock["longhealth"]["dataset_sha256"]},
                "frozen": frozen, "jobs": build_jobs(args, tasks)}
        spec_path = log_dir / f"{model}.spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        worker_file = Path(override) if override else HERE / info["file"]
        env = clean_env(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE="1",
                        PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
        print(f"\n######## {model} ({time.strftime('%H:%M:%S')}) jobs={[(j['task'], j['stage']) for j in spec['jobs']]} ########", flush=True)
        with open(log_dir / f"{model}.log", "w") as logfile:
            Child.proc = subprocess.Popen([python, str(worker_file), "--spec", str(spec_path)], cwd=str(HERE), env=env,
                                          start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in Child.proc.stdout:
                print(line, end="", flush=True)
                logfile.write(line)
                logfile.flush()
            code = Child.proc.wait()
        Child.proc = None
        status_file = Path(spec["status_path"])
        status = json.loads(status_file.read_text()) if status_file.exists() else {"outcome": "crashed", "message": f"worker exited {code} without a status file"}
        summary.append((model, status["outcome"], (status.get("message") or "")[:120], (time.perf_counter() - started) / 60))
        if Child.stop or code == lh_core.EXIT_INTERRUPT:
            Child.stop = True
    print("\n===== run summary =====")
    print(pd.DataFrame(summary, columns=["model", "outcome", "message", "minutes"]).round(1).to_string(index=False), flush=True)
    return 130 if Child.stop else (0 if all(s[1] == "completed" for s in summary) else 3)


def cmd_status(args):
    paths = Paths(args.project_dir)
    rows = []
    for folder in sorted(paths.runs.glob("*")) if paths.runs.exists() else []:
        try:
            manifest = json.loads((folder / "manifest.json").read_text())
            cfg = manifest["config"]
            preds = lh_core.read_jsonl_readonly(folder / "predictions.jsonl")[0] if (folder / "predictions.jsonl").exists() else []
        except Exception as exc:
            rows.append({"folder": folder.name, "model": "?", "task": "?", "stage": f"INVALID: {exc}"[:60]})
            continue
        last = {r["example_id"]: r["status"] for r in preds}
        counts = pd.Series(list(last.values())).value_counts().to_dict()
        n_ok = counts.get("ok", 0)
        rows.append({"model": cfg["model"], "task": cfg["task"], "setting": cfg["setting"],
                     "stage": "full" if cfg["max_questions"] is None and cfg["repeats"] is None else "pilot/custom",
                     "expected": manifest["expected_requests"], "ok": n_ok, "oom": counts.get("oom", 0),
                     "error": counts.get("error", 0), "overlong": counts.get("overlong", 0),
                     "complete": n_ok == manifest["expected_requests"], "run_id": manifest["run_id"]})
    print(pd.DataFrame(rows).to_string(index=False) if rows else "No runs yet.")
# ---- Ported verbatim from the notebook (cell "probability_metrics"); only the globals they use are defined above. ----
def probability_metrics(row):
    """Brier and NLL with duplicate gold options merged; identical to the usual definitions when the gold is unique."""
    vals={c:float(v) for c,v in row['probabilities'].items()}
    gold_set=list(row.get('gold_set') or [row['gold']])
    if not all(np.isfinite(x) and x>=0 for x in vals.values()):
        raise ValueError('Negative, NaN or infinite probability')
    if abs(sum(vals.values())-1)>1e-3:
        raise ValueError('Probabilities must sum to one')
    if row['gold'] not in gold_set or any(g not in vals for g in gold_set) or row['prediction'] not in vals:
        raise ValueError('Gold or predicted choice missing')
    if row['prediction']!=max(vals,key=vals.get):
        raise ValueError('Prediction must match argmax of saved probabilities')
    p_gold=sum(vals[g] for g in gold_set)
    return {'brier':sum(v**2 for c,v in vals.items() if c not in gold_set)+(p_gold-1.0)**2,
            'nll':-math.log(max(p_gold,1e-12)),
            'confidence':vals[row['prediction']],
            'correct':int(row['prediction'] in gold_set)}

def ece_from_frame(frame,bins=10):
    """Weighted equal-width expected calibration error of top-choice confidence."""
    if frame.empty: return float('nan')
    confidence=frame['confidence'].to_numpy(dtype=float)
    correctness=frame['correct'].to_numpy(dtype=float)
    indices=np.minimum((confidence*bins).astype(int),bins-1)
    return float(sum((indices==j).mean()*abs(correctness[indices==j].mean()-confidence[indices==j].mean())
              for j in range(bins) if np.any(indices==j)))

def cluster_ci(frame,field='correct',n_boot=1200,seed=2026):
    """Resample whole patients (not correlated repeated requests) with replacement."""
    if frame.empty or frame.patient_id.nunique()<2: return (float('nan'),float('nan'))
    patient_groups=[grp[field].to_numpy(dtype=float) for _,grp in frame.groupby('patient_id',sort=True)]
    rng=np.random.default_rng(seed); p=len(patient_groups)
    means=[np.concatenate([patient_groups[i] for i in rng.integers(0,p,size=p)]).mean() for _ in range(n_boot)]
    return tuple(float(x) for x in np.quantile(means,[.025,.975]))

REQUIRED_OK_FIELDS=('example_id','input_sha256','probabilities','gold','prediction','latency_s','model','setting',
                    'answerable','position_bucket','patient_id','question_idx','repeat','task')

def run_inventory(runs):
    return pd.DataFrame([{'model':r['manifest']['config']['model'],'task':r['manifest']['config']['task'],
                          'setting':r['manifest']['config']['setting'],'run_id':r['run_id'],
                          'successful':r['n_success'],'expected':r['n_expected'],'overlong':r['n_overlong'],
                          'failed':r['n_failed'],'complete':r['is_complete'],'full_protocol':r['is_full_protocol'],
                          'examples_sha256':(r['manifest']['config'].get('examples_sha256') or '')[:12],
                          'folder':str(r['path'])} for r in runs])

def select_runs(runs,overrides=None):
    """Prefer full-protocol, then complete, then most successful, then newest. Never choose by accuracy."""
    overrides=overrides or {}
    groups={}
    for r in runs:
        cfg=r['manifest']['config']
        groups.setdefault((cfg['model'],cfg['task'],cfg['setting']),[]).append(r)
    selected={}
    for key,items in groups.items():
        if key in overrides:
            candidates=[r for r in items if r['run_id']==overrides[key]]
            if len(candidates)!=1: raise ValueError(f'Invalid run override: {key}={overrides[key]}')
            selected[key]=candidates[0]
        else:
            selected[key]=sorted(items,key=lambda r:(r['is_full_protocol'],r['is_complete'],r['n_success'],r['timestamp']),
                                 reverse=True)[0]
    return selected

COMPARABLE_KEYS=('schema_version','protocol','task','max_tokens','max_questions','repeats','seed',
                 'longhealth_revision','dataset_sha256','examples_sha256','context_policy')

def comparable_configs(a,b):
    ca=a['manifest']['config'];cb=b['manifest']['config']
    return all(ca.get(k)==cb.get(k) for k in COMPARABLE_KEYS)

def summary_one(run,n_boot=1200):
    f=run['frame'];cfg=run['manifest']['config']
    if f.empty: return None
    ci=cluster_ci(f,n_boot=n_boot)
    return {'model':cfg['model'],'setting':cfg['setting'],'task':cfg['task'],'n':len(f),
            'expected_n':run['n_expected'],'coverage':len(f)/run['n_expected'],
            'overlong':run['n_overlong'],'failed':run['n_failed'],
            'patients':f['patient_id'].nunique(),'accuracy':float(f['correct'].mean()),
            'accuracy_ci_low':ci[0],'accuracy_ci_high':ci[1],
            'brier':float(f['brier'].mean()),'nll':float(f['nll'].mean()),
            'ece10':ece_from_frame(f),'median_latency_s':float(f['latency_s'].median()),
            'full_protocol':run['is_full_protocol'],'complete':run['is_complete'],'run_id':run['run_id']}

def common_item_accuracy(selected,setting='zero_shot'):
    """Accuracy of each model on the items that every available model answered successfully (identical inputs)."""
    rows=[]
    for task in EVAL_TASKS:
        runs={m:selected.get((m,task,setting)) for m in MODELS}
        runs={m:r for m,r in runs.items() if r is not None and not r['frame'].empty}
        if len(runs)<2: continue
        reference=next(iter(runs.values()))
        if not all(comparable_configs(reference,r) for r in runs.values()):
            print(f'{task}: common-item table skipped; selected runs use different frozen sets or settings.')
            continue
        frames={m:r['frame'].set_index('example_key') for m,r in runs.items()}
        common=set.intersection(*(set(f.index) for f in frames.values()))
        common=sorted(k for k in common if len({frames[m].at[k,'input_sha256'] for m in frames})==1
                      and len({frames[m].at[k,'gold'] for m in frames})==1)
        for m,f in frames.items():
            sub=f.loc[common].reset_index()
            ci=cluster_ci(sub,n_boot=BOOTSTRAP_DRAWS) if len(sub) else (float('nan'),)*2
            rows.append({'task':task,'model':m,'models':'+'.join(sorted(frames)),'n_common':len(common),
                         'accuracy_common':float(sub.correct.mean()) if len(sub) else float('nan'),
                         'ci_low':ci[0],'ci_high':ci[1],'n_own':len(f),'accuracy_own':float(f.correct.mean())})
    return pd.DataFrame(rows)

def task23_answerability(selected):
    rows=[]
    for (model,task,setting),r in selected.items():
        if task!='task23' or r['frame'].empty: continue
        for category,frame in r['frame'].groupby('answer_category'):
            ci=cluster_ci(frame,n_boot=BOOTSTRAP_DRAWS)
            rows.append({'model':model,'setting':setting,'category':category,'n':len(frame),
                         'accuracy':float(frame.correct.mean()),'ci_low':ci[0],'ci_high':ci[1],
                         'brier':float(frame.brier.mean()),'nll':float(frame.nll.mean()),'ece10':ece_from_frame(frame)})
    return pd.DataFrame(rows)

def position_results(selected):
    rows=[]
    for (model,task,setting),r in selected.items():
        f=r['frame']
        if f.empty: continue
        f=f[(f.answerable==True)&f.position_bucket.isin(['early','middle','late'])]
        for loc,frame in f.groupby('position_bucket'):
            rows.append({'model':model,'setting':setting,'task':task,'position':loc,
                         'n':len(frame),'accuracy':float(frame.correct.mean())})
    return pd.DataFrame(rows)

def repeat_consistency(selected):
    """Grouped by question AND answerability: Task 2/3 repeats alternate between gold F and the real answer."""
    rows=[]
    for (model,task,setting),r in selected.items():
        f=r['frame']
        if f.empty: continue
        for category,part in f.groupby('answer_category'):
            grouped=part.groupby(['patient_id','question_idx'])
            if not grouped.size().max()>1: continue
            rows.append({'model':model,'setting':setting,'task':task,'items':category,'questions':len(grouped),
                         'all_repetitions_correct':float(grouped.correct.min().mean()),
                         'predictions_change':float(grouped.prediction.nunique().gt(1).mean()),
                         'mean_question_accuracy':float(grouped.correct.mean().mean())})
    return pd.DataFrame(rows)

def pair_strict(a,b,n_boot=1200,seed=2026):
    """Compare only request keys with the SAME input SHA-256 and gold answer."""
    if not comparable_configs(a,b):
        return {'status':'incomparable_config','n_shared_keys':0,'n_exact_inputs':0}
    fa=a['frame'];fb=b['frame']
    if fa.empty or fb.empty: return {'status':'empty','n_shared_keys':0,'n_exact_inputs':0}
    A=fa.set_index('example_key');B=fb.set_index('example_key')
    shared=A.index.intersection(B.index)
    if not len(shared): return {'status':'no_shared_keys','n_shared_keys':0,'n_exact_inputs':0}
    left=A.loc[shared];right=B.loc[shared]
    valid=(left.input_sha256.to_numpy()==right.input_sha256.to_numpy()) & (left.gold.to_numpy()==right.gold.to_numpy())
    left=left.iloc[np.flatnonzero(valid)].copy();right=right.iloc[np.flatnonzero(valid)].copy()
    result={'status':'ok' if len(left) else 'different_native_input',
            'n_shared_keys':len(shared),'n_exact_inputs':len(left),'exact_match_fraction':len(left)/len(shared)}
    if not len(left): return result
    a_cor=left.correct.to_numpy(dtype=int);b_cor=right.correct.to_numpy(dtype=int)
    a_only=int(((a_cor==1)&(b_cor==0)).sum());b_only=int(((a_cor==0)&(b_cor==1)).sum())
    result.update({'model_a':a['manifest']['config']['model'],'model_b':b['manifest']['config']['model'],
                   'setting_a':a['manifest']['config']['setting'],'setting_b':b['manifest']['config']['setting'],
                   'task':a['manifest']['config']['task'],
                   'accuracy_a':float(a_cor.mean()),'accuracy_b':float(b_cor.mean()),
                   'delta_accuracy_a_minus_b':float(np.mean(a_cor-b_cor)),
                   'a_correct_b_wrong':a_only,'a_wrong_b_correct':b_only,'patients':left.patient_id.nunique()})
    # Repeats and questions are clustered within 20 patients: test one mean difference per patient.
    diff=left[['patient_id']].copy();diff['delta']=(a_cor-b_cor)
    by_patient=diff.groupby('patient_id',sort=True).delta.mean().to_numpy(dtype=float)
    result['delta_patient_macro']=float(by_patient.mean())
    result['cluster_permutation_p']=float('nan')
    result['delta_ci_low'],result['delta_ci_high']=(float('nan'),float('nan'))
    if len(by_patient)>=2:
        rng=np.random.default_rng(seed)
        # Patient-cluster sign-flip test: exact for <= 15 patients, else 19,999 Monte Carlo draws.
        n_patients=len(by_patient);observed=abs(by_patient.mean())
        if n_patients<=15:
            idx=np.arange(2**n_patients,dtype=np.uint32)[:,None]
            bits=(idx >> np.arange(n_patients,dtype=np.uint32)) & 1
            signs=2*bits.astype(np.int8)-1
            simulated=np.abs((signs*by_patient).mean(axis=1))
            pval=float(np.mean(simulated>=observed-1e-12))
        else:
            signs=rng.choice([-1,1],size=(19999,n_patients))
            simulated=np.abs((signs*by_patient).mean(axis=1))
            pval=float((1+np.sum(simulated>=observed-1e-12))/(len(simulated)+1))
        result['cluster_permutation_p']=pval
        resampled=rng.choice(by_patient,size=(n_boot,n_patients),replace=True).mean(axis=1)
        result['delta_ci_low'],result['delta_ci_high']=(float(np.quantile(resampled,.025)),float(np.quantile(resampled,.975)))
    return result

def paired_table(selected,bootstrap_draws=1200):
    results=[]
    for task in EVAL_TASKS:
        keys=[k for k in selected if k[1]==task]
        by_setting={s:[k for k in keys if k[2]==s] for s in {k[2] for k in keys}}
        for setting,group in by_setting.items():
            for ka,kb in itertools.combinations(sorted(group),2):
                score=pair_strict(selected[ka],selected[kb],n_boot=bootstrap_draws)
                score.update({'task':task,'comparison':f'{ka[0]} vs {kb[0]}','setting':setting,
                              'run_a':selected[ka]['run_id'],'run_b':selected[kb]['run_id']})
                results.append(score)
    out=pd.DataFrame(results)
    if out.empty: return out
    out['p_holm']=np.nan
    if 'cluster_permutation_p' not in out: return out
    # Holm correction per task/setting across the model pairs.
    for _,indices in out.groupby(['task','setting']).groups.items():
        valid=out.loc[list(indices),'cluster_permutation_p'].dropna()
        if valid.empty: continue
        sorted_ids=valid.sort_values().index.tolist()
        max_adjusted=0.0;m=len(sorted_ids)
        for j,idx in enumerate(sorted_ids):
            max_adjusted=max(max_adjusted,min(1.0,(m-j)*float(out.loc[idx,'cluster_permutation_p'])))
            out.loc[idx,'p_holm']=max_adjusted
    return out



REQUIRED_OK_FIELDS = ('example_id', 'input_sha256', 'probabilities', 'gold', 'prediction', 'latency_s', 'model', 'setting',
                      'answerable', 'position_bucket', 'patient_id', 'question_idx', 'repeat', 'task')


def read_run(run_folder):
    """Load one run. Raises ValueError for anything invalid (compare reports it instead of silently skipping)."""
    run_folder = Path(run_folder)
    manifest_file, predictions_file = run_folder / 'manifest.json', run_folder / 'predictions.jsonl'
    if not (manifest_file.exists() and predictions_file.exists()):
        raise ValueError('manifest.json or predictions.jsonl missing')
    m = json.loads(manifest_file.read_text(encoding='utf8'))
    if m.get('config', {}).get('schema_version') != 4:
        raise ValueError('not a v4 run')
    rows, dropped = lh_core.read_jsonl_readonly(predictions_file)   # an unterminated last line is ignored, not modified
    ok = [row for row in rows if row.get('status') == 'ok']
    keys = [r['example_id'] for r in ok]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate successful rows')
    for row in ok:
        missing = [f for f in REQUIRED_OK_FIELDS if f not in row]
        if missing:
            raise ValueError(f'missing fields {missing}')
        if row['model'] != m['config']['model'] or row['task'] != m['config']['task'] or row['setting'] != m['config']['setting']:
            raise ValueError('predictions disagree with the run manifest')
        probability_metrics(row)
    last_status = {r['example_id']: r['status'] for r in rows}
    unresolved = {k: s for k, s in last_status.items() if k not in set(keys)}
    frame = pd.DataFrame(ok)
    if not frame.empty:
        scores = frame.apply(probability_metrics, axis=1, result_type='expand')
        for column in scores.columns:
            frame[column] = scores[column].to_numpy()
        frame['example_key'] = frame['example_id']
        frame['latency_s'] = pd.to_numeric(frame['latency_s'], errors='coerce')
        frame['answer_category'] = frame['answerable'].map({True: 'answerable', False: 'unanswerable'})
    n_expected = int(m['expected_requests'])
    return {'run_id': m['run_id'], 'path': run_folder, 'manifest': m, 'rows': rows, 'frame': frame,
            'n_success': len(ok), 'n_expected': n_expected,
            'n_overlong': sum(s == 'overlong' for s in unresolved.values()),
            'n_failed': sum(s != 'overlong' for s in unresolved.values()),
            'is_complete': len(set(keys)) == n_expected,
            'is_full_protocol': (m['config']['max_questions'] is None and m['config']['repeats'] is None),
            'timestamp': predictions_file.stat().st_mtime, 'dropped_tail_bytes': dropped}


def discover_runs(results_root):
    """Returns (valid runs, invalid [(folder, reason)])."""
    valid, invalid = [], []
    folders = sorted((Path(results_root) / 'runs').glob('*')) if (Path(results_root) / 'runs').exists() else []
    for folder in folders:
        if not folder.is_dir():
            continue
        try:
            valid.append(read_run(folder))
        except Exception as exc:
            invalid.append((str(folder), f'{type(exc).__name__}: {exc}'))
    return valid, invalid


def compute_rankings(selected):
    rank_rows, by_task = [], {}
    for task in EVAL_TASKS:
        trio = [selected.get((model, task, 'zero_shot')) for model in MODELS]
        if any(r is None for r in trio):
            print(f'{task}: cannot rank; not all three models have a zero-shot run.'); continue
        if any(not r['is_full_protocol'] or not r['is_complete'] for r in trio):
            print(f'{task}: cannot rank; at least one run is a pilot or has unresolved items.'); continue
        if not all(comparable_configs(trio[0], r) for r in trio[1:]):
            print(f'{task}: cannot rank; model runs use different frozen sets or configurations.'); continue
        task_scores = {r['manifest']['config']['model']: float(r['frame'].correct.mean()) for r in trio}
        by_task[task] = task_scores
        for model, acc in task_scores.items():
            rank_rows.append({'task': task, 'model': model, 'accuracy': acc,
                              'rank': 1 + sum(other > acc for other in task_scores.values()), 'basis': 'full-protocol request accuracy'})
    if len(EVAL_TASKS) > 1 and all(t in by_task for t in EVAL_TASKS):   # a one-task "macro" would just repeat that task
        macro = {m: float(np.mean([by_task[t][m] for t in EVAL_TASKS])) for m in MODELS}
        for model, acc in macro.items():
            rank_rows.append({'task': 'exploratory_equal_task_macro', 'model': model, 'accuracy': acc,
                              'rank': 1 + sum(other > acc for other in macro.values()), 'basis': 'unofficial 50/50 task mean'})
    return pd.DataFrame(rank_rows)


def headline_table(selected, common):
    """One number per model and task. The basis rule is fixed in advance (PLAN.md 8.3):
    own accuracy only when all three runs are complete full-protocol runs on the same frozen set; otherwise accuracy on the
    items every available model answered, with each model's coverage shown next to it."""
    rows = []
    nan = float('nan')
    for task in EVAL_TASKS:
        trio = {m: selected.get((m, task, 'zero_shot')) for m in MODELS}
        all_complete = (all(r is not None and r['is_complete'] and r['is_full_protocol'] and not r['frame'].empty for r in trio.values())
                        and all(comparable_configs(trio[MODELS[0]], r) for r in trio.values()))
        for m in MODELS:
            r = trio[m]
            if r is None or r['frame'].empty:
                rows.append({'task': task, 'model': m, 'accuracy': nan, 'basis': 'missing', 'n': 0, 'coverage': 0.0, 'ci_low': nan, 'ci_high': nan})
                continue
            coverage = len(r['frame']) / r['n_expected']
            if all_complete:
                ci = cluster_ci(r['frame'], n_boot=BOOTSTRAP_DRAWS)
                rows.append({'task': task, 'model': m, 'accuracy': float(r['frame'].correct.mean()), 'basis': 'own_complete',
                             'n': len(r['frame']), 'coverage': coverage, 'ci_low': ci[0], 'ci_high': ci[1]})
                continue
            c = common[(common.task == task) & (common.model == m)] if not common.empty else common
            if c.empty:
                rows.append({'task': task, 'model': m, 'accuracy': nan, 'basis': 'common_items_unavailable', 'n': 0,
                             'coverage': coverage, 'ci_low': nan, 'ci_high': nan})
            else:
                c = c.iloc[0]
                rows.append({'task': task, 'model': m, 'accuracy': float(c.accuracy_common), 'basis': 'common_items',
                             'n': int(c.n_common), 'coverage': coverage, 'ci_low': float(c.ci_low), 'ci_high': float(c.ci_high)})
    return pd.DataFrame(rows)


def loaders_gpu_validated(selected):
    """True only when every selected run recorded passing known-answer and longest-input gates."""
    if not selected:
        return False
    for run in selected.values():
        gates_file = run['path'] / 'gates.json'
        if not gates_file.exists():
            return False
        gates = json.loads(gates_file.read_text()).get('gates', {})
        if not (gates.get('known_answer', {}).get('passed') and gates.get('probe', {}).get('passed')):
            return False
    return True


def make_figures(report_dir, selected, summaries, answerability, positions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def grouped_bars(ax, groups, values_for, labels, width=.33):
        for i, label in enumerate(labels):
            xs = np.arange(len(groups)) + (i - (len(labels) - 1) / 2) * (width + .01)
            ax.bar(xs, [values_for(label, g) for g in groups], width=width, label=label)
        ax.set_xticks(np.arange(len(groups)), groups)

    def save_figure(filename):
        plt.tight_layout(); plt.savefig(report_dir / filename, dpi=160, bbox_inches='tight'); plt.close()

    def lookup(frame, index_col, value_col):
        table = frame.set_index(index_col) if not frame.empty else frame
        return lambda key: table.loc[key, value_col] if (not frame.empty and key in table.index) else np.nan

    base = summaries[summaries.setting == 'zero_shot'] if not summaries.empty else pd.DataFrame()
    if not base.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        grouped_bars(ax, MODELS, lambda t, m: lookup(base[base.task == t], 'model', 'accuracy')(m), EVAL_TASKS)
        ax.set_ylim(0, 1); ax.set_ylabel('Request-level accuracy'); ax.set_title('LongHealth, zero-shot (task-specific)')
        ax.legend(); save_figure('zero_shot_accuracy_by_task.png')
        fig, ax = plt.subplots(figsize=(9, 4.5))
        grouped_bars(ax, MODELS, lambda t, m: lookup(base[base.task == t], 'model', 'median_latency_s')(m), EVAL_TASKS)
        ax.set_ylabel('Median wall-time per request (seconds)'); ax.set_title('Native inference latency (hardware-dependent)')
        ax.legend(); save_figure('latency_by_model.png')
    if not answerability.empty:
        a = answerability[answerability.setting == 'zero_shot']
        if not a.empty:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            grouped_bars(ax, MODELS, lambda c, m: lookup(a[a.category == c], 'model', 'accuracy')(m), ('answerable', 'unanswerable'))
            ax.set_ylim(0, 1); ax.set_ylabel('Accuracy'); ax.set_title('Tasks 2/3: evidence present vs removed')
            ax.legend(); save_figure('task23_answerability.png')
    for task in EVAL_TASKS:
        entries = [(m, selected.get((m, task, 'zero_shot'))) for m in MODELS]
        entries = [(m, r) for m, r in entries if r and not r['frame'].empty]
        if not entries:
            continue
        fig, ax = plt.subplots(figsize=(6, 5)); ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        for m, r in entries:
            f = r['frame']; conf = f.confidence.to_numpy(); ix = np.minimum((conf * 10).astype(int), 9)
            pts = [(conf[ix == j].mean(), f.correct.to_numpy()[ix == j].mean()) for j in range(10) if (ix == j).any()]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=m)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean top-choice confidence', ylabel='Observed accuracy', title=f'{task}: reliability (zero-shot)')
        ax.legend(); save_figure(f'{task}_reliability.png')
    if not positions.empty:
        p = positions[positions.setting == 'zero_shot']
        for task in EVAL_TASKS:
            q = p[p.task == task]
            if q.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for m in MODELS:
                subset = q[q.model == m].set_index('position')
                if not subset.empty:
                    ax.plot(['early', 'middle', 'late'], [subset.loc[v, 'accuracy'] if v in subset.index else np.nan
                                                          for v in ('early', 'middle', 'late')], marker='o', label=m)
            ax.set(ylim=(0, 1), xlabel='Approximate supporting-evidence position', ylabel='Accuracy', title=f'{task}: position robustness (answerable items)')
            ax.legend(); save_figure(f'{task}_evidence_position.png')


def parse_run_overrides(items):
    out = {}
    for item in items or []:
        key, run_id = item.split('=', 1)
        model, task, setting = key.split(':')
        out[(model.upper(), task, setting)] = run_id
    return out


def cmd_compare(args):
    global BOOTSTRAP_DRAWS, EVAL_TASKS
    BOOTSTRAP_DRAWS = args.bootstrap_draws
    EVAL_TASKS = tuple(getattr(args, 'tasks', None) or TASKS)   # the report covers only the tasks that were run
    paths = Paths(args.project_dir)
    runs, invalid = discover_runs(paths.results)
    if invalid:
        print('INVALID RUNS:'); [print('  ', folder, '->', reason) for folder, reason in invalid]
        if not args.allow_invalid:
            fail('Invalid run directories found. Fix or move them, or pass --allow-invalid to report without them.', 2)
    runs = [r for r in runs if not (r['manifest']['config']['model'] == 'DREX'
                                    and r['manifest']['config'].get('quantization', 'native') != args.drex_quantization)]
    inventory = run_inventory(runs)
    if inventory.empty:
        fail('No v4 runs found under ' + str(paths.runs))
    selected = select_runs(runs, parse_run_overrides(args.run_id))
    selection = pd.DataFrame([{'model': k[0], 'task': k[1], 'setting': k[2], 'run_id': v['run_id'], 'successful': v['n_success'],
                               'expected': v['n_expected'], 'overlong': v['n_overlong'], 'failed': v['n_failed'],
                               'full_protocol': v['is_full_protocol'], 'complete': v['is_complete']} for k, v in selected.items()])
    missing = [(m, t) for m in MODELS for t in EVAL_TASKS if (m, t, 'zero_shot') not in selected]
    print(selection.sort_values(['task', 'setting', 'model']).to_string(index=False))
    print('Missing zero-shot model/task runs:', missing if missing else 'none')
    if any(not r['is_full_protocol'] or not r['is_complete'] for r in selected.values()):
        print('WARNING: Selected runs include pilots or incomplete data; do not label the comparison a complete benchmark.')
    summaries = pd.DataFrame([s for s in (summary_one(r, n_boot=BOOTSTRAP_DRAWS) for r in selected.values()) if s])
    common, answerability, positions = common_item_accuracy(selected), task23_answerability(selected), position_results(selected)
    consistency, rankings = repeat_consistency(selected), compute_rankings(selected)
    paired = paired_table(selected, bootstrap_draws=BOOTSTRAP_DRAWS)
    headline = headline_table(selected, common)
    lock = load_lock(args.lock_file) if Path(args.lock_file or LOCK_FILE).exists() else None
    digest = hashlib.sha256(json.dumps(sorted((list(k), v['run_id']) for k, v in selected.items())).encode()).hexdigest()[:8]
    report_dir = paths.reports / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{digest}"
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = {'inventory': inventory, 'selection': selection, 'summaries': summaries, 'common_items': common,
              'answerability': answerability, 'position': positions, 'repeat_consistency': consistency,
              'rankings': rankings, 'paired_tests': paired, 'headline': headline,
              'skipped_runs': pd.DataFrame(invalid, columns=['folder', 'reason'])}
    for name, table in tables.items():
        table.to_csv(report_dir / f'{name}.csv', index=False)
    for title, table in (('Summary', summaries), ('Common items', common), ('Headline', headline), ('Paired tests', paired)):
        if not table.empty:
            print(f'\n== {title} =='); print(table.round(4).to_string(index=False))
    make_figures(report_dir, selected, summaries, answerability, positions)
    full_possible = not missing and all(
        selected[(m, t, 'zero_shot')]['is_complete'] and selected[(m, t, 'zero_shot')]['is_full_protocol']
        and comparable_configs(selected[(m, t, 'zero_shot')], selected[(MODELS[0], t, 'zero_shot')]) for m in MODELS for t in EVAL_TASKS)
    manifest = {'generated_utc': datetime.now(timezone.utc).isoformat(),
                'title': 'LongHealth native structured-decision benchmark comparison',
                'selected_runs': {f'{k[0]}:{k[1]}:{k[2]}': v['run_id'] for k, v in selected.items()},
                'frozen_example_sets': sorted({v['manifest']['config'].get('examples_sha256') or '' for v in selected.values()}),
                'missing_zero_shot_runs': [list(x) for x in missing], 'full_comparison_possible': full_possible,
                'revisions': lock, 'drex_quantization': args.drex_quantization,
                'cautions': ['Decision-adapted, not the official free-text leaderboard',
                             'All models scored one frozen example set built with a reference tokenizer',
                             'A non-significant p-value is not evidence of equal performance',
                             'Timing differs by GPU and runtime', '20 underlying patients; bootstrap intervals are exploratory'],
                'model_loaders_real_gpu_validated': loaders_gpu_validated(selected)}
    (report_dir / 'report_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('\nComparison artifacts:', report_dir)
    if not full_possible:
        print('NOT A COMPLETE BENCHMARK: model/task coverage, completeness or frozen-set agreement is missing.')
    return 0


def cmd_all(args):
    paths, lock = Paths(args.project_dir), load_lock(args.lock_file)
    if frozen_files(paths, lock, args.seed, args.state_token_budget) is None:
        cmd_build_frozen(args)
    code = cmd_run(args)
    if code == 130:
        return code
    compare_code = cmd_compare(args)
    return code or compare_code


def parse_overrides(items):
    out = {}
    for item in items or []:
        model, path = item.split('=', 1)
        out[model.upper()] = path
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-dir', default=None, help='models folder (required for setup-envs)')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--max-tokens', type=int, default=16000)
    p.add_argument('--state-token-budget', type=int, default=14000)
    p.add_argument('--lock-file', default=None, help=argparse.SUPPRESS)
    p.add_argument('--worker-override', action='append', default=[], metavar='MODEL=path', help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('lock'); s.add_argument('--update', action='store_true'); s.set_defaults(fn=cmd_lock)
    s = sub.add_parser('setup-envs'); s.add_argument('--models', nargs='+', default=list(MODELS))
    s.add_argument('--hf-cache', default=None, metavar='DIR',
                   help='shared Hugging Face hub cache (holds models--ORG--NAME); checkpoints are verified and linked instead of downloaded')
    s.set_defaults(fn=cmd_setup_envs)
    s = sub.add_parser('build-frozen'); s.set_defaults(fn=cmd_build_frozen)
    def run_opts(s):
        s.add_argument('--models', nargs='+', default=['all'])
        s.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS))
        s.add_argument('--gpu', default='1', help='PHYSICAL GPU index (CUDA_VISIBLE_DEVICES); the worker sees it as cuda:0')
        s.add_argument('--pilot-only', action='store_true'); s.add_argument('--no-pilot', action='store_true')
        s.add_argument('--max-questions', type=int, default=None); s.add_argument('--repeats', type=int, default=None)
        s.add_argument('--drex-int8', action='store_true'); s.add_argument('--allow-probe-oom', action='store_true')
    def compare_opts(s):
        s.add_argument('--drex-quantization', choices=['native', '8bit'], default='native')
        s.add_argument('--run-id', action='append', metavar='MODEL:task:setting=ID')
        s.add_argument('--allow-invalid', action='store_true'); s.add_argument('--bootstrap-draws', type=int, default=1200)
    s = sub.add_parser('run'); run_opts(s); s.set_defaults(fn=cmd_run)
    s = sub.add_parser('status'); s.set_defaults(fn=cmd_status)
    s = sub.add_parser('compare'); compare_opts(s); s.set_defaults(fn=cmd_compare)
    s.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS), help='tasks the report covers (`all` uses its own --tasks)')
    s = sub.add_parser('all'); run_opts(s); compare_opts(s); s.set_defaults(fn=cmd_all)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.worker_override = parse_overrides(args.worker_override)
    if args.project_dir is None and args.command != "setup-envs":   # LongHealth-only commands keep their old default
        args.project_dir = str(HERE.parent / "longhealth_decision")
    global SEED
    SEED = args.seed
    code = args.fn(args)
    sys.exit(code or 0)


if __name__ == '__main__':
    main()
