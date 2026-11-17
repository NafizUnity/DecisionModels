"""MedDecider-27B / MedDecider-31B worker. Runs in the `realdata-med` conda env (Python 3.12, torch 2.8, transformers 5.22.2, peft).

Ported from GeminiOps/MedDecision/model_meddecider.py (which ran both models on LongHealth Task 1). Scoring path: the model
repo's own `meddecider_infer.MedDecider`, loaded from the pinned adapter folder with the pinned local base folder
(`base=`), so nothing is fetched from the Hub.
- Options: the model reads lettered option texts ("A. ..."), at most 10 (LETTERS = "ABCDEFGHIJ"; model cards: "2 to 10
  answer options"). Each option's text is its code, or "CODE: description" when the build used --descriptions.
- Readout: decide()'s loop line for line (27B: letter_logits with its global bidirectional mask, temperature 1.75;
  31B: logits_to_keep=1 letter logits, temperature 1.49), both option orders (as given, reversed) averaged. decide()
  itself keys its result by option text and rounds to 4 decimals; this loop keeps positions and full precision.
  load() checks it against the real decide() on a made-up request before anything is scored.
"""
import gc
import hashlib
import importlib
import json
import sys
from importlib import metadata
from pathlib import Path

import lh_core
from lh_core import EnvironmentMismatch, OverlongInput


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Adapter:
    quantization = "bf16"

    def __init__(self, spec):
        self.spec, self.cfg, self.name = spec, spec["config"], spec["model"]
        info = lh_core.MODELS_INFO[self.name]
        root = Path(spec["project_dir"])
        self.adapter_dir, self.base_dir, self.base_repo = root / info["weights_dir"], root / info["base_dir"], info["base_repo"]
        self.min_free_vram_gb = info["min_vram_gb"]
        self.max_options = info["max_options"]
        self.max_tokens = int(self.cfg["max_tokens"])

    def check_environment(self):
        import torch
        from packaging.version import Version
        if sys.version_info[:2] != (3, 12):
            raise EnvironmentMismatch(f"expected Python 3.12 (realdata-med), found {sys.version.split()[0]}")
        if Version(metadata.version("transformers")) < Version("5.18"):
            raise EnvironmentMismatch(f"meddecider_infer.py needs transformers>=5.18, found {metadata.version('transformers')}")
        for pkg in ("peft", "flash-linear-attention"):
            try:
                metadata.version(pkg)
            except metadata.PackageNotFoundError as exc:
                raise EnvironmentMismatch(f"{pkg} is not installed (meddecider_infer.py lists it); rerun setup-envs") from exc
        if not torch.cuda.is_available():
            raise EnvironmentMismatch("CUDA is not available")
        for folder, files in ((self.adapter_dir, ("meddecider_infer.py", "meddecider_config.json", "adapter_config.json",
                                                  "adapter_model.safetensors")),
                              (self.base_dir, ("config.json", "tokenizer.json", "tokenizer_config.json"))):
            missing = [f for f in files if not (folder / f).is_file()]
            if missing:
                raise EnvironmentMismatch(f"{folder} lacks {missing}; run realdata_setup.sh")
        self.md_cfg = json.loads((self.adapter_dir / "meddecider_config.json").read_text())
        if self.md_cfg["base_model"].split("@")[0] != self.base_repo:
            raise EnvironmentMismatch(f"meddecider_config.json names base {self.md_cfg['base_model']}, expected {self.base_repo}")

    def load(self):
        import torch
        torch.manual_seed(self.cfg["seed"])
        sys.dont_write_bytecode = True               # the adapter folder may be a link into a shared cache
        sys.path.insert(0, str(self.adapter_dir))    # after lh_core was imported
        self.mdi = importlib.import_module("meddecider_infer")
        if Path(self.mdi.__file__).resolve().parent != self.adapter_dir.resolve():
            raise EnvironmentMismatch(f"imported {self.mdi.__file__}, not the pinned copy in {self.adapter_dir}")
        if len(self.mdi.LETTERS) != self.max_options:
            raise EnvironmentMismatch(f"meddecider_infer.LETTERS has {len(self.mdi.LETTERS)} letters, lh_core says {self.max_options}")
        self.code_sha256 = sha256_file(self.mdi.__file__)
        self.md = self.mdi.MedDecider(str(self.adapter_dir), base=str(self.base_dir), device="cuda:0")
        diff = self._self_check()
        lh_core.log(f"{self.name}: loaded on base {self.base_dir.name} | mask {self.md_cfg.get('mask', 'causal')} | choice temperature "
                    f"{self._temperature()} | meddecider_infer.py sha256 {self.code_sha256[:12]} | self-check vs decide(): max diff {diff:.1e}")

    def _temperature(self):
        md = self.md
        return md.temperature("choice") if hasattr(md, "temperature") else md.cfg["temperature"].get("choice", 1.0)

    def _order_probs(self, text, n):
        """One option order: exactly the per-order lines of the model's decide()."""
        import torch
        md, T = self.md, self._temperature()
        if hasattr(md, "letter_logits"):                       # MedDecider-27B (global_bidir mask inside letter_logits)
            return torch.softmax(md.letter_logits(text, n) / T, -1)
        with torch.no_grad():                                  # MedDecider-31B (decide() is @torch.no_grad)
            ids = md.tok(text, return_tensors="pt", add_special_tokens=False).to(md.model.device)
            z = md.model(**ids, logits_to_keep=1).logits[0, -1, md.letter_ids[:n]].float().cpu() / T
        return torch.softmax(z, -1)

    def _score(self, state, question, options):
        import torch
        n = len(options)
        if not 2 <= n <= len(self.mdi.LETTERS):
            raise ValueError(f"{self.name} scores 2 to {len(self.mdi.LETTERS)} options, got {n}; build with --top {len(self.mdi.LETTERS)} or less")
        perms = [list(range(n)), list(range(n))[::-1]]
        texts = [self.md.prompt(state, question, [options[i] for i in order]) for order in perms]
        n_tokens = max(len(self.md.tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
        if n_tokens > self.max_tokens:
            raise OverlongInput(f"MedDecider prompt is {n_tokens} tokens; limit is {self.max_tokens}")
        probs = torch.zeros(n)
        for order, text in zip(perms, texts):
            p = self._order_probs(text, n)
            for pos, orig in enumerate(order):
                probs[orig] += p[pos] / len(perms)
        return [float(x) for x in probs]

    def _self_check(self):
        """The position-preserving loop must reproduce the model's own decide() (made-up request, no data)."""
        state, question = "Patient with poorly controlled type 2 diabetes.", "Which ICD-10-CA code best matches the diagnosis described above?"
        options = ["E119: Type 2 diabetes mellitus without complications", "I10: Essential (primary) hypertension", "J459: Asthma, unspecified"]
        mine = self._score(state, question, options)
        theirs = self.md.decide(state, question, options, "choice")
        diff = max(abs(a - theirs[o]) for a, o in zip(mine, options))
        if diff > 2e-3:
            raise EnvironmentMismatch(f"{self.name}: position-preserving scorer differs from decide() by {diff:.4f}")
        return diff

    def predict(self, record):
        q = record["questions"][lh_core.QUESTION_ID]
        keys = list(q["criteria"])
        options = [k if not q["criteria"][k] else f"{k}: {q['criteria'][k]}" for k in keys]
        return dict(zip(keys, self._score(record["state"], q["instructions"], options)))

    def cleanup_after_failure(self):
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def reset_peak_vram(self):
        import torch
        torch.cuda.reset_peak_memory_stats()

    def peak_vram_gb(self):
        import torch
        return torch.cuda.max_memory_allocated() / 1024 ** 3

    def gpu_name(self):
        import torch
        return torch.cuda.get_device_name(0)


if __name__ == "__main__":
    lh_core.worker_main(Adapter)
