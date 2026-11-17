"""BioDecision v2 4B worker. Runs in the `realdata-med` conda env (same stack as MedDecider).

Ported from GeminiOps/MedDecision/model_biodecision.py. Scoring path, from the model repo:
- prompt: `biodecision_server.question_options` + `biodecision_server.render` (its system prompt and the JSON payload
  {state, question, options: [{label, key, description}]}, chat template with thinking off). Each option's key is the
  code; its description is the code's long description with --descriptions, otherwise the code again (question_options'
  own rule for an empty value). At most 24 options (biodecision_server.LETTERS = "A".."X").
- readout: the card's Transformers quickstart: logits at the answer position (logits_to_keep=1), the option letters'
  logits divided by the global temperature from biodecision_config.json, softmax.
load() checks that render()'s text tokenises to the ids apply_chat_template gives directly (made-up request, no data).
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
        self.weights = Path(spec["project_dir"]) / info["weights_dir"]
        self.min_free_vram_gb = info["min_vram_gb"]
        self.max_options = info["max_options"]
        self.max_tokens = int(self.cfg["max_tokens"])

    def check_environment(self):
        import torch
        from packaging.version import Version
        if Version(metadata.version("transformers")) < Version("5.18"):
            raise EnvironmentMismatch(f"expected the realdata-med stack (transformers>=5.18), found {metadata.version('transformers')}")
        if not torch.cuda.is_available():
            raise EnvironmentMismatch("CUDA is not available")
        missing = [f for f in ("biodecision_server.py", "biodecision_config.json", "config.json", "tokenizer.json",
                               "model.safetensors.index.json") if not (self.weights / f).is_file()]
        if missing:
            raise EnvironmentMismatch(f"{self.weights} lacks {missing}; run realdata_setup.sh")

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(self.cfg["seed"])
        sys.dont_write_bytecode = True               # the checkpoint folder may be a link into a shared cache
        sys.path.insert(0, str(self.weights))
        self.bds = importlib.import_module("biodecision_server")   # top-level imports are stdlib only; vLLM is imported lazily
        if Path(self.bds.__file__).resolve().parent != self.weights.resolve():
            raise EnvironmentMismatch(f"imported {self.bds.__file__}, not the pinned copy")
        if len(self.bds.LETTERS) != self.max_options:
            raise EnvironmentMismatch(f"biodecision_server.LETTERS has {len(self.bds.LETTERS)} letters, lh_core says {self.max_options}")
        self.code_sha256 = sha256_file(self.bds.__file__)
        self.bio_cfg = json.loads((self.weights / "biodecision_config.json").read_text())
        self.T = float(self.bio_cfg["temperature"])
        self.tok = AutoTokenizer.from_pretrained(str(self.weights))
        self.model = AutoModelForCausalLM.from_pretrained(str(self.weights), dtype=torch.bfloat16, device_map="cuda:0").eval()
        encoded = [self.tok.encode(c, add_special_tokens=False) for c in self.bds.LETTERS]
        if any(len(e) != 1 for e in encoded):
            raise EnvironmentMismatch("option letters are not single tokens")
        self.letter_ids = [e[0] for e in encoded]   # as the server's Decider.__init__
        self._self_check()
        lh_core.log(f"BIODECISION: loaded, temperature {self.T}, biodecision_server.py sha256 {self.code_sha256[:12]}, "
                    f"prompt matches the card's quickstart")

    def _prompt(self, state, question):
        text, opts, keys = self.bds.question_options({"type": "choice", "instructions": question["instructions"],
                                                      "criteria": question["criteria"]})
        return self.bds.render(self.tok, state, text, opts), opts, keys

    def _self_check(self):
        """render()'s text must tokenise to the ids the card's quickstart gets from apply_chat_template directly."""
        state = "Patient with poorly controlled type 2 diabetes."
        q = {"type": "choice", "instructions": "Which ICD-10-CA code best matches the diagnosis described above?",
             "criteria": {"E119": "Type 2 diabetes mellitus without complications", "I10": None}}
        prompt, opts, _ = self._prompt(state, q)
        tagged = [{"label": self.bds.LETTERS[i], "key": k, "description": d} for i, (k, d) in enumerate(opts)]
        msgs = [{"role": "system", "content": self.bds.SYSTEM},
                {"role": "user", "content": json.dumps({"state": state, "question": q["instructions"], "options": tagged},
                                                       ensure_ascii=False)}]
        direct = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True)["input_ids"]
        direct = list(direct[0]) if direct and isinstance(direct[0], list) else list(direct)
        if self.tok(prompt, add_special_tokens=False)["input_ids"] != direct:
            raise EnvironmentMismatch("BioDecision prompt differs from the card's apply_chat_template quickstart")
        if self.bds.SYSTEM != self.bio_cfg.get("system_prompt"):
            raise EnvironmentMismatch("biodecision_server.SYSTEM differs from biodecision_config.json system_prompt")

    def predict(self, record):
        import torch
        q = record["questions"][lh_core.QUESTION_ID]
        if not 2 <= len(q["criteria"]) <= len(self.bds.LETTERS):
            raise ValueError(f"BioDecision scores 2 to {len(self.bds.LETTERS)} options, got {len(q['criteria'])}; build with --top 24 or less")
        prompt, _, keys = self._prompt(record["state"], q)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        n_tokens = int(enc["input_ids"].shape[1])
        if n_tokens > self.max_tokens:
            raise OverlongInput(f"BioDecision prompt is {n_tokens} tokens; limit is {self.max_tokens}")
        with torch.inference_mode():
            logits = self.model(**enc.to(self.model.device), logits_to_keep=1).logits[0, -1].float()
        z = logits[self.letter_ids[:len(keys)]] / self.T
        return dict(zip(keys, torch.softmax(z, -1).cpu().tolist()))

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
