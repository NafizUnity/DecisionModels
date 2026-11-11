"""Cloudflare Clef worker. Runs in the `realdata-clef` conda env (torch 2.11, transformers 5.10.2)."""
import gc
import sys
from importlib import metadata
from pathlib import Path

import lh_core
from lh_core import EnvironmentMismatch, OverlongInput


class Adapter:
    name = "CLEF"
    quantization = "native"
    min_free_vram_gb = lh_core.MODELS_INFO["CLEF"]["min_vram_gb"]

    def __init__(self, spec):
        self.spec, self.cfg, self.device = spec, spec["config"], spec["device"]
        self.weights = Path(spec["project_dir"]) / lh_core.MODELS_INFO["CLEF"]["weights_dir"]

    def check_environment(self):
        import torch
        if not torch.__version__.startswith("2.11."):
            raise EnvironmentMismatch(f"Clef was tested with torch 2.11, found {torch.__version__}")
        if metadata.version("transformers") != "5.10.2":
            raise EnvironmentMismatch(f"Clef needs transformers 5.10.2, found {metadata.version('transformers')}")
        if not torch.cuda.is_available():
            raise EnvironmentMismatch("CUDA is not available")
        if not (self.weights / "joint_schema_model.py").exists():
            raise EnvironmentMismatch(f"Clef files not found at {self.weights}; run setup-envs")

    def load(self):
        import torch
        torch.manual_seed(self.cfg["seed"])
        sys.path.insert(0, str(self.weights))   # after lh_core has been imported
        sys.dont_write_bytecode = True          # no __pycache__ inside the checkpoint (it may be a shared, read-only cache)
        from joint_schema_model import collate_records, encode_record, load_release_model
        self._encode, self._collate = encode_record, collate_records
        self.model, self.processor = load_release_model(str(self.weights), device=self.device)

    def predict(self, record):
        import torch
        # encode_record truncates the state to max_length without raising; encode in full and reject instead.
        enc = self._encode(self.processor.tokenizer, record, max_length=10 ** 9, processor=self.processor)
        if len(enc.input_ids) > self.cfg["max_tokens"]:
            raise OverlongInput(f"Clef input is {len(enc.input_ids)} tokens; limit is {self.cfg['max_tokens']}.")
        batch = self._collate([enc], self.processor.tokenizer.pad_token_id, torch.device(self.device))
        with torch.inference_mode():
            logits = self.model(batch)[0]
        assert len(logits) == 1 and len(enc.questions) == 1
        values = logits[0].float().softmax(-1).detach().cpu().tolist()
        return dict(zip(enc.questions[0].option_ids, values))

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
