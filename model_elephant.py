"""Elephant-1 worker. Runs in the `realdata-elephant` conda env (torch 2.8, qev from the checkpoint's setup.sh)."""
import gc
import sys
from pathlib import Path

import lh_core
from lh_core import EnvironmentMismatch, OverlongInput

# qev/engine.py DecisionEngine.decide raises ValueError("state plus longest question is N tokens, limit is M") and
# ValueError("request is N tokens, limit is M") for over-length requests. Only that exact phrase on a ValueError counts as overlong.
# (To be confirmed by the Step-2 spike with a real over-length request.)
OVERLONG_MESSAGES = ("tokens, limit is",)


class Adapter:
    name = "ELEPHANT"
    quantization = "native"
    min_free_vram_gb = lh_core.MODELS_INFO["ELEPHANT"]["min_vram_gb"]

    def __init__(self, spec):
        self.spec, self.cfg, self.device = spec, spec["config"], spec["device"]
        self.weights = Path(spec["project_dir"]) / lh_core.MODELS_INFO["ELEPHANT"]["weights_dir"]

    def check_environment(self):
        import torch
        if not torch.__version__.startswith("2.8."):
            raise EnvironmentMismatch(f"Elephant targets PyTorch 2.8, found {torch.__version__}")
        if not torch.cuda.is_available():
            raise EnvironmentMismatch("CUDA is not available")
        try:
            import qev  # noqa: F401
        except ImportError as exc:
            raise EnvironmentMismatch(f"qev runtime not importable ({exc}); rerun setup-envs") from exc
        if not self.weights.is_dir():
            raise EnvironmentMismatch(f"Checkpoint not found at {self.weights}; run setup-envs")

    def load(self):
        import torch
        torch.manual_seed(self.cfg["seed"])
        from qev.engine import DecisionEngine
        from qev.schema import SystemOneRequest
        self._request_cls = SystemOneRequest
        self.engine = DecisionEngine(str(self.weights), device=self.device)

    def predict(self, record):
        try:
            resp, _ = self.engine.decide(self._request_cls(state=record["state"], questions=record["questions"]))
        except Exception as exc:
            if isinstance(exc, ValueError) and any(m in str(exc) for m in OVERLONG_MESSAGES):
                raise OverlongInput(str(exc)) from exc
            raise
        result = resp.answers["longhealth"]
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif not isinstance(result, dict):
            result = vars(result)
        return result["probabilities"]

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
