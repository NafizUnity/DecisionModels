"""Drex Max worker. Runs in the `realdata-drex` conda env (Python 3.12, torch >= 2.14, transformers 5.17.0)."""
import gc
import sys
from importlib import metadata
from pathlib import Path

import lh_core
from lh_core import EnvironmentMismatch, OverlongInput

INFO = lh_core.MODELS_INFO["DREX"]


class Adapter:
    name = "DREX"

    def __init__(self, spec):
        self.spec, self.cfg, self.device = spec, spec["config"], spec["device"]
        self.int8 = bool(self.cfg.get("drex_int8"))
        self.quantization = "8bit" if self.int8 else "native"
        self.min_free_vram_gb = INFO["min_vram_gb_int8"] if self.int8 else INFO["min_vram_gb"]
        self.weights = Path(spec["project_dir"]) / INFO["weights_dir"]
        self.repo = Path(spec["project_dir"]) / "drex-decision-models"

    def check_environment(self):
        import torch
        if sys.version_info < (3, 12):
            raise EnvironmentMismatch(f"Drex needs Python >= 3.12, found {sys.version.split()[0]}")
        major_minor = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if major_minor < (2, 14):
            raise EnvironmentMismatch(f"Drex needs torch >= 2.14, found {torch.__version__}")
        if metadata.version("transformers") != "5.17.0":
            raise EnvironmentMismatch(f"Drex needs transformers 5.17.0, found {metadata.version('transformers')}")
        try:
            metadata.version("flash-linear-attention")
        except metadata.PackageNotFoundError as exc:
            raise EnvironmentMismatch("flash-linear-attention is not installed") from exc
        if not torch.cuda.is_available():
            raise EnvironmentMismatch("CUDA is not available")
        if not (self.repo / ".git").exists() or not self.weights.is_dir():
            raise EnvironmentMismatch(f"Drex repo or checkpoint missing under {self.spec['project_dir']}; run setup-envs")

    def load(self):
        import torch
        torch.manual_seed(self.cfg["seed"])
        sys.path.insert(0, str(self.repo))   # after lh_core has been imported
        from drex_max.model import DecisionModel, answer
        self._answer = answer
        self.model = DecisionModel(str(self.weights), device=self.device, load_in_8bit=self.int8)

    def predict(self, record):
        question = record["questions"]["longhealth"]
        row = {"state": record["state"], "question": question}
        try:
            probabilities = self.model.predict([row], batch_size=1, max_length=self.cfg["max_tokens"])[0]
        except ValueError as exc:
            if str(exc).startswith("Input longer than"):
                raise OverlongInput(str(exc)) from exc
            raise
        return self._answer(question, probabilities)["probabilities"]

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
