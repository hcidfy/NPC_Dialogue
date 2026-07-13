from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    @property
    def datasets_raw(self) -> Path:
        return self.root / "datasets" / "raw"
    @property
    def datasets_cleaned(self) -> Path:
        return self.root / "datasets" / "cleaned"
    @property
    def datasets_processed(self) -> Path:
        return self.root / "datasets" / "processed"
    @property
    def models_base(self) -> Path:
        return self.root / "models" / "Qwen3-4B-Instruct"
    @property
    def models_lora(self) -> Path:
        return self.root / "models" / "lora"
    @property
    def models_merged(self) -> Path:
        return self.root / "models" / "Qwen3-4B-Instruct-merged"
    @property
    def outputs(self) -> Path:
        return self.root / "outputs"
    @property
    def outputs_logs(self) -> Path:
        return self.outputs / "logs"
    @property
    def outputs_checkpoints(self) -> Path:
        return self.outputs / "checkpoints"
    @property
    def outputs_figures(self) -> Path:
        return self.outputs / "figures"

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET_ID = "LooksJuicy/Chinese-Roleplay-Novel"
DEFAULT_SEED = 42

def get_project_root() -> Path:
    return Path(__file__).resolve().parent

def get_paths() -> ProjectPaths:
    return ProjectPaths(root=get_project_root())
