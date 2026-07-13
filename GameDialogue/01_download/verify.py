from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from pathlib import Path
from GameDialogue.config import get_paths
from GameDialogue.io_utils import setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default=str(get_paths().models_base))
    p.add_argument("--dataset_dir", type=str, default=str(get_paths().datasets_raw / "chinese_roleplay_novel"))
    return p.parse_args()

def _require_any(path: Path, names: list[str]) -> None:
    for n in names:
        if any(ch in n for ch in ["*", "?", "["]):
            if any(True for _ in path.glob(n)):
                return
            continue
        if (path / n).exists():
            return
    raise FileNotFoundError(f"{path} missing any of: {names}")

def _verify_dataset_dir(dataset_dir: Path) -> None:
    if (dataset_dir / "dataset_dict.json").exists():
        split_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
        if not split_dirs:
            raise FileNotFoundError(f"{dataset_dir} has dataset_dict.json but no split directories")
        for split_dir in split_dirs:
            _require_any(split_dir, ["dataset_info.json", "state.json"])
        return
    _require_any(dataset_dir, ["dataset_info.json", "state.json"])

def main() -> None:
    args = parse_args()
    logger = setup_logging()
    model_dir = Path(args.model_dir)
    dataset_dir = Path(args.dataset_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    _require_any(model_dir, ["config.json"])
    _require_any(model_dir, ["tokenizer.json", "tokenizer.model"])
    _require_any(model_dir, ["model.safetensors", "pytorch_model.bin", "model-*.safetensors"])
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    _verify_dataset_dir(dataset_dir)
    logger.info("Verify OK.")

if __name__ == "__main__":
    main()
