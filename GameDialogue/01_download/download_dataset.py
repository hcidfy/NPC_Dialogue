from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from pathlib import Path
from datasets import load_dataset
from GameDialogue.config import DEFAULT_DATASET_ID, get_paths
from GameDialogue.io_utils import setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=str(get_paths().datasets_raw / "chinese_roleplay_novel"))
    return p.parse_args()

def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    logger = setup_logging()
    logger.info(f"Loading dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id, split=args.split)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to disk: {save_dir}")
    ds.save_to_disk(str(save_dir))
    logger.info("Done.")

if __name__ == "__main__":
    main()
