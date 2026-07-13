from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from datasets import Dataset, DatasetDict
from GameDialogue.config import DEFAULT_SEED, get_paths
from GameDialogue.dataset_utils import load_from_dir, save_to_dir
from GameDialogue.io_utils import setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default=str(get_paths().datasets_cleaned / "chatml"))
    p.add_argument("--out_dir", type=str, default=str(get_paths().datasets_processed / "split"))
    p.add_argument("--test_size", type=float, default=0.02)
    p.add_argument("--val_size", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()

def _ensure_splits(ds: Dataset | DatasetDict, test_size: float, val_size: float, seed: int) -> DatasetDict:
    if isinstance(ds, DatasetDict):
        keys = set(ds.keys())
        if {"train", "validation", "test"}.issubset(keys):
            return ds
        base = ds["train"] if "train" in keys else next(iter(ds.values()))
    else:
        base = ds
    tmp = base.train_test_split(test_size=test_size + val_size, seed=seed, shuffle=True)
    remain = tmp['test']
    ratio = test_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.5
    tmp2 = remain.train_test_split(test_size=ratio, seed=seed, shuffle=True)
    return DatasetDict({'train': tmp['train'], 'validation': tmp2['train'], 'test': tmp2['test']})

def main() -> None:
    args = parse_args(); logger = setup_logging(); ds = load_from_dir(Path(args.in_dir))
    split = _ensure_splits(ds, args.test_size, args.val_size, args.seed)
    for k in split.keys():
        logger.info(f"{k}: {split[k].num_rows}")
    save_to_dir(split, Path(args.out_dir)); logger.info(f"Saved: {args.out_dir}")

if __name__ == '__main__':
    main()
