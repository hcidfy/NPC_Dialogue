from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from datasets import Dataset, DatasetDict
from GameDialogue.chatml import as_messages
from GameDialogue.config import get_paths
from GameDialogue.dataset_utils import get_splits, load_from_dir, load_raw_dataset, save_to_dir
from GameDialogue.io_utils import setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default=str(get_paths().datasets_raw / "chinese_roleplay_novel"))
    p.add_argument("--out_dir", type=str, default=str(get_paths().datasets_cleaned / "step1_no_null"))
    p.add_argument("--from_hf_if_missing", action="store_true")
    return p.parse_args()

def _keep(ex: dict) -> bool:
    msgs = as_messages(ex)
    if msgs is None:
        return False
    return any((m.get("content") or "").strip() for m in msgs)

def main() -> None:
    args = parse_args(); logger = setup_logging(); in_dir = Path(args.in_dir)
    ds = load_from_dir(in_dir) if in_dir.exists() else load_raw_dataset()
    out_splits = {}
    for name, split_ds in get_splits(ds).items():
        filtered = split_ds.filter(_keep)
        logger.info(f"split={name} rows {split_ds.num_rows} -> {filtered.num_rows}")
        out_splits[name] = filtered
    out = DatasetDict(out_splits) if len(out_splits) > 1 else next(iter(out_splits.values()))
    save_to_dir(out, Path(args.out_dir))
    logger.info(f"Saved: {args.out_dir}")

if __name__ == '__main__':
    main()
