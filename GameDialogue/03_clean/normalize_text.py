from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from datasets import Dataset, DatasetDict
from GameDialogue.chatml import normalize_text
from GameDialogue.config import get_paths
from GameDialogue.dataset_utils import get_splits, load_from_dir, save_to_dir
from GameDialogue.io_utils import setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default=str(get_paths().datasets_cleaned / "step1_no_null"))
    p.add_argument("--out_dir", type=str, default=str(get_paths().datasets_cleaned / "step2_norm"))
    return p.parse_args()

def _normalize_example(ex: dict) -> dict:
    out = dict(ex)
    for k, v in ex.items():
        if isinstance(v, str):
            out[k] = normalize_text(v)
    for list_key in ["messages", "conversations"]:
        if list_key in out and isinstance(out[list_key], list):
            new_list = []
            for m in out[list_key]:
                if not isinstance(m, dict):
                    continue
                mm = dict(m)
                for key in ["content", "value"]:
                    if key in mm and isinstance(mm[key], str):
                        mm[key] = normalize_text(mm[key])
                new_list.append(mm)
            out[list_key] = new_list
    return out

def main() -> None:
    args = parse_args(); logger = setup_logging(); ds = load_from_dir(Path(args.in_dir))
    out_splits = {name: split_ds.map(_normalize_example, num_proc=1) for name, split_ds in get_splits(ds).items()}
    for name, split_ds in out_splits.items():
        logger.info(f"split={name} rows={split_ds.num_rows}")
    out = DatasetDict(out_splits) if len(out_splits) > 1 else next(iter(out_splits.values()))
    save_to_dir(out, Path(args.out_dir))
    logger.info(f"Saved: {args.out_dir}")

if __name__ == '__main__':
    main()
