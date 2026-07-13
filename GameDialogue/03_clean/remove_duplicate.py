from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import hashlib
from datasets import Dataset, DatasetDict
from GameDialogue.chatml import as_messages
from GameDialogue.config import get_paths
from GameDialogue.dataset_utils import get_splits, load_from_dir, save_to_dir
from GameDialogue.io_utils import compact_text, setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default=str(get_paths().datasets_cleaned / "step2_norm"))
    p.add_argument("--out_dir", type=str, default=str(get_paths().datasets_cleaned / "step3_dedup"))
    return p.parse_args()

def _dedup_key(ex: dict) -> str:
    msgs = as_messages(ex)
    if msgs is not None:
        s = "\n".join([f"{m['role']}:{compact_text(m['content'])}" for m in msgs if m.get("content")])
    else:
        s = "\n".join([f"{k}:{compact_text(v)}" for k, v in ex.items() if isinstance(v, str) and v.strip()])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _dedup_split(ds: Dataset) -> Dataset:
    seen, keep = set(), []
    for i in range(ds.num_rows):
        key = _dedup_key(ds[i])
        if key in seen:
            continue
        seen.add(key); keep.append(i)
    return ds.select(keep)

def main() -> None:
    args = parse_args(); logger = setup_logging(); ds = load_from_dir(Path(args.in_dir))
    out_splits = {}
    for name, split_ds in get_splits(ds).items():
        dedup = _dedup_split(split_ds)
        logger.info(f"split={name} rows {split_ds.num_rows} -> {dedup.num_rows}")
        out_splits[name] = dedup
    out = DatasetDict(out_splits) if len(out_splits) > 1 else next(iter(out_splits.values()))
    save_to_dir(out, Path(args.out_dir)); logger.info(f"Saved: {args.out_dir}")

if __name__ == '__main__':
    main()
