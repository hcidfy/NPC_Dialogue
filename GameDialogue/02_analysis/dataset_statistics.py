from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from collections import Counter
from typing import Any
from datasets import Dataset
from GameDialogue.config import DEFAULT_DATASET_ID, get_paths
from GameDialogue.dataset_utils import get_columns, get_splits, load_raw_dataset, pick_example
from GameDialogue.io_utils import save_json, setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--raw_dir", type=str, default=str(get_paths().datasets_raw / "chinese_roleplay_novel"))
    p.add_argument("--out_dir", type=str, default=str(get_paths().outputs / "logs"))
    return p.parse_args()

def _basic_stats(ds: Dataset) -> dict[str, Any]:
    cols = ds.column_names
    out: dict[str, Any] = {"rows": ds.num_rows, "columns": cols}
    out["string_columns"] = [c for c in cols if isinstance(ds[0].get(c), str)]
    out["preview"] = [{k: ds[i].get(k) for k in cols[:8]} for i in range(min(3, ds.num_rows))]
    out["column_types_head"] = dict(Counter(type(ds[0].get(c)).__name__ for c in cols))
    return out

def main() -> None:
    args = parse_args()
    logger = setup_logging()
    ds = load_raw_dataset(dataset_id=args.dataset_id, local_dir=Path(args.raw_dir))
    logger.info(f"Columns: {get_columns(ds)}")
    stats = {"dataset_id": args.dataset_id, "splits": {}}
    for name, split_ds in get_splits(ds).items():
        stats["splits"][name] = _basic_stats(split_ds)
    stats["example_head"] = pick_example(ds)
    save_json(Path(args.out_dir) / 'dataset_statistics.json', stats)
    logger.info('Done.')

if __name__ == '__main__':
    main()
