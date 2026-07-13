from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from datasets import Dataset, DatasetDict
from GameDialogue.chatml import as_messages, ensure_chat_pair
from GameDialogue.config import get_paths
from GameDialogue.dataset_utils import get_splits, load_from_dir, save_to_dir
from GameDialogue.io_utils import setup_logging
DROP_KEYS = {"conversations", "text", "instruction", "input", "output", "prompt", "response", "question", "answer"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default=str(get_paths().datasets_cleaned / "step3_dedup"))
    p.add_argument("--out_dir", type=str, default=str(get_paths().datasets_cleaned / "chatml"))
    p.add_argument("--keep_other_fields", action="store_true")
    return p.parse_args()

def _convert(ex: dict, keep_other_fields: bool) -> dict | None:
    msgs = as_messages(ex)
    if msgs is None:
        return None
    msgs = ensure_chat_pair(msgs)
    if msgs is None:
        return None
    out = {"messages": msgs}
    if keep_other_fields:
        for k, v in ex.items():
            if k in DROP_KEYS or k == 'messages':
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
    return out

def _map_split(ds: Dataset, keep_other_fields: bool) -> Dataset:
    mapped = ds.map(lambda ex: _convert(ex, keep_other_fields) or {"__drop__": True}, num_proc=1)
    if "__drop__" in mapped.column_names:
        mapped = mapped.filter(lambda e: not e.get("__drop__", False)).remove_columns(["__drop__"])
    return mapped

def main() -> None:
    args = parse_args(); logger = setup_logging(); ds = load_from_dir(Path(args.in_dir))
    out_splits = {}
    for name, split_ds in get_splits(ds).items():
        out_splits[name] = _map_split(split_ds, args.keep_other_fields)
        logger.info(f"split={name} out_rows={out_splits[name].num_rows}")
    out = DatasetDict(out_splits) if len(out_splits) > 1 else next(iter(out_splits.values()))
    save_to_dir(out, Path(args.out_dir)); logger.info(f"Saved: {args.out_dir}")

if __name__ == '__main__':
    main()
