from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import numpy as np
from transformers import AutoTokenizer
from GameDialogue.chatml import as_messages, ensure_chat_pair, format_for_tokenizer
from GameDialogue.config import DEFAULT_MODEL_ID, get_paths
from GameDialogue.dataset_utils import get_splits, load_raw_dataset
from GameDialogue.io_utils import save_json, setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id_or_path", type=str, default=str(get_paths().models_base))
    p.add_argument("--dataset_dir", type=str, default=str(get_paths().datasets_raw / "chinese_roleplay_novel"))
    p.add_argument("--max_rows", type=int, default=20000)
    p.add_argument("--out_dir", type=str, default=str(get_paths().outputs / "logs"))
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    logger = setup_logging()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id_or_path if args.model_id_or_path else DEFAULT_MODEL_ID, trust_remote_code=args.trust_remote_code, use_fast=True)
    ds = load_raw_dataset(local_dir=Path(args.dataset_dir))
    lengths = []
    kept = seen = 0
    for _, split_ds in get_splits(ds).items():
        for i in range(min(args.max_rows, split_ds.num_rows)):
            seen += 1
            msgs = as_messages(split_ds[i])
            if msgs is None:
                continue
            msgs = ensure_chat_pair(msgs) or msgs
            text = format_for_tokenizer(tokenizer, msgs, add_generation_prompt=False)
            lengths.append(len(tokenizer(text, add_special_tokens=False).input_ids))
            kept += 1
    arr = np.asarray(lengths, dtype=np.int64) if lengths else np.asarray([], dtype=np.int64)
    stats = {"seen": seen, "kept": kept, "token_length": {"count": int(arr.size)} if arr.size == 0 else {"count": int(arr.size), "min": int(arr.min()), "p95": float(np.percentile(arr, 95)), "max": int(arr.max()), "mean": float(arr.mean())}}
    save_json(Path(args.out_dir) / 'tokenizer_statistics.json', stats)
    logger.info('Done.')

if __name__ == '__main__':
    main()
