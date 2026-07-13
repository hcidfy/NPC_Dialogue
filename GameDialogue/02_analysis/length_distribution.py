from __future__ import annotations
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np
from transformers import AutoTokenizer
from GameDialogue.chatml import as_messages, ensure_chat_pair, format_for_tokenizer
from GameDialogue.config import get_paths
from GameDialogue.dataset_utils import get_splits, load_raw_dataset
from GameDialogue.io_utils import save_json, setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id_or_path", type=str, default=str(get_paths().models_base))
    p.add_argument("--dataset_dir", type=str, default=str(get_paths().datasets_raw / "chinese_roleplay_novel"))
    p.add_argument("--max_rows", type=int, default=20000)
    p.add_argument("--fig_dir", type=str, default=str(get_paths().outputs_figures))
    p.add_argument("--out_dir", type=str, default=str(get_paths().outputs / "logs"))
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    logger = setup_logging()
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id_or_path, trust_remote_code=args.trust_remote_code, use_fast=True)
    ds = load_raw_dataset(local_dir=Path(args.dataset_dir))
    char_lens, token_lens = [], []
    for _, split_ds in get_splits(ds).items():
        for i in range(min(args.max_rows, split_ds.num_rows)):
            msgs = as_messages(split_ds[i])
            if msgs is None:
                continue
            msgs = ensure_chat_pair(msgs) or msgs
            text = format_for_tokenizer(tokenizer, msgs, add_generation_prompt=False)
            char_lens.append(len(text))
            token_lens.append(len(tokenizer(text, add_special_tokens=False).input_ids))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for values, title, out_path in [(char_lens, 'Char Length Distribution', fig_dir / 'char_length_hist.png'), (token_lens, 'Token Length Distribution', fig_dir / 'token_length_hist.png')]:
        arr = np.asarray(values, dtype=np.int64)
        if arr.size:
            plt.figure(figsize=(8, 5)); plt.hist(arr, bins=50); plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()
    save_json(out_dir / 'length_distribution.json', {'char_count': len(char_lens), 'token_count': len(token_lens)})
    logger.info('Done.')

if __name__ == '__main__':
    main()
