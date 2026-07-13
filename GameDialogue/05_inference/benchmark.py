from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from transformers import AutoTokenizer, pipeline

from GameDialogue.config import DEFAULT_MODEL_ID, get_paths
from GameDialogue.io_utils import save_json, setup_logging

DEFAULT_PROMPTS = [
    "你好，陌生人。你是谁？",
    "我想买点药水，你这里有什么？",
    "最近镇上发生了什么奇怪的事吗？",
    "给我一个去城堡的路线建议。",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--out_path", type=str, default=str(get_paths().outputs / "logs" / "benchmark.json"))
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()


def _extract(out: object) -> str:
    if isinstance(out, list) and out and isinstance(out[0], dict):
        gen = out[0].get("generated_text")
        return str(gen[-1].get("content", "") if isinstance(gen, list) and gen else gen).strip()
    return str(out).strip()


def main() -> None:
    args = parse_args()
    logger = setup_logging(get_paths().outputs_logs / "benchmark.log")
    pipe = pipeline("text-generation", model=args.model_path)
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    rows = []
    for prompt in DEFAULT_PROMPTS:
        t0 = time.perf_counter()
        out = pipe([{"role": "user", "content": prompt}], max_new_tokens=args.max_new_tokens)
        t1 = time.perf_counter()
        assistant = _extract(out)
        gen_tokens = len(tok(assistant, add_special_tokens=False).input_ids)
        rows.append(
            {
                "prompt": prompt,
                "latency_sec": t1 - t0,
                "generated_tokens": gen_tokens,
                "tokens_per_sec": gen_tokens / max(t1 - t0, 1e-6),
            }
        )
        logger.info(f"prompt_len={len(prompt)} tok={gen_tokens} t={t1 - t0:.3f}s")

    save_json(Path(args.out_path), {"model_path": args.model_path, "rows": rows})


if __name__ == "__main__":
    main()
