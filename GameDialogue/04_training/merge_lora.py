from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from GameDialogue.config import get_paths
from GameDialogue.io_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default=str(get_paths().models_base))
    p.add_argument("--lora_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=str(get_paths().models_merged))
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def _latest_lora_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    cands = [p for p in root.iterdir() if p.is_dir()]
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def main() -> None:
    args = parse_args()
    logger = setup_logging(get_paths().outputs_logs / "merge_lora.log")

    lora_dir = Path(args.lora_dir) if args.lora_dir else _latest_lora_dir(get_paths().models_lora)
    if lora_dir is None:
        raise ValueError("lora_dir is required (or put adapters under GameDialogue/models/lora/)")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code, dtype=dtype)
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model = model.merge_and_unload()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"Merged model saved: {out_dir}")


if __name__ == "__main__":
    main()
