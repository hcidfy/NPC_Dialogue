from __future__ import annotations

import sys
import inspect
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from GameDialogue.chatml import format_for_tokenizer
from GameDialogue.config import get_paths
from GameDialogue.io_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=str(get_paths().models_merged))
    p.add_argument("--dataset_dir", type=str, default=str(get_paths().datasets_processed / "split"))
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_eval_rows", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(get_paths().outputs_logs / "evaluate.log")

    ds = load_from_disk(args.dataset_dir)
    if not isinstance(ds, DatasetDict):
        raise ValueError("dataset_dir must be a DatasetDict with train/validation/test")

    eval_ds = ds.get("validation") or ds.get("test")
    if eval_ds is None:
        raise ValueError("No validation/test split found")
    if args.max_eval_rows is not None and eval_ds.num_rows > args.max_eval_rows:
        eval_ds = eval_ds.select(list(range(args.max_eval_rows)))

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_ds = eval_ds.map(
        lambda ex: {"text": format_for_tokenizer(tokenizer, ex["messages"], add_generation_prompt=False)},
        remove_columns=[c for c in eval_ds.column_names if c != "messages"],
        num_proc=1,
    )
    eval_tok = eval_ds.map(
        lambda ex: (lambda enc: {**enc, "labels": enc["input_ids"].copy()})(
            tokenizer(ex["text"], truncation=True, max_length=args.max_length, padding=False, add_special_tokens=False)
        ),
        remove_columns=eval_ds.column_names,
        num_proc=1,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code, dtype=dtype)

    trainer_kwargs = {
        "model": model,
        "args": TrainingArguments(output_dir=str(get_paths().outputs_checkpoints / "eval_tmp"), per_device_eval_batch_size=1),
        "eval_dataset": eval_tok,
        "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    }
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer

    metrics = Trainer(**trainer_kwargs).evaluate()
    loss = float(metrics.get("eval_loss", float("nan")))
    ppl = math.exp(loss) if loss == loss and loss < 100 else float("inf")
    logger.info(f"eval_loss={loss:.6f}, perplexity={ppl:.3f}")


if __name__ == "__main__":
    main()
