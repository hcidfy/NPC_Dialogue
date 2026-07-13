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
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from GameDialogue.chatml import format_for_tokenizer
from GameDialogue.config import DEFAULT_MODEL_ID, DEFAULT_SEED, get_paths
from GameDialogue.io_utils import now_ts, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default=str(get_paths().models_base))
    p.add_argument("--dataset_dir", type=str, default=str(get_paths().datasets_processed / "split"))
    p.add_argument("--output_dir", type=str, default=str(get_paths().outputs_checkpoints / "lora"))
    p.add_argument("--lora_dir", type=str, default=str(get_paths().models_lora / f"lora_{now_ts()}"))
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--torch_empty_cache_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--report_to", type=str, default="none")
    return p.parse_args()


def _resolve_model_id_or_path(s: str) -> str:
    p = Path(s)
    if p.exists():
        return str(p)
    return s or DEFAULT_MODEL_ID


def _get_model_dtype(args: argparse.Namespace) -> torch.dtype | None:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def _estimate_warmup_steps(train_size: int, args: argparse.Namespace) -> int:
    effective_batch = max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_size / effective_batch))
    total_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs))
    return int(total_steps * args.warmup_ratio)


def _build_training_args(args: argparse.Namespace, output_dir: Path, has_eval: bool, train_size: int) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": _estimate_warmup_steps(train_size, args),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": args.report_to if args.report_to and args.report_to != "none" else [],
        "seed": args.seed,
        "do_train": True,
        "do_eval": has_eval,
        "save_only_model": True,
    }
    if has_eval:
        kwargs["eval_steps"] = args.eval_steps

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "torch_empty_cache_steps" in params and args.torch_empty_cache_steps > 0:
        kwargs["torch_empty_cache_steps"] = args.torch_empty_cache_steps
    if has_eval:
        if "eval_strategy" in params:
            kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in params:
            kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def _build_trainer(
    model: torch.nn.Module,
    train_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    data_collator,
    tokenizer,
) -> Trainer:
    kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def _align_special_tokens(model, tokenizer) -> None:
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            gen_cfg.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            gen_cfg.bos_token_id = tokenizer.bos_token_id


def main() -> None:
    args = parse_args()
    logger = setup_logging(get_paths().outputs_logs / "train_lora.log")

    base_model = _resolve_model_id_or_path(args.base_model)
    ds = load_from_disk(args.dataset_dir)
    if not isinstance(ds, DatasetDict):
        raise ValueError("dataset_dir must be a DatasetDict with train/validation/test")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(ex: dict) -> dict:
        return {"text": format_for_tokenizer(tokenizer, ex["messages"], add_generation_prompt=False)}

    ds_text = ds.map(
        to_text,
        remove_columns=[c for c in ds["train"].column_names if c != "messages"],
        num_proc=1,
    )

    def tokenize(ex: dict) -> dict:
        enc = tokenizer(
            ex["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds_text.map(tokenize, remove_columns=ds_text["train"].column_names, num_proc=1)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=args.trust_remote_code,
        dtype=_get_model_dtype(args),
        attn_implementation=args.attn_implementation,
    )
    _align_special_tokens(model, tokenizer)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_eval = (not args.no_eval) and "validation" in tokenized and tokenized["validation"].num_rows > 0
    train_args = _build_training_args(
        args,
        output_dir=output_dir,
        has_eval=has_eval,
        train_size=tokenized["train"].num_rows,
    )

    trainer = _build_trainer(
        model=model,
        train_args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,
    )

    logger.info("Start training")
    trainer.train()
    logger.info("Training done")

    lora_dir = Path(args.lora_dir)
    lora_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    logger.info(f"Saved LoRA adapter: {lora_dir}")


if __name__ == "__main__":
    main()
