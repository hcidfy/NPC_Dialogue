from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY312 = r"D:\anaconda\envs\py312\python.exe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the GameDialogue workflow from EDA to training/merge/inference."
    )
    p.add_argument("--skip-eda", action="store_true", help="Skip EDA scripts.")
    p.add_argument("--skip-clean", action="store_true", help="Skip data cleaning scripts.")
    p.add_argument("--skip-train", action="store_true", help="Skip LoRA training.")
    p.add_argument("--skip-merge", action="store_true", help="Skip LoRA merge.")
    p.add_argument("--chat", action="store_true", help="Launch interactive chat after merge.")
    return p.parse_args()


def run_step(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    env = os.environ.copy()
    py_root = Path(PY312).resolve().parent
    env_prefix = py_root
    env_path_parts = [
        str(env_prefix),
        str(env_prefix / "Library" / "bin"),
        str(env_prefix / "Scripts"),
    ]
    env["PATH"] = ";".join(env_path_parts + [env.get("PATH", "")])
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def main() -> None:
    args = parse_args()

    eda_steps = [
        [PY312, "GameDialogue/02_analysis/dataset_statistics.py"],
        [PY312, "GameDialogue/02_analysis/tokenizer_statistics.py"],
        [PY312, "GameDialogue/02_analysis/length_distribution.py"],
    ]
    clean_steps = [
        [PY312, "GameDialogue/03_clean/remove_null.py"],
        [PY312, "GameDialogue/03_clean/normalize_text.py"],
        [PY312, "GameDialogue/03_clean/remove_duplicate.py"],
        [PY312, "GameDialogue/03_clean/convert_chatml.py"],
        [PY312, "GameDialogue/03_clean/split_dataset.py"],
    ]
    train_step = [
        PY312,
        "GameDialogue/04_training/train_lora.py",
        "--fp16",
        "--gradient_checkpointing",
        "--max_length",
        "512",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "16",
    ]
    merge_step = [PY312, "GameDialogue/04_training/merge_lora.py"]
    chat_step = [
        PY312,
        "GameDialogue/05_inference/chat.py",
        "--model_path",
        "GameDialogue/models/Qwen3-4B-Instruct-merged",
    ]

    if not args.skip_eda:
        for step in eda_steps:
            run_step(step)

    if not args.skip_clean:
        for step in clean_steps:
            run_step(step)

    if not args.skip_train:
        run_step(train_step)

    if not args.skip_merge:
        run_step(merge_step)

    if args.chat:
        run_step(chat_step)


if __name__ == "__main__":
    main()
