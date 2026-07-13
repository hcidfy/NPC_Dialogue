from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from transformers import pipeline

from GameDialogue.config import DEFAULT_MODEL_ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--system", type=str, default="你是一个游戏 NPC，请用自然、简短的方式与玩家对话。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = pipeline("text-generation", model=args.model_path)
    pipe.model.generation_config.max_new_tokens = args.max_new_tokens
    pipe.model.generation_config.max_length = None

    messages = [{"role": "system", "content": args.system}]
    while True:
        user = input("User> ").strip()
        if not user:
            break
        messages.append({"role": "user", "content": user})
        out = pipe(messages, temperature=args.temperature, top_p=args.top_p)
        if isinstance(out, list) and out and isinstance(out[0], dict):
            gen = out[0].get("generated_text")
            assistant = gen[-1].get("content", "") if isinstance(gen, list) and gen else str(gen)
        else:
            assistant = str(out)
        assistant = assistant.strip()
        print(f"Assistant> {assistant}")
        messages.append({"role": "assistant", "content": assistant})


if __name__ == "__main__":
    main()
