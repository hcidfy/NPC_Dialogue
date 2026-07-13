from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from transformers import pipeline

from GameDialogue.config import DEFAULT_MODEL_ID, get_paths
from GameDialogue.io_utils import now_ts, save_json, setup_logging

DEFAULT_PROMPTS = [
    "请你扮演旅馆老板，向我推荐今晚的房间。",
    "我是一名新手冒险者，第一次进森林，有什么提醒？",
    "最近城门口的守卫为什么变得紧张？",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default=str(get_paths().models_base) if get_paths().models_base.exists() else DEFAULT_MODEL_ID)
    p.add_argument("--tuned_model", type=str, default=str(get_paths().models_merged))
    p.add_argument("--out_dir", type=str, default=str(get_paths().outputs / "logs"))
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()


def _extract(out: object) -> str:
    if isinstance(out, list) and out and isinstance(out[0], dict):
        gen = out[0].get("generated_text")
        return str(gen[-1].get("content", "") if isinstance(gen, list) and gen else gen).strip()
    return str(out).strip()


def main() -> None:
    args = parse_args()
    logger = setup_logging(get_paths().outputs_logs / "compare.log")
    base_pipe = pipeline("text-generation", model=args.base_model)
    tuned_pipe = pipeline("text-generation", model=args.tuned_model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = now_ts()

    rows = []
    for prompt in DEFAULT_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        rows.append(
            {
                "prompt": prompt,
                "base": _extract(base_pipe(messages, max_new_tokens=args.max_new_tokens)),
                "tuned": _extract(tuned_pipe(messages, max_new_tokens=args.max_new_tokens)),
            }
        )
        logger.info("one prompt done")

    md_path = out_dir / f"compare_{tag}.md"
    lines = ["# Compare Before/After", ""]
    for i, row in enumerate(rows, 1):
        lines += [
            f"## Case {i}",
            "",
            f"**Prompt**\n\n{row['prompt']}",
            "",
            "**Base**\n\n" + row["base"],
            "",
            "**Tuned**\n\n" + row["tuned"],
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    save_json(out_dir / f"compare_{tag}.json", {"base_model": args.base_model, "tuned_model": args.tuned_model, "rows": rows})


if __name__ == "__main__":
    main()
