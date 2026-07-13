from __future__ import annotations
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable
import numpy as np

def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("gamedialogue")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def compact_text(s: str) -> str:
    return " ".join((s or "").split())
