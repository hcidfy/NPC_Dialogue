from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from GameDialogue.config import DEFAULT_MODEL_ID, get_paths
from GameDialogue.io_utils import setup_logging

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--local_dir", type=str, default=str(get_paths().models_base))
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--token", type=str, default=None)
    p.add_argument("--token_file", type=str, default=str(get_paths().root / ".secrets" / "hf_token.txt"))
    return p.parse_args()

def resolve_token(cli_token: str | None, token_file: str | None) -> str | None:
    if cli_token:
        return cli_token.strip()
    for k in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"):
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()
    if token_file:
        p = Path(token_file)
        if p.exists():
            t = p.read_text(encoding="utf-8").strip()
            return t or None
    return None

def is_model_dir_complete(model_dir: Path) -> bool:
    has_config = (model_dir / "config.json").exists()
    has_tokenizer = (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer.model").exists()
    has_weights = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
    has_sharded_weights = any(model_dir.glob("model-*.safetensors"))
    return has_config and has_tokenizer and (has_weights or has_sharded_weights)

def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    logger = setup_logging()
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model: {args.model_id} -> {local_dir}")
    token = resolve_token(args.token, args.token_file)
    snapshot_download(repo_id=args.model_id, local_dir=str(local_dir), revision=args.revision, token=token)
    if is_model_dir_complete(local_dir):
        logger.info("Done. Model files look complete.")
        return
    raise RuntimeError("Model directory exists but looks incomplete. This usually means the HF download timed out or the remote repo could not be reached. Please retry.")

if __name__ == "__main__":
    main()
