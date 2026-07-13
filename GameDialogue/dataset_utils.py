from __future__ import annotations
from pathlib import Path
from typing import Any
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from GameDialogue.config import DEFAULT_DATASET_ID, get_paths

def load_raw_dataset(dataset_id: str = DEFAULT_DATASET_ID, local_dir: Path | None = None, split: str | None = None) -> Dataset | DatasetDict:
    if local_dir is None:
        local_dir = get_paths().datasets_raw / "chinese_roleplay_novel"
    if local_dir.exists():
        ds = load_from_disk(str(local_dir))
    else:
        ds = load_dataset(dataset_id, split=split)
    return ds

def load_from_dir(path: Path) -> Dataset | DatasetDict:
    return load_from_disk(str(path))

def save_to_dir(ds: Dataset | DatasetDict, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))

def get_splits(ds: Dataset | DatasetDict) -> dict[str, Dataset]:
    if isinstance(ds, DatasetDict):
        return dict(ds)
    return {"train": ds}

def get_columns(ds: Dataset | DatasetDict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, split_ds in get_splits(ds).items():
        out[name] = list(split_ds.column_names)
    return out

def pick_example(ds: Dataset | DatasetDict, split: str = "train", i: int = 0) -> dict[str, Any]:
    splits = get_splits(ds)
    if split not in splits:
        split = next(iter(splits.keys()))
    return splits[split][i]
