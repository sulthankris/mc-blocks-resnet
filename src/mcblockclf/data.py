"""Dataset preparation and PyTorch data loading."""

from __future__ import annotations

import hashlib
import random
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mcblockclf.utils import ensure_dir, load_json, save_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
REQUIRED_MANIFEST_COLUMNS = ["relative_path", "class_name", "class_idx", "split", "sha256"]


def is_image_file(path: Path) -> bool:
    """Return True when a path has a supported image extension."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_class_images(raw_dir: Path) -> dict[str, list[Path]]:
    """Discover image files grouped by their immediate parent directory name.

    This supports both ``raw/class_name/image.png`` and already split-looking
    layouts such as ``raw/train/class_name/image.png``. The script ignores the
    existing split labels and creates its own deterministic split.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist: {raw_dir}. Run scripts/download_midas.py "
            "or place the extracted MiDaS folder under data/raw/."
        )

    grouped: dict[str, list[Path]] = defaultdict(list)
    image_dirs: dict[Path, list[Path]] = defaultdict(list)
    for path in sorted(raw_dir.rglob("*")):
        if is_image_file(path):
            image_dirs[path.parent].append(path)

    for image_dir, files in image_dirs.items():
        class_name = image_dir.name
        if class_name.startswith(".") or class_name.lower() in {"raw", "images", "img"}:
            continue
        grouped[class_name].extend(sorted(files))

    if len(grouped) < 2:
        raise ValueError(
            "Could not discover at least two image classes in raw data. Expected a layout like "
            "data/raw/MiDaS-60_small/<class_name>/*.png. If the dataset is missing, open "
            "https://osf.io/whgy6/, download MiDaS-60_small or MiDaS-60_large, extract it "
            "under data/raw/, then rerun this command."
        )
    return {class_name: sorted(paths) for class_name, paths in grouped.items()}


def _stable_class_seed(seed: int, class_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{class_name}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _split_counts(n_items: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n_items < 3:
        raise ValueError("Each class must contain at least three images for train/val/test splits.")
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.6f}")

    train_count = max(1, int(n_items * train_ratio))
    val_count = max(1, int(n_items * val_ratio))
    test_count = n_items - train_count - val_count

    while test_count < 1:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            raise ValueError("Could not allocate at least one sample per split.")
        test_count = n_items - train_count - val_count

    return train_count, val_count, test_count


def _copy_split_file(source: Path, destination_dir: Path, index: int, digest: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{index:05d}_{digest[:12]}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return destination


def prepare_dataset(
    raw_dir: Path,
    out_dir: Path,
    manifest_dir: Path,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_per_class: int = 0,
) -> dict[str, Any]:
    """Create deterministic train/validation/test splits and save manifests."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    manifest_dir = Path(manifest_dir)
    class_images = discover_class_images(raw_dir)
    class_names = sorted(class_images)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {str(idx): class_name for class_name, idx in class_to_idx.items()}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(manifest_dir)

    rows: list[dict[str, Any]] = []
    hash_to_split: dict[str, str] = {}
    split_counts = {split: 0 for split in SPLIT_NAMES}
    class_counts: dict[str, dict[str, int]] = {}
    duplicate_counts_by_class: dict[str, int] = {}

    for class_name in class_names:
        digest_to_path: dict[str, Path] = {}
        for source in class_images[class_name]:
            digest = sha256_file(source)
            if digest not in digest_to_path:
                digest_to_path[digest] = source
        duplicate_counts_by_class[class_name] = len(class_images[class_name]) - len(digest_to_path)
        paths = list(digest_to_path.values())
        source_to_digest = {source: digest for digest, source in digest_to_path.items()}
        rng = random.Random(_stable_class_seed(seed, class_name))
        rng.shuffle(paths)
        if max_per_class and max_per_class > 0:
            paths = paths[:max_per_class]

        train_count, val_count, test_count = _split_counts(
            len(paths), train_ratio, val_ratio, test_ratio
        )
        split_files = {
            "train": paths[:train_count],
            "val": paths[train_count : train_count + val_count],
            "test": paths[train_count + val_count : train_count + val_count + test_count],
        }
        class_counts[class_name] = {split: len(files) for split, files in split_files.items()}
        class_counts[class_name]["total"] = sum(class_counts[class_name].values())

        for split, files in split_files.items():
            if not files:
                raise ValueError(f"Class {class_name!r} has no samples in split {split!r}.")
            split_counts[split] += len(files)
            for local_index, source in enumerate(files):
                digest = source_to_digest[source]
                previous_split = hash_to_split.get(digest)
                if previous_split is not None and previous_split != split:
                    raise ValueError(
                        "Duplicate image content would appear across splits: "
                        f"{source} has SHA-256 {digest} already assigned to {previous_split}."
                    )
                hash_to_split[digest] = split
                destination = _copy_split_file(
                    source,
                    out_dir / split / class_name,
                    local_index,
                    digest,
                )
                rows.append(
                    {
                        "relative_path": destination.relative_to(out_dir).as_posix(),
                        "class_name": class_name,
                        "class_idx": class_to_idx[class_name],
                        "split": split,
                        "sha256": digest,
                    }
                )

    manifest = pd.DataFrame(rows, columns=REQUIRED_MANIFEST_COLUMNS)
    _validate_manifest(manifest, class_to_idx)

    manifest.to_csv(manifest_dir / "split_manifest.csv", index=False)
    save_json(class_to_idx, manifest_dir / "class_to_idx.json")
    save_json(idx_to_class, manifest_dir / "idx_to_class.json")

    smallest_class = min(class_counts.items(), key=lambda item: item[1]["total"])
    largest_class = max(class_counts.items(), key=lambda item: item[1]["total"])
    summary = {
        "dataset": "MiDaS",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "raw_dir": str(raw_dir),
        "processed_dir": str(out_dir),
        "manifest_dir": str(manifest_dir),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "max_per_class": max_per_class,
        "num_classes": len(class_names),
        "num_images": int(len(manifest)),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "duplicate_images_removed_by_class": duplicate_counts_by_class,
        "duplicate_images_removed_total": int(sum(duplicate_counts_by_class.values())),
        "smallest_class": {"name": smallest_class[0], "count": smallest_class[1]["total"]},
        "largest_class": {"name": largest_class[0], "count": largest_class[1]["total"]},
    }
    save_json(summary, manifest_dir / "dataset_summary.json")
    return summary


def _validate_manifest(manifest: pd.DataFrame, class_to_idx: dict[str, int]) -> None:
    missing_columns = set(REQUIRED_MANIFEST_COLUMNS) - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

    duplicate_paths = manifest["relative_path"].duplicated()
    if duplicate_paths.any():
        duplicated = manifest.loc[duplicate_paths, "relative_path"].tolist()
        raise ValueError(f"Duplicate files in manifest: {duplicated[:5]}")

    for class_name, expected_idx in class_to_idx.items():
        class_rows = manifest[manifest["class_name"] == class_name]
        observed_idx = set(class_rows["class_idx"].tolist())
        if observed_idx != {expected_idx}:
            raise ValueError(f"Class index mismatch for {class_name}: {observed_idx} != {expected_idx}")
        observed_splits = set(class_rows["split"].tolist())
        if observed_splits != set(SPLIT_NAMES):
            raise ValueError(f"Class {class_name} does not appear in every split: {observed_splits}")

    cross_split_hashes = manifest.groupby("sha256")["split"].nunique()
    bad_hashes = cross_split_hashes[cross_split_hashes > 1]
    if not bad_hashes.empty:
        raise ValueError(
            "Duplicate image content appears across splits for SHA-256 values: "
            f"{bad_hashes.index[:5].tolist()}"
        )


def load_class_mappings(manifest_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load class mappings from a manifest directory."""
    class_to_idx = load_json(Path(manifest_dir) / "class_to_idx.json")
    idx_to_class_raw = load_json(Path(manifest_dir) / "idx_to_class.json")
    idx_to_class = {int(idx): class_name for idx, class_name in idx_to_class_raw.items()}
    return class_to_idx, idx_to_class


def imagenet_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Build train and evaluation transforms using ImageNet normalization."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def build_dataloaders(
    processed_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Build train, validation, and test dataloaders from processed ImageFolder data."""
    processed_dir = Path(processed_dir)
    train_transform, eval_transform = imagenet_transforms(image_size)

    datasets_by_split = {
        "train": datasets.ImageFolder(processed_dir / "train", transform=train_transform),
        "val": datasets.ImageFolder(processed_dir / "val", transform=eval_transform),
        "test": datasets.ImageFolder(processed_dir / "test", transform=eval_transform),
    }
    class_to_idx = datasets_by_split["train"].class_to_idx
    for split, dataset in datasets_by_split.items():
        if dataset.class_to_idx != class_to_idx:
            raise ValueError(f"Class mapping mismatch in {split} split.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        datasets_by_split["train"],
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(datasets_by_split["val"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets_by_split["test"], shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, class_to_idx
