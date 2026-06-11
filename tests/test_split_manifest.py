from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from mcblockclf.data import REQUIRED_MANIFEST_COLUMNS, prepare_dataset
from mcblockclf.utils import load_json


def _make_mock_dataset(raw_dir: Path) -> None:
    for class_index, class_name in enumerate(["andesite", "oak_planks", "stone"]):
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(5):
            color = (
                (class_index * 80 + image_index * 7) % 255,
                (class_index * 40 + image_index * 19) % 255,
                (class_index * 25 + image_index * 31) % 255,
            )
            Image.new("RGB", (24, 24), color=color).save(class_dir / f"image_{image_index}.png")


def test_manifest_columns_no_overlap_and_split_coverage(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_mock_dataset(raw_dir)
    out_dir = tmp_path / "processed"
    manifest_dir = tmp_path / "manifests"

    prepare_dataset(raw_dir=raw_dir, out_dir=out_dir, manifest_dir=manifest_dir, seed=42)
    manifest = pd.read_csv(manifest_dir / "split_manifest.csv")

    assert list(manifest.columns) == REQUIRED_MANIFEST_COLUMNS
    assert not manifest["relative_path"].duplicated().any()
    for class_name in manifest["class_name"].unique():
        class_splits = set(manifest.loc[manifest["class_name"] == class_name, "split"])
        assert class_splits == {"train", "val", "test"}


def test_class_indices_are_deterministic(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_mock_dataset(raw_dir)
    first_manifest_dir = tmp_path / "manifests_first"
    second_manifest_dir = tmp_path / "manifests_second"

    prepare_dataset(raw_dir=raw_dir, out_dir=tmp_path / "processed_first", manifest_dir=first_manifest_dir, seed=42)
    prepare_dataset(raw_dir=raw_dir, out_dir=tmp_path / "processed_second", manifest_dir=second_manifest_dir, seed=42)

    first_mapping = load_json(first_manifest_dir / "class_to_idx.json")
    second_mapping = load_json(second_manifest_dir / "class_to_idx.json")
    assert first_mapping == second_mapping
    assert first_mapping == {"andesite": 0, "oak_planks": 1, "stone": 2}
