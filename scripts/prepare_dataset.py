"""Prepare deterministic MiDaS train/validation/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcblockclf.data import prepare_dataset
from mcblockclf.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MiDaS images into deterministic class-balanced splits.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing extracted MiDaS files.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output directory for split ImageFolder data.")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"), help="Directory for split manifests and class mappings.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Per-class train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Per-class validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Per-class test ratio.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Limit images per class; 0 uses all images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = resolve_path(args.raw_dir)
    out_dir = resolve_path(args.out_dir)
    manifest_dir = resolve_path(args.manifest_dir)
    print(f"Preparing dataset from: {raw_dir}")
    print(f"Writing processed images to: {out_dir}")
    print(f"Writing manifests to: {manifest_dir}")
    summary = prepare_dataset(
        raw_dir=raw_dir,
        out_dir=out_dir,
        manifest_dir=manifest_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_per_class=args.max_per_class,
    )
    print("Dataset prepared successfully.")
    print(f"Classes: {summary['num_classes']}")
    print(f"Images: {summary['num_images']}")
    print(f"Split counts: {summary['split_counts']}")
    print(f"Smallest class: {summary['smallest_class']}")
    print(f"Largest class: {summary['largest_class']}")
    print("Next command:")
    print("python scripts/train.py --config configs/default.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
