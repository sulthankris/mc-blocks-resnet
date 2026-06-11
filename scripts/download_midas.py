"""Download or detect the MiDaS dataset.

The official MiDaS data is hosted through OSF. Direct programmatic downloads can
change or require browser interaction, so this script uses a safe manual fallback
instead of guessing unstable URLs.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_existing_images(path: Path) -> bool:
    """Return True when a directory already contains image files."""
    return path.exists() and any(
        child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in path.rglob("*")
    )


def write_source_file(out_dir: Path, variant: str, manual_action: str) -> None:
    """Record dataset provenance in data/raw/SOURCE.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    source_text = f"""Dataset: MiDaS: A Large-Scale Minecraft Dataset for Non-Natural Image Benchmarking
Variant requested: MiDaS-60_{variant}
Source URL: https://osf.io/whgy6/
GitHub: https://github.com/MinecraftDataset/MiDaS
Publication: https://www.raillab.org/publication/torpey-2024-midas/
Recorded at UTC: {datetime.now(UTC).replace(microsecond=0).isoformat()}
Manual action: {manual_action}
"""
    (out_dir / "SOURCE.txt").write_text(source_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect MiDaS data or print manual OSF download instructions.")
    parser.add_argument("--variant", choices=["small", "large"], default="small", help="MiDaS variant to use.")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="Raw data output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out
    print(f"Checking MiDaS data under: {out_dir}")
    if has_existing_images(out_dir):
        write_source_file(out_dir, args.variant, "Existing extracted dataset detected locally.")
        print("Existing image files were detected; SOURCE.txt has been written.")
        print("Next command:")
        print("python scripts/prepare_dataset.py --raw-dir data/raw --out-dir data/processed --manifest-dir data/manifests --seed 42 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    requested_name = f"MiDaS-60_{args.variant}"
    print(
        f"No extracted MiDaS images were found in {out_dir}. Direct OSF download is not attempted because stable file URLs are not documented in the project instructions."
    )
    print("Manual download required:")
    print("1. Open https://osf.io/whgy6/")
    print(f"2. Download {requested_name}")
    print("3. Extract the downloaded folder under data/raw/")
    print("4. Re-run this command, then run scripts/prepare_dataset.py")
    print("This script exits with a nonzero status to avoid silently continuing without data.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
