"""Generate report figures from manifests, training history, and metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcblockclf.figures import (
    save_class_distribution,
    save_confusion_matrix,
    save_top_misclassifications,
    save_training_curves,
    save_worst_classes_by_f1,
)
from mcblockclf.paths import resolve_path
from mcblockclf.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Matplotlib figures for evaluation reports and the poster.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing history.csv.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics"), help="Directory containing evaluation CSV/JSON files.")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"), help="Manifest directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"), help="Figure output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    metrics_dir = resolve_path(args.metrics_dir)
    manifest_dir = resolve_path(args.manifest_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir))

    save_class_distribution(manifest_dir / "split_manifest.csv", out_dir / "class_distribution.png")
    save_training_curves(run_dir / "history.csv", out_dir / "training_curves.png")
    save_confusion_matrix(metrics_dir / "confusion_matrix.csv", out_dir / "confusion_matrix.png")
    save_top_misclassifications(metrics_dir / "confusion_matrix.csv", out_dir / "top_misclassifications.png")
    save_worst_classes_by_f1(
        metrics_dir / "classification_report.csv",
        manifest_dir / "idx_to_class.json",
        out_dir / "worst_classes_by_f1.png",
    )
    print(f"Figures written to: {out_dir}")
    print("Next command:")
    print(f"python scripts/predict_examples.py --checkpoint {run_dir / 'best_model.pt'} --processed-dir data/processed --manifest-dir {manifest_dir} --out-dir {out_dir} --num-samples 24 --seed 42")
    return 0


if __name__ == "__main__":
    sys.exit(main())
