"""Evaluate a trained checkpoint on the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcblockclf.data import build_dataloaders, load_class_mappings
from mcblockclf.metrics import (
    classification_report_dataframe,
    compute_classification_metrics,
    confusion_matrix_dataframe,
)
from mcblockclf.models import build_model
from mcblockclf.paths import resolve_path
from mcblockclf.train_loop import evaluate_epoch
from mcblockclf.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint on the MiDaS test split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt or last_model.pt.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Processed ImageFolder directory.")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"), help="Manifest directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/metrics"), help="Metrics output directory.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size; defaults to checkpoint config.")
    parser.add_argument("--image-size", type=int, default=None, help="Input image size; defaults to checkpoint config.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count; defaults to checkpoint config.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, or mps.")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _normalize_idx_to_class(raw: dict[str, str] | dict[int, str]) -> dict[int, str]:
    return {int(index): class_name for index, class_name in raw.items()}


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    processed_dir = resolve_path(args.processed_dir)
    manifest_dir = resolve_path(args.manifest_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    device = select_device(args.device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    image_size = args.image_size or int(config.get("data", {}).get("image_size", 224))
    batch_size = args.batch_size or int(config.get("data", {}).get("batch_size", 64))
    num_workers = args.num_workers if args.num_workers is not None else int(config.get("data", {}).get("num_workers", 0))
    seed = int(config.get("project", {}).get("seed", 42))

    manifest_class_to_idx, manifest_idx_to_class = load_class_mappings(manifest_dir)
    checkpoint_class_to_idx = checkpoint.get("class_to_idx")
    checkpoint_idx_to_class = _normalize_idx_to_class(checkpoint.get("idx_to_class", {}))
    if checkpoint_class_to_idx != manifest_class_to_idx:
        raise ValueError("Checkpoint class_to_idx does not match manifest class_to_idx.")
    if checkpoint_idx_to_class != manifest_idx_to_class:
        raise ValueError("Checkpoint idx_to_class does not match manifest idx_to_class.")

    _train_loader, _val_loader, test_loader, loader_class_to_idx = build_dataloaders(
        processed_dir=processed_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    if loader_class_to_idx != manifest_class_to_idx:
        raise ValueError("Processed data class mapping does not match manifest mapping.")

    class_names = [manifest_idx_to_class[index] for index in range(len(manifest_idx_to_class))]
    model = build_model(
        model_name=str(checkpoint["model_name"]),
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    first_images, _ = next(iter(test_loader))
    with torch.no_grad():
        output_dim = model(first_images[:1].to(device)).shape[1]
    if output_dim != len(class_names):
        raise ValueError(f"Model output dimension {output_dim} does not match number of classes {len(class_names)}.")

    criterion = nn.CrossEntropyLoss()
    test_metrics_epoch, y_true, y_pred, y_scores = evaluate_epoch(
        model, test_loader, criterion, device, use_amp=False, split_name="test"
    )
    metrics = compute_classification_metrics(y_true, y_pred, y_scores, class_names)
    metrics.update(
        {
            "test_loss": test_metrics_epoch.loss,
            "test_accuracy": test_metrics_epoch.accuracy,
            "checkpoint": str(checkpoint_path),
            "model_name": checkpoint["model_name"],
            "num_classes": len(class_names),
            "class_names": class_names,
            "image_size": image_size,
        }
    )
    summary_path = manifest_dir / "dataset_summary.json"
    if summary_path.exists():
        dataset_summary = load_json(summary_path)
        dataset_summary["image_size"] = image_size
        metrics["dataset_summary"] = dataset_summary

    save_json(metrics, out_dir / "test_metrics.json")
    report = classification_report_dataframe(y_true, y_pred, class_names)
    report.to_csv(out_dir / "classification_report.csv", index=False)
    confusion = confusion_matrix_dataframe(metrics["confusion_matrix"], class_names)
    confusion.to_csv(out_dir / "confusion_matrix.csv")

    dataset = test_loader.dataset
    paths = [Path(sample_path).relative_to(processed_dir).as_posix() for sample_path, _ in dataset.samples]
    top5_indices = np.argsort(y_scores, axis=1)[:, ::-1][:, : min(5, len(class_names))]
    prediction_rows = []
    for row_index, path in enumerate(paths):
        pred_idx = int(y_pred[row_index])
        true_idx = int(y_true[row_index])
        top5_labels = [class_names[int(index)] for index in top5_indices[row_index]]
        prediction_rows.append(
            {
                "path": path,
                "true_idx": true_idx,
                "true_label": class_names[true_idx],
                "pred_idx": pred_idx,
                "pred_label": class_names[pred_idx],
                "confidence": float(y_scores[row_index, pred_idx]),
                "correct": bool(pred_idx == true_idx),
                "top5_labels": ";".join(top5_labels),
            }
        )
    pd.DataFrame(prediction_rows).to_csv(out_dir / "predictions_test.csv", index=False)

    run_metrics_path = checkpoint_path.parent / "metrics_test.json"
    if checkpoint_path.parent.exists():
        save_json(metrics, run_metrics_path)

    print(f"Metrics written to: {out_dir}")
    print("Next command:")
    print(f"python scripts/make_figures.py --run-dir {checkpoint_path.parent} --metrics-dir {out_dir} --manifest-dir {manifest_dir} --out-dir reports/figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
