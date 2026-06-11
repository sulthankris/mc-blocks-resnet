"""Create deterministic sample-prediction grids from the test split."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcblockclf.data import imagenet_transforms, load_class_mappings
from mcblockclf.models import build_model
from mcblockclf.paths import resolve_path
from mcblockclf.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample prediction and misclassification grids.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt or last_model.pt.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Processed ImageFolder directory.")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"), help="Manifest directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"), help="Output directory for prediction figures.")
    parser.add_argument("--num-samples", type=int, default=24, help="Number of examples in each grid.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sample seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Prediction batch size; defaults to checkpoint config.")
    parser.add_argument("--image-size", type=int, default=None, help="Input image size; defaults to checkpoint config.")
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


@torch.no_grad()
def predict_dataset(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[list[float]] = []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        y_true.extend(targets.tolist())
        y_pred.extend(predictions.cpu().tolist())
        y_scores.extend(probabilities.cpu().tolist())
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_scores)


def _short_label(label: str, max_chars: int = 20) -> str:
    return label if len(label) <= max_chars else f"{label[: max_chars - 1]}..."


def image_brightness(path: str) -> float:
    """Estimate image brightness from a small grayscale thumbnail."""
    with Image.open(path).convert("L") as image:
        image.thumbnail((32, 32))
        return float(np.asarray(image, dtype=np.float32).mean())


def select_bright_examples(
    candidate_indices: list[int],
    samples: list[tuple[str, int]],
    n_items: int,
    rng: random.Random,
) -> list[int]:
    """Select visible examples from the brightest half of candidates."""
    if not candidate_indices or n_items <= 0:
        return []
    scored = [(image_brightness(samples[index][0]), index) for index in candidate_indices]
    scored.sort(reverse=True)
    bright_pool = [index for _score, index in scored[: max(n_items * 4, n_items)]]
    if len(bright_pool) <= n_items:
        return bright_pool[:n_items]
    return rng.sample(bright_pool, k=n_items)


def render_prediction_grid(
    indices: list[int],
    samples: list[tuple[str, int]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    ensure_dir(out_path.parent)
    if not indices:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Tidak ada contoh yang tersedia.", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    columns = 4
    rows = math.ceil(len(indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.5))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes_array.ravel():
        axis.set_visible(False)

    for axis, sample_index in zip(axes_array.ravel(), indices, strict=False):
        path, _ = samples[sample_index]
        image = Image.open(path).convert("RGB")
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        true_idx = int(y_true[sample_index])
        pred_idx = int(y_pred[sample_index])
        correct = true_idx == pred_idx
        color = "#2f7d32" if correct else "#b23a2f"
        confidence = float(y_scores[sample_index, pred_idx])
        axis.set_title(
            f"Pred: {_short_label(class_names[pred_idx])}\nTrue: {_short_label(class_names[true_idx])}\nConf: {confidence:.2f}",
            fontsize=8,
            color=color,
        )
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        axis.set_visible(True)

    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    processed_dir = resolve_path(args.processed_dir)
    manifest_dir = resolve_path(args.manifest_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    device = select_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    image_size = args.image_size or int(config.get("data", {}).get("image_size", 224))
    batch_size = args.batch_size or int(config.get("data", {}).get("batch_size", 64))

    manifest_class_to_idx, idx_to_class = load_class_mappings(manifest_dir)
    if checkpoint.get("class_to_idx") != manifest_class_to_idx:
        raise ValueError("Checkpoint class mapping does not match manifest mapping.")
    class_names = [idx_to_class[index] for index in range(len(idx_to_class))]

    _train_transform, eval_transform = imagenet_transforms(image_size)
    test_dataset = datasets.ImageFolder(processed_dir / "test", transform=eval_transform)
    if test_dataset.class_to_idx != manifest_class_to_idx:
        raise ValueError("Test ImageFolder class mapping does not match manifest mapping.")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(
        model_name=str(checkpoint["model_name"]),
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    y_true, y_pred, y_scores = predict_dataset(model, test_loader, device)

    all_indices = list(range(len(test_dataset.samples)))
    rng = random.Random(args.seed)
    wrong_indices = [index for index in all_indices if int(y_true[index]) != int(y_pred[index])]
    correct_indices = [index for index in all_indices if int(y_true[index]) == int(y_pred[index])]
    target_wrong = min(max(args.num_samples // 3, 2), len(wrong_indices))
    target_correct = max(0, min(args.num_samples - target_wrong, len(correct_indices)))
    selected_wrong = select_bright_examples(wrong_indices, test_dataset.samples, target_wrong, rng)
    selected_correct = select_bright_examples(correct_indices, test_dataset.samples, target_correct, rng)
    selected = selected_correct + selected_wrong
    rng.shuffle(selected)

    selected_wrong_grid = select_bright_examples(
        wrong_indices,
        test_dataset.samples,
        min(args.num_samples, len(wrong_indices)),
        rng,
    )

    render_prediction_grid(
        selected,
        test_dataset.samples,
        y_true,
        y_pred,
        y_scores,
        class_names,
        out_dir / "sample_predictions.png",
        "Contoh Prediksi Terang: Benar dan Salah",
    )
    render_prediction_grid(
        selected_wrong_grid,
        test_dataset.samples,
        y_true,
        y_pred,
        y_scores,
        class_names,
        out_dir / "misclassified_examples.png",
        "Contoh Salah Klasifikasi pada Set Uji",
    )

    print(f"Prediction figures written to: {out_dir}")
    print("Next command:")
    print("python scripts/build_poster.py --config configs/poster.yaml --metrics reports/metrics/test_metrics.json --figures reports/figures --out-html reports/poster/poster.html --out-pdf reports/poster/poster_A2.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
