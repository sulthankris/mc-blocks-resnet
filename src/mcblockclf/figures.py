"""Matplotlib figure generation for reports and posters."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcblockclf.utils import ensure_dir, load_json

FIGURE_DPI = 300


def _require_file(path: Path, command_hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}. Run: {command_hint}")


def _short_label(label: str, max_chars: int = 18) -> str:
    return label if len(label) <= max_chars else f"{label[: max_chars - 1]}..."


def save_class_distribution(manifest_csv: Path, out_path: Path) -> None:
    """Save a poster-friendly class and split distribution summary."""
    _require_file(manifest_csv, "python scripts/prepare_dataset.py --raw-dir data/raw --out-dir data/processed --manifest-dir data/manifests")
    frame = pd.read_csv(manifest_csv)
    pivot = frame.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    for split in ["train", "val", "test"]:
        if split not in pivot.columns:
            pivot[split] = 0
    pivot = pivot[["train", "val", "test"]]
    totals = pivot.sum(axis=1)
    split_totals = pivot[["train", "val", "test"]].sum()

    fig, (ax_split, ax_class) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.2]})
    colors = ["#3f7d3a", "#d6a53a", "#b23a2f"]
    bars = ax_split.bar(["latih", "validasi", "uji"], split_totals.to_numpy(), color=colors)
    ax_split.set_title("Jumlah Citra per Split", fontsize=15, weight="bold")
    ax_split.set_ylabel("Jumlah citra")
    ax_split.grid(axis="y", color="#d8d2c6", linewidth=0.6, alpha=0.8)
    for bar in bars:
        value = int(bar.get_height())
        ax_split.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}".replace(",", "."), ha="center", va="bottom", fontsize=11, weight="bold")

    ax_class.hist(totals.to_numpy(), bins=8, color="#3f7d3a", edgecolor="white")
    ax_class.axvline(totals.min(), color="#b23a2f", linestyle="--", linewidth=1.5, label=f"min {int(totals.min())}")
    ax_class.axvline(totals.max(), color="#d6a53a", linestyle="--", linewidth=1.5, label=f"maks {int(totals.max())}")
    ax_class.set_title("Sebaran Jumlah Citra per Kelas", fontsize=15, weight="bold")
    ax_class.set_xlabel("Citra per kelas")
    ax_class.set_ylabel("Jumlah kelas")
    ax_class.legend(frameon=False)
    ax_class.grid(axis="y", color="#d8d2c6", linewidth=0.6, alpha=0.8)
    fig.suptitle("Distribusi Dataset Setelah Deduplikasi dan Split", fontsize=17, weight="bold")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_training_curves(history_csv: Path, out_path: Path) -> None:
    """Save loss and validation metric curves."""
    _require_file(history_csv, "python scripts/train.py --config configs/default.yaml")
    history = pd.read_csv(history_csv)
    required = {"epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history.csv is missing columns: {sorted(missing)}")

    best_idx = int(history["val_macro_f1"].idxmax())
    best_epoch = int(history.loc[best_idx, "epoch"])
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 5.2))
    ax_loss.plot(history["epoch"], history["train_loss"], marker="o", linewidth=2.2, color="#3f7d3a", label="loss latih")
    ax_loss.plot(history["epoch"], history["val_loss"], marker="o", linewidth=2.2, color="#b23a2f", label="loss validasi")
    ax_loss.axvline(best_epoch, color="#d6a53a", linestyle="--", linewidth=1.4)
    ax_loss.set_title("Loss Selama Pelatihan", fontsize=15, weight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(frameon=False)
    ax_loss.grid(color="#d8d2c6", linewidth=0.6, alpha=0.8)

    ax_metric.plot(history["epoch"], history["val_accuracy"], marker="o", linewidth=2.2, color="#6f7572", label="akurasi validasi")
    ax_metric.plot(history["epoch"], history["val_macro_f1"], marker="o", linewidth=2.2, color="#d6a53a", label="macro-F1 validasi")
    ax_metric.axvline(best_epoch, color="#b23a2f", linestyle="--", linewidth=1.4, label=f"terbaik epoch {best_epoch}")
    ax_metric.set_title("Metrik Validasi", fontsize=15, weight="bold")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Nilai")
    lower = max(0.0, min(history["val_accuracy"].min(), history["val_macro_f1"].min()) - 0.03)
    ax_metric.set_ylim(lower, 1.005)
    ax_metric.legend(frameon=False)
    ax_metric.grid(color="#d8d2c6", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_confusion_matrix(confusion_csv: Path, out_path: Path) -> None:
    """Save a readable normalized confusion matrix for classes with most errors."""
    _require_file(confusion_csv, "python scripts/evaluate.py --checkpoint runs/<run_id>/best_model.pt --processed-dir data/processed --manifest-dir data/manifests --out-dir reports/metrics")
    frame = pd.read_csv(confusion_csv, index_col=0)
    matrix = frame.to_numpy(dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    labels = list(frame.index)
    row_errors = matrix.sum(axis=1) - np.diag(matrix)
    selected_indices = np.argsort(row_errors)[::-1][:12]
    selected_indices = np.array(sorted(selected_indices, key=lambda idx: labels[idx]))
    selected = normalized[np.ix_(selected_indices, selected_indices)]
    selected_labels = [labels[index] for index in selected_indices]

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    image = ax.imshow(selected, cmap="YlGn", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Proporsi")
    ticks = np.arange(len(selected_labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([_short_label(label, 12) for label in selected_labels], rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels([_short_label(label, 12) for label in selected_labels], fontsize=9)
    ax.set_title("Matriks Konfusi: Kelas dengan Error Terbanyak", fontsize=14, weight="bold")
    ax.set_xlabel("Kelas prediksi")
    ax.set_ylabel("Kelas asli")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_top_misclassifications(confusion_csv: Path, out_path: Path, top_n: int = 10) -> None:
    """Save the largest off-diagonal confusion pairs."""
    _require_file(confusion_csv, "python scripts/evaluate.py --checkpoint runs/<run_id>/best_model.pt --processed-dir data/processed --manifest-dir data/manifests --out-dir reports/metrics")
    frame = pd.read_csv(confusion_csv, index_col=0)
    matrix = frame.to_numpy(dtype=int)
    labels = list(frame.index)
    pairs: list[tuple[str, int]] = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and matrix[i, j] > 0:
                pairs.append((f"{_short_label(true_label, 13)} -> {_short_label(pred_label, 13)}", int(matrix[i, j])))
    pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    if pairs:
        names = [item[0] for item in pairs][::-1]
        values = [item[1] for item in pairs][::-1]
        ax.barh(names, values, color="#b23a2f")
        ax.set_xlabel("Jumlah salah klasifikasi")
        for index, value in enumerate(values):
            ax.text(value + 0.3, index, str(value), va="center", fontsize=10, weight="bold")
    else:
        ax.text(0.5, 0.5, "Tidak ada salah klasifikasi pada set uji.", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    ax.set_title("Pasangan Kelas yang Paling Sering Tertukar", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#d8d2c6", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_worst_classes_by_f1(report_csv: Path, idx_to_class_json: Path, out_path: Path, top_n: int = 10) -> None:
    """Save a bar chart of the lowest per-class F1-scores."""
    _require_file(report_csv, "python scripts/evaluate.py --checkpoint runs/<run_id>/best_model.pt --processed-dir data/processed --manifest-dir data/manifests --out-dir reports/metrics")
    _require_file(idx_to_class_json, "python scripts/prepare_dataset.py --raw-dir data/raw --out-dir data/processed --manifest-dir data/manifests")
    class_names = set(load_json(idx_to_class_json).values())
    report = pd.read_csv(report_csv)
    report = report[report["class_name"].isin(class_names)].copy()
    if "f1-score" not in report.columns:
        raise ValueError("classification_report.csv must contain an f1-score column.")
    report = report.sort_values("f1-score", ascending=True).head(top_n)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    if not report.empty:
        names = [_short_label(name, 18) for name in report["class_name"]][::-1]
        values = report["f1-score"].to_numpy()[::-1]
        ax.barh(names, values, color="#d6a53a")
        lower = max(0.0, min(values) - 0.08)
        ax.set_xlim(lower, 1.0)
        ax.set_xlabel("F1-score")
        for index, value in enumerate(values):
            ax.text(min(value + 0.003, 0.997), index, f"{value:.3f}", va="center", fontsize=9, weight="bold")
    else:
        ax.text(0.5, 0.5, "Laporan per kelas tidak tersedia.", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    ax.set_title("Kelas dengan F1-score Terendah", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#d8d2c6", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
