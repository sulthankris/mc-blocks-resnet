"""Classification metric computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def top_k_accuracy(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute top-k accuracy from class scores or probabilities."""
    if scores.ndim != 2:
        raise ValueError("scores must have shape (n_samples, n_classes)")
    k = min(k, scores.shape[1])
    top_k = np.argsort(scores, axis=1)[:, -k:]
    return float(np.mean(np.any(top_k == y_true.reshape(-1, 1), axis=1)))


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    scores: list[list[float]] | np.ndarray | None,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute required image-classification metrics."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_names)))

    if scores is None:
        score_array = np.zeros((len(y_true_array), len(class_names)), dtype=float)
        score_array[np.arange(len(y_pred_array)), y_pred_array] = 1.0
    else:
        score_array = np.asarray(scores, dtype=float)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        labels=labels,
        average="weighted",
        zero_division=0,
    )[2]
    matrix = confusion_matrix(y_true_array, y_pred_array, labels=labels)

    return {
        "accuracy_top1": float(accuracy_score(y_true_array, y_pred_array)),
        "accuracy_top5": top_k_accuracy(y_true_array, score_array, k=5),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": {
            class_name: float(value) for class_name, value in zip(class_names, precision, strict=True)
        },
        "per_class_recall": {
            class_name: float(value) for class_name, value in zip(class_names, recall, strict=True)
        },
        "per_class_f1": {
            class_name: float(value) for class_name, value in zip(class_names, f1, strict=True)
        },
        "per_class_support": {
            class_name: int(value) for class_name, value in zip(class_names, support, strict=True)
        },
        "confusion_matrix": matrix.tolist(),
    }


def classification_report_dataframe(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Return a scikit-learn classification report as a DataFrame."""
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).transpose().reset_index(names="class_name")
    return frame


def confusion_matrix_dataframe(matrix: list[list[int]], class_names: list[str]) -> pd.DataFrame:
    """Return a labelled confusion matrix DataFrame."""
    return pd.DataFrame(matrix, index=class_names, columns=class_names)
