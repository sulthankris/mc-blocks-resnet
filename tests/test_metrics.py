from __future__ import annotations

import numpy as np

from mcblockclf.metrics import compute_classification_metrics


def test_metric_computation_returns_required_keys() -> None:
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    scores = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.9, 0.05, 0.05],
            [0.1, 0.2, 0.7],
        ]
    )
    metrics = compute_classification_metrics(y_true, y_pred, scores, ["a", "b", "c"])

    assert "accuracy_top1" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "confusion_matrix" in metrics
    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
