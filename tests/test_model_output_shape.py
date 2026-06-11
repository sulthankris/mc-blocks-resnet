from __future__ import annotations

import torch

from mcblockclf.models import build_model


def test_small_cnn_output_shape() -> None:
    model = build_model("small_cnn", num_classes=60)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 60)


def test_resnet18_output_shape_without_download() -> None:
    model = build_model("resnet18", num_classes=60, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 60)
