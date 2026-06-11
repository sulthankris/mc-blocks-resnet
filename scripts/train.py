"""Train a Minecraft block image classifier."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcblockclf.config import apply_overrides, load_yaml, save_yaml, str_to_bool
from mcblockclf.data import build_dataloaders, load_class_mappings
from mcblockclf.metrics import compute_classification_metrics
from mcblockclf.models import build_model
from mcblockclf.paths import resolve_path
from mcblockclf.seed import seed_everything
from mcblockclf.train_loop import evaluate_epoch, save_checkpoint, train_one_epoch
from mcblockclf.utils import ensure_dir, load_json, save_json, write_environment_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch classifier for MiDaS Minecraft block images.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="YAML config path.")
    parser.add_argument("--model", dest="model_name", default=None, help="Model name: small_cnn or resnet18.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--processed-dir", type=Path, default=None, help="Processed ImageFolder directory.")
    parser.add_argument("--manifest-dir", type=Path, default=None, help="Manifest directory.")
    parser.add_argument("--image-size", type=int, default=None, help="Input image size.")
    parser.add_argument("--pretrained", type=str_to_bool, nargs="?", const=True, default=None, help="Use pretrained weights for supported models.")
    parser.add_argument("--freeze-backbone", type=str_to_bool, nargs="?", const=True, default=None, help="Freeze the pretrained backbone.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--amp", type=str_to_bool, nargs="?", const=True, default=None, help="Use CUDA automatic mixed precision when available.")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    """Select a torch device with an automatic fallback."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(config["training"].get("optimizer", "adamw")).lower()
    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "adam":
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = str(config["training"].get("scheduler", "none")).lower()
    epochs = int(config["training"]["epochs"])
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def setup_logging(run_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "logs.txt", encoding="utf-8")],
        force=True,
    )


def main() -> int:
    args = parse_args()
    base_config = load_yaml(resolve_path(args.config))
    config = apply_overrides(
        base_config,
        {
            "model.name": args.model_name,
            "training.epochs": args.epochs,
            "data.batch_size": args.batch_size,
            "training.lr": args.lr,
            "project.seed": args.seed,
            "runtime.device": args.device,
            "data.processed_dir": str(args.processed_dir) if args.processed_dir is not None else None,
            "data.manifest_dir": str(args.manifest_dir) if args.manifest_dir is not None else None,
            "data.image_size": args.image_size,
            "model.pretrained": args.pretrained,
            "model.freeze_backbone": args.freeze_backbone,
            "data.num_workers": args.num_workers,
            "runtime.amp": args.amp,
        },
    )

    seed = int(config["project"]["seed"])
    seed_everything(seed, deterministic=bool(config["runtime"].get("deterministic", True)))
    processed_dir = resolve_path(config["data"]["processed_dir"])
    manifest_dir = resolve_path(config["data"]["manifest_dir"])
    runs_dir = ensure_dir(resolve_path(config["outputs"].get("runs_dir", "runs")))
    model_name = str(config["model"]["name"])
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"
    run_dir = ensure_dir(runs_dir / run_id)
    setup_logging(run_dir)

    logging.info("Processed data: %s", processed_dir)
    logging.info("Manifest directory: %s", manifest_dir)
    class_to_idx_manifest, idx_to_class = load_class_mappings(manifest_dir)
    train_loader, val_loader, _test_loader, class_to_idx_loader = build_dataloaders(
        processed_dir=processed_dir,
        image_size=int(config["data"]["image_size"]),
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 0)),
        seed=seed,
    )
    if class_to_idx_manifest != class_to_idx_loader:
        raise ValueError("Class mapping mismatch between manifest and processed ImageFolder data.")

    class_names = [idx_to_class[index] for index in range(len(idx_to_class))]
    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=bool(config["model"].get("pretrained", True)),
        freeze_backbone=bool(config["model"].get("freeze_backbone", False)),
    )
    device = select_device(str(config["runtime"].get("device", "auto")))
    model.to(device)
    logging.info("Training %s on %s", model_name, device)

    save_yaml(config, run_dir / "config_resolved.yaml")
    save_json(class_to_idx_manifest, run_dir / "class_to_idx.json")
    save_json({str(key): value for key, value in idx_to_class.items()}, run_dir / "idx_to_class.json")
    if (manifest_dir / "dataset_summary.json").exists():
        dataset_summary = load_json(manifest_dir / "dataset_summary.json")
        dataset_summary["image_size"] = int(config["data"]["image_size"])
        save_json(dataset_summary, run_dir / "dataset_summary.json")
    write_environment_json(run_dir / "environment.json", seed=seed, command_args=vars(args))

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    use_amp = device.type == "cuda" and bool(config["runtime"].get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_macro_f1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    patience = int(config["training"].get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    epochs = int(config["training"]["epochs"])

    try:
        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, epoch)
            val_metrics, y_true, y_pred, y_scores = evaluate_epoch(
                model, val_loader, criterion, device, use_amp=False, split_name=f"epoch {epoch} val"
            )
            val_full_metrics = compute_classification_metrics(y_true, y_pred, y_scores, class_names)
            val_full_metrics.update(
                {"epoch": epoch, "val_loss": val_metrics.loss, "val_accuracy": val_metrics.accuracy}
            )
            save_json(val_full_metrics, run_dir / "metrics_val.json")

            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_macro_f1": val_metrics.macro_f1,
                "lr": current_lr,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

            is_best = float(val_metrics.macro_f1 or 0.0) > best_macro_f1
            if is_best:
                best_macro_f1 = float(val_metrics.macro_f1 or 0.0)
                best_epoch = epoch
                best_metrics = val_full_metrics
                epochs_without_improvement = 0
                save_checkpoint(
                    str(run_dir / "best_model.pt"),
                    model,
                    model_name,
                    len(class_names),
                    class_to_idx_manifest,
                    idx_to_class,
                    config,
                    epoch,
                    best_metrics,
                )
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                str(run_dir / "last_model.pt"),
                model,
                model_name,
                len(class_names),
                class_to_idx_manifest,
                idx_to_class,
                config,
                epoch,
                val_full_metrics,
            )

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(float(val_metrics.macro_f1 or 0.0))
            elif scheduler is not None:
                scheduler.step()

            logging.info(
                "epoch=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f best_epoch=%s",
                epoch,
                train_metrics.loss,
                train_metrics.accuracy,
                val_metrics.loss,
                val_metrics.accuracy,
                val_metrics.macro_f1 or 0.0,
                best_epoch,
            )
            if patience > 0 and epochs_without_improvement >= patience:
                logging.info("Early stopping after %s epochs without validation macro-F1 improvement.", patience)
                break
    except KeyboardInterrupt:
        logging.warning("Training interrupted; saving partial last_model.pt if possible.")
        save_checkpoint(
            str(run_dir / "last_model.pt"),
            model,
            model_name,
            len(class_names),
            class_to_idx_manifest,
            idx_to_class,
            config,
            history[-1]["epoch"] if history else 0,
            best_metrics,
        )
        raise

    print(f"Training complete. Run directory: {run_dir}")
    print("Next command:")
    print(f"python scripts/evaluate.py --checkpoint {run_dir / 'best_model.pt'} --processed-dir {processed_dir} --manifest-dir {manifest_dir} --out-dir reports/metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
