"""Poster rendering utilities."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mcblockclf.config import load_yaml
from mcblockclf.utils import ensure_dir, load_json

FORBIDDEN_POSTER_PHRASES = [
    "di era digital",
    "revolusioner",
    "di-train",
    "meng-classify",
    "feature visual",
    "secara komprehensif dan mendalam",
]


def format_int_id(value: int | float | str | None) -> str:
    """Format an integer using Indonesian thousands separators."""
    if value is None or value == "":
        return "tidak tersedia"
    return f"{int(value):,}".replace(",", ".")


def format_decimal_id(value: float | int | None, digits: int = 3) -> str:
    """Format a decimal number using a comma separator."""
    if value is None:
        return "tidak tersedia"
    return f"{float(value):.{digits}f}".replace(".", ",")


def format_percent_id(value: float | int | None, digits: int = 1) -> str:
    """Format a proportion as an Indonesian percentage."""
    if value is None:
        return "tidak tersedia"
    return f"{float(value) * 100:.{digits}f}%".replace(".", ",")


def validate_poster_language(text: str) -> None:
    """Detect a small set of forbidden AI-style or mixed-language phrases."""
    lowered = text.lower()
    found = [phrase for phrase in FORBIDDEN_POSTER_PHRASES if phrase in lowered]
    if found:
        raise ValueError(f"Poster text contains forbidden phrases from BAHASA-GUIDE.md: {found}")


def _page_dimensions(orientation: str) -> dict[str, Any]:
    if orientation == "portrait":
        return {"page_size": "A2 portrait", "width_mm": 420, "height_mm": 594}
    return {"page_size": "A2 landscape", "width_mm": 594, "height_mm": 420}


def _load_metrics(metrics_path: Path, allow_placeholder: bool) -> dict[str, Any]:
    if metrics_path.exists():
        return load_json(metrics_path)
    if allow_placeholder:
        return {
            "placeholder": True,
            "accuracy_top1": None,
            "macro_f1": None,
            "weighted_f1": None,
            "dataset_summary": {
                "num_classes": 60,
                "num_images": 36000,
                "split_counts": {"train": None, "val": None, "test": None},
                "seed": 42,
            },
        }
    raise FileNotFoundError(
        f"Missing metrics file: {metrics_path}. Run scripts/evaluate.py first, or pass --allow-placeholder for a draft poster."
    )


def _copy_required_figures(
    required_figures: list[str],
    figures_dir: Path,
    assets_dir: Path,
    allow_placeholder: bool,
) -> dict[str, str | None]:
    ensure_dir(assets_dir)
    copied: dict[str, str | None] = {}
    missing: list[str] = []
    for figure_name in required_figures:
        source = figures_dir / figure_name
        destination = assets_dir / figure_name
        if source.exists():
            shutil.copy2(source, destination)
            copied[figure_name] = f"assets/{figure_name}"
        else:
            missing.append(figure_name)
            copied[figure_name] = None
    if missing and not allow_placeholder:
        raise FileNotFoundError(
            "Missing required poster figures: "
            f"{missing}. Run scripts/make_figures.py and scripts/predict_examples.py first."
        )
    return copied


def _top_error_pairs(metrics: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    matrix = metrics.get("confusion_matrix") or []
    class_names = metrics.get("class_names") or []
    pairs: list[dict[str, Any]] = []
    for true_index, row in enumerate(matrix):
        for pred_index, count in enumerate(row):
            if true_index != pred_index and count:
                pairs.append(
                    {
                        "true": class_names[true_index] if true_index < len(class_names) else str(true_index),
                        "pred": class_names[pred_index] if pred_index < len(class_names) else str(pred_index),
                        "count": int(count),
                    }
                )
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:limit]


def _worst_classes(metrics: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    per_class_f1 = metrics.get("per_class_f1") or {}
    rows = [{"class_name": class_name, "f1": float(value)} for class_name, value in per_class_f1.items()]
    return sorted(rows, key=lambda item: item["f1"])[:limit]


def build_poster_context(
    config_path: Path,
    metrics_path: Path,
    figures_dir: Path,
    out_html: Path,
    allow_placeholder: bool = False,
) -> dict[str, Any]:
    """Build the Jinja context for the Indonesian poster."""
    config = load_yaml(config_path)
    poster_cfg = config.get("poster", {})
    rendering_cfg = config.get("rendering", {})
    allow_placeholder = allow_placeholder or bool(rendering_cfg.get("allow_placeholder", False))
    metrics = _load_metrics(metrics_path, allow_placeholder=allow_placeholder)
    dataset_summary = metrics.get("dataset_summary")
    if dataset_summary is None and not allow_placeholder:
        raise ValueError(
            "Metrics file does not contain dataset_summary. Re-run scripts/evaluate.py with the manifest directory."
        )
    dataset_summary = dataset_summary or {}

    required_figures = config.get("assets", {}).get("required_figures", [])
    assets_dir = out_html.parent / "assets"
    figure_paths = _copy_required_figures(
        required_figures=required_figures,
        figures_dir=figures_dir,
        assets_dir=assets_dir,
        allow_placeholder=allow_placeholder,
    )

    split_counts = dataset_summary.get("split_counts", {})
    train_count = split_counts.get("train")
    val_count = split_counts.get("val")
    test_count = split_counts.get("test")
    context = {
        "poster": poster_cfg,
        "metrics": metrics,
        "dataset": dataset_summary,
        "figures": figure_paths,
        "placeholder": bool(metrics.get("placeholder")) or allow_placeholder,
        "page": _page_dimensions(str(poster_cfg.get("orientation", "landscape")).lower()),
        "formatted": {
            "accuracy_top1": format_percent_id(metrics.get("accuracy_top1")),
            "accuracy_top5": format_percent_id(metrics.get("accuracy_top5"), digits=2),
            "macro_f1": format_decimal_id(metrics.get("macro_f1")),
            "weighted_f1": format_decimal_id(metrics.get("weighted_f1")),
            "num_classes": format_int_id(dataset_summary.get("num_classes")),
            "num_images": format_int_id(dataset_summary.get("num_images")),
            "train_count": format_int_id(train_count),
            "val_count": format_int_id(val_count),
            "test_count": format_int_id(test_count),
            "image_size": dataset_summary.get("image_size", "224"),
            "seed": dataset_summary.get("seed", 42),
        },
        "analysis": {
            "top_error_pairs": _top_error_pairs(metrics),
            "worst_classes": _worst_classes(metrics),
        },
    }
    return context


def render_poster_html(context: dict[str, Any], template_dir: Path, out_html: Path) -> None:
    """Render poster HTML from the Jinja template."""
    environment = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = environment.get_template("template.html")
    html = template.render(**context)
    validate_poster_language(html)
    ensure_dir(out_html.parent)
    out_html.write_text(html, encoding="utf-8")


def export_pdf(out_html: Path, out_pdf: Path) -> bool:
    """Export HTML to PDF with WeasyPrint. Return False when fallback is needed."""
    try:
        from weasyprint import HTML
    except ImportError:
        return False

    ensure_dir(out_pdf.parent)
    HTML(filename=str(out_html)).write_pdf(str(out_pdf))
    return True
