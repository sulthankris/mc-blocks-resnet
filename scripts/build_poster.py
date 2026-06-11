"""Render the A2 scientific poster from HTML and local assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcblockclf.paths import resolve_path
from mcblockclf.poster import build_poster_context, export_pdf, render_poster_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Indonesian A2 poster from Jinja HTML and local figures.")
    parser.add_argument("--config", type=Path, default=Path("configs/poster.yaml"), help="Poster YAML config.")
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics/test_metrics.json"), help="Evaluation metrics JSON.")
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"), help="Directory containing required PNG figures.")
    parser.add_argument("--out-html", type=Path, default=Path("reports/poster/poster.html"), help="Output HTML path.")
    parser.add_argument("--out-pdf", type=Path, default=Path("reports/poster/poster_A2.pdf"), help="Output PDF path.")
    parser.add_argument("--allow-placeholder", action="store_true", help="Allow missing metrics/figures and mark the poster as a draft.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    metrics_path = resolve_path(args.metrics)
    figures_dir = resolve_path(args.figures)
    out_html = resolve_path(args.out_html)
    out_pdf = resolve_path(args.out_pdf)
    context = build_poster_context(
        config_path=config_path,
        metrics_path=metrics_path,
        figures_dir=figures_dir,
        out_html=out_html,
        allow_placeholder=args.allow_placeholder,
    )
    render_poster_html(context, PROJECT_ROOT / "poster", out_html)
    print(f"Poster HTML written to: {out_html}")
    try:
        exported = export_pdf(out_html, out_pdf)
    except Exception as exc:  # WeasyPrint can fail because of platform-native libraries.
        exported = False
        print(f"WeasyPrint PDF export failed: {exc}")
    if exported:
        print(f"Poster PDF written to: {out_pdf}")
    else:
        print("PDF was not exported. Fallback instruction:")
        print("Open reports/poster/poster.html in Chrome/Edge and print to PDF using A2 landscape, background graphics enabled, scale 100%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
