"""Path helpers for script-based workflows."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a possibly relative path against the project root by default."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base or PROJECT_ROOT) / candidate
