"""General project utilities."""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_json_serializable(data: Any) -> Any:
    """Convert common runtime objects into JSON-serializable values."""
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {str(key): make_json_serializable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [make_json_serializable(value) for value in data]
    if hasattr(data, "item") and callable(data.item):
        try:
            return data.item()
        except ValueError:
            pass
    return data


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON with stable formatting."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(make_json_serializable(data), file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_environment_json(path: str | Path, seed: int, command_args: dict[str, Any]) -> None:
    """Write Python, PyTorch, CUDA, OS, seed, and CLI metadata for a run."""
    import torch

    try:
        import torchvision
    except ImportError:  # pragma: no cover - torchvision is a project dependency.
        torchvision = None

    cuda_available = torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    env = {
        "created_at_utc": utc_timestamp(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torchvision_version": getattr(torchvision, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "mps_available": bool(getattr(torch.backends, "mps", None))
        and torch.backends.mps.is_available(),
        "seed": seed,
        "argv": sys.argv,
        "command_args": command_args,
    }
    save_json(env, path)
