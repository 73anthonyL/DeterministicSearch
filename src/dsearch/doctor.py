"""`dsearch doctor`: the facts needed to debug an install, in one place.

Everything here is read-only and cheap except the device probe, which imports
torch; that is the one slow line and it is wrapped so a broken torch install
is reported rather than fatal.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path

from dsearch import __version__
from dsearch.index import TIERS, home, list_sources, load_config

# Packages whose versions matter when something goes wrong.
PACKAGES = ("pymupdf", "sentence-transformers", "torch", "huggingface-hub", "numpy", "streamlit")


def hf_cache_dir() -> Path:
    """Where the Hugging Face Hub keeps downloaded models."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:  # Old hub versions, or none at all.
        return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def model_cache_folder(model: str) -> str:
    """The hub's folder name for a model: `models--org--name`.

    Bare sentence-transformers names resolve under that organisation.
    """
    repo = model if "/" in model else f"sentence-transformers/{model}"
    return "models--" + repo.replace("/", "--")


def cached_models(cache: Path | None = None) -> dict[str, bool]:
    """Which tiers' models are already on disk, so first-use downloads are predictable."""
    cache = cache or hf_cache_dir()
    return {tier: (cache / model_cache_folder(model)).is_dir() for tier, model in TIERS.items()}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def compute_device() -> str:
    """The device torch would pick: cuda, mps, or cpu. 'unavailable' if torch cannot import."""
    try:
        import torch
    except Exception as exc:  # ImportError, or a broken binary raising OSError.
        return f"unavailable ({type(exc).__name__})"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def directory_size(path: Path) -> int:
    """Total bytes under `path`; 0 if it does not exist."""
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def format_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def diagnostics(*, probe_device: bool = True) -> dict:
    """Everything `dsearch doctor` reports, as plain data for both renderers."""
    library = home()
    sources = list_sources() if library.is_dir() else []
    return {
        "dsearch": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "library": {
            "path": str(library),
            "exists": library.is_dir(),
            "sources": len(sources),
            "bytes": directory_size(library),
            "measured_rates": load_config().get("pages_per_second", {}),
        },
        "models": {
            "cache": str(hf_cache_dir()),
            "tiers": {
                tier: {"model": model, "cached": cached_models()[tier]}
                for tier, model in TIERS.items()
            },
            "offline": os.environ.get("HF_HUB_OFFLINE") in {"1", "true", "True", "yes"},
            "device": compute_device() if probe_device else "not probed",
        },
        "packages": {name: package_version(name) for name in PACKAGES},
        "ui_available": importlib.util.find_spec("streamlit") is not None,
    }
