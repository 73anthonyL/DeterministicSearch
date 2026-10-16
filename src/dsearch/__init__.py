"""dsearch — local-first semantic evidence search for academic PDFs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deterministic-search")
except PackageNotFoundError:  # A checkout that was never installed.
    __version__ = "0+unknown"
