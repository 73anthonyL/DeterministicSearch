"""The bundled public-domain sample: where it is and how `demo` uses it.

The PDF ships inside the package (see `data/README.md`), so `dsearch demo`
works from a wheel with no clone and no download beyond the model.
"""

from __future__ import annotations

import importlib.resources
import shutil
from pathlib import Path

SAMPLE_FILENAME = "sample.pdf"
SAMPLE_TITLE = "The Souls of Black Folk"
SAMPLE_AUTHOR = "Du Bois, W. E. B."
SAMPLE_QUERY = "What does it feel like to be seen as a problem?"


def sample_path() -> Path:
    """The packaged PDF. A `Path` in every supported install (wheel, editable, sdist)."""
    resource = importlib.resources.files("dsearch") / "data" / SAMPLE_FILENAME
    with importlib.resources.as_file(resource) as path:
        return Path(path)


def copy_sample(directory: Path) -> Path:
    """Copy the sample into `directory` (created if needed) and return the copy's path.

    A copy rather than the package's own file, so the library records a path
    the reader can see, open, and later re-add after an index-format bump.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / SAMPLE_FILENAME
    source = sample_path()
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, target)
    return target
