"""Helpers to load data resources in this package.

It's a mess in Python, but see here for recommendations and examples:

- https://stackoverflow.com/a/58941536/3519145
- https://github.com/wimglenn/resources-example

Having importlib_resources means we can use the same APIs as those natively
available in Python>=3.9.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files

from ..utils import StanceTopic, Task

DATA_DIR = Path("datasets")
PRED_DIR = Path("predictions")
LABEL_FNM = "{}/test_labels.txt".format
PRED_FNM = "{}.txt".format
"""Convenient pre-defined paths."""


def relpath(path: Path):
    return files(__package__) / path


def open(path: Path, mode="r", *args, **kwargs):
    return relpath(path).open(mode, *args, **kwargs)


def read_text(path: Path):
    return relpath(path).read_text()


def read_lines(path: Path) -> Iterable[str]:
    """Lazily read lines in a local text file."""
    with open(path) as file:
        for line in file:
            yield line.strip()
