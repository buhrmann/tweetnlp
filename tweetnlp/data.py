"""Helpers to load datasets, in numpy, pandas or huggingface formats."""
from __future__ import annotations

from functools import partial
from typing import Iterable

from .resources import DATA_DIR, LABEL_FNM, PRED_DIR, PRED_FNM, read_lines
from .utils import Split, StanceTopic, Task


def label_mapping(task: Task, allow_emoji: bool = True) -> dict:
    """Create a mapping from integer strings to human-readable labels."""
    lines = read_lines(DATA_DIR / task.name / "mapping.txt")
    val_idx = 1 if (task != Task.emoji or allow_emoji) else 2
    return {tokens[0]: tokens[val_idx] for tokens in (line.split("\t") for line in lines)}


def map_labels(labels: Iterable, task: Task) -> list:
    """Map integer string labels to human-readable labels."""
    mapping = label_mapping(task)
    return [mapping.get(label, label) for label in labels]


def dataset(
    task: Task,
    split: Split = Split.train,
    humanize: bool = False,
) -> tuple[list, list]:
    """Get texts and corresponding labels for the given task and split."""
    topics = [t.name for t in StanceTopic] if task == Task.stance else [""]
    texts, labels = [], []
    for topic in topics:
        texts.extend(read_lines(DATA_DIR / task.name / topic / f"{split.name}_text.txt"))
        labels.extend(
            read_lines(DATA_DIR / task.name / topic / f"{split.name}_labels.txt")
        )

    if humanize:
        labels = map_labels(labels, task)

    return texts, labels


def labels(task: Task, gold: bool = True, humanize: bool = False):
    """Returns the gold labels or predicted labels for a given task."""

    if task != Task.stance:
        path = DATA_DIR / LABEL_FNM(task.name) if gold else PRED_DIR / PRED_FNM(task.name)
        return list(read_lines(path))

    labels = []
    path = (DATA_DIR if gold else PRED_DIR) / task.name
    for topic in StanceTopic:
        fnm = LABEL_FNM(topic.name) if gold else PRED_FNM(topic.name)
        labels.extend(read_lines(path / fnm))

    if humanize:
        labels = map_labels(labels, task)

    return labels


test_labels = partial(labels, gold=True)
test_preds = partial(labels, gold=False)
