"""Evaluate predictions and models on tweetnlp tasks."""
from __future__ import annotations

from functools import partial
from numbers import Number
from typing import Callable, Iterable

from sklearn.metrics import f1_score, recall_score

from .data import dataset, test_labels
from .utils import Split, Task, preprocess

f1_macro = partial(f1_score, average="macro")
recall_macro = partial(recall_score, average="macro")


def f1_mean(true, pred, labels):
    """Macro (mean) F1 of selected classes only."""
    return f1_score(true, pred, average=None, labels=labels).mean()


SCORERS: dict[Task, Callable] = {
    Task.emoji: f1_macro,
    Task.emotion: f1_macro,
    Task.hate: f1_macro,
    Task.irony: lambda t, p: f1_mean(t, p, labels=["1"] if "1" in t else ["irony"]),
    Task.offensive: f1_macro,
    Task.sentiment: recall_macro,
    Task.stance: lambda t, p: f1_mean(
        t, p, labels=["1", "2"] if "1" in t else ["against", "favor"]
    ),
}
"""Metric to use for each task."""


def score(pred: Iterable, task: Task) -> Number:
    """Return the score for a single task given predictions for test split."""
    labels = test_labels(task)

    if len(pred) != len(labels):
        raise ValueError(
            f"Predictions (n={len(pred)}) don't have correct length "
            f"for selected task (n={len(labels)})!"
        )

    return SCORERS[task](labels, pred)


def evaluate(clf, task: Task, preproc=True):
    """Given a classification model, evaluate it on specified tasks test set."""
    texts, labels = dataset(task, split=Split.test)

    if preproc:
        texts = [preprocess(txt) for txt in texts]

    pred = clf.predict(texts)
    return SCORERS[task](labels, pred)
