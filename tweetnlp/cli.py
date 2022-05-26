"""Command-line interface."""
from pathlib import Path

import typer

from .classify import PretrainedCardiffClassifier
from .eval import evaluate, score
from .utils import Task

CLI = typer.Typer()


@CLI.command()
def score_pred(
    task: Task = typer.Argument(..., case_sensitive=False),
    pred: Path = typer.Argument(..., exists=True, file_okay=True, resolve_path=True),
):
    """Evaluate a predictions file against a task's test labels.

    E.g.: the following should return a score of 1.0 exactly:

        >> tweetnlp score_pred emoji tweeteval/resources/datasets/emoji/test_labels.txt
    """
    with open(pred) as file:
        pred = [line.strip() for line in file.readlines()]

    s = score(pred, task)
    print(s)


@CLI.command()
def score_model(
    task: Task = typer.Argument(..., case_sensitive=False),
    # model: str = typer.Argument,
):
    """Evaluate a model against a task's test labels.

    E.g.
        >> tweetnlp score_model emoji
    """
    model = PretrainedCardiffClassifier(task)
    s = evaluate(model, task, preproc=False)
    print(s)
