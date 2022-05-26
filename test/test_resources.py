from pathlib import Path

from tweetnlp import resources


def test_read_lines():
    path = Path("datasets/emoji/test_labels.txt")
    labels = list(resources.read_lines(path))
    assert len(labels) == 50_000
    assert labels[:5] == ["2", "10", "6", "1", "16"]
