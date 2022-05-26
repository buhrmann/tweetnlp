from tweetnlp import Task, data, eval


def test_scoring():
    task = Task.emoji
    pred = data.test_labels(task)
    s = eval.score(pred, task)
    assert s == 1.0
