"""Wrappers for classification models.

Also see:
    - https://scikit-learn.org/stable/developers/develop.html

"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm.auto import tqdm
from transformers import pipeline

from .data import map_labels
from .utils import Task, preprocess

BASEURL = "cardiffnlp/twitter-roberta-base-{}".format


class PretrainedCardiffClassifier(BaseEstimator, ClassifierMixin):
    """A minimal sklearn-compmatible wrapper for pretrained CardiffNLP models."""

    def __init__(
        self,
        task: Task,
        batch_size: int = 1,
        n_workers: int = 1,
        humanize: bool = False,
    ):
        self.task = task
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.humanize = humanize

    def _load(self):
        model = BASEURL(self.task.name)
        print(f"Loading model {model}")
        return pipeline("text-classification", model=model)

    def fit(self, X, y=None):
        """Already fitted. No fine-tuning for now."""
        return self

    def predict(self, X):

        if not hasattr(self, "_pipe"):
            self._pipe = self._load()

        X = [preprocess(txt) for txt in tqdm(X, desc="Preprocessing")]
        X = (x for x in X)  # Force hugginface to use generator otherwise no progress

        gen = self._pipe(X, batch_size=self.batch_size, num_workers=self.n_workers)
        res = list(tqdm(gen, desc="Classifying"))
        labels, scores = zip(*[r.values() for r in res])

        self.scores_ = scores

        labels = [label.replace("LABEL_", "") for label in labels]
        if self.humanize:
            labels = map_labels(labels, task=self.task)

        print(f"First 10 labels: {labels[:10]}")
        return labels

    __call__ = predict


def TfidfRegression(tfidf_cfg=None, svd_cfg=None, lr_cfg=None):
    """TfIdf followed by LogReg, just a template for dumb baselines."""
    tfidf_cfg = tfidf_cfg or {}
    svd_cfg = {**{"n_components": 300}, **(svd_cfg or {})}
    lr_cfg = {**{"Cs": 10, "max_iter": 500}, **(lr_cfg or {})}

    preproc = FunctionTransformer(lambda X: [preprocess(t) for t in X])

    return make_pipeline(
        preproc,
        TfidfVectorizer(**tfidf_cfg),
        TruncatedSVD(**svd_cfg),
        LogisticRegressionCV(**lr_cfg),
    )
