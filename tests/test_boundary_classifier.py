"""Tests for boundary classifier training/loading."""

import numpy as np
import pytest

from document_intelligence.stage1.boundary_classifier import BoundaryClassifier
from document_intelligence.stage1.page_features import PagePairFeatures


def test_classifier_train_predict():
    clf = BoundaryClassifier(model_type="logistic_regression")
    X = np.random.rand(20, 9)
    y = np.array([1] * 10 + [0] * 10)
    result = clf.train(X, y, val_size=0.3)
    assert result.train_size > 0
    pairs = [
        PagePairFeatures(page_a_number=i, page_b_number=i + 1, features=X[i])
        for i in range(5)
    ]
    decisions = clf.predict_decisions(pairs)
    assert len(decisions) == 5


def test_classifier_save_load(tmp_path):
    clf = BoundaryClassifier()
    X = np.random.rand(10, 9)
    y = np.array([1] * 5 + [0] * 5)
    clf.train(X, y)
    path = tmp_path / "model.joblib"
    clf.save(path)
    clf2 = BoundaryClassifier()
    clf2.load(path)
    pairs = [PagePairFeatures(page_a_number=1, page_b_number=2, features=X[0])]
    d1 = clf.predict_decisions(pairs)
    d2 = clf2.predict_decisions(pairs)
    assert d1[0].is_boundary == d2[0].is_boundary
    
def test_predict_decisions_returns_native_types():
    clf = BoundaryClassifier()
    X = np.random.rand(10, 9)
    y = np.array([1] * 5 + [0] * 5)
    clf.train(X, y)
    pairs = [PagePairFeatures(page_a_number=1, page_b_number=2, features=X[0])]
    decisions = clf.predict_decisions(pairs)
    assert isinstance(decisions[0].is_boundary, bool)  # not numpy.bool_
    assert isinstance(decisions[0].score, float)        # not numpy.float64
