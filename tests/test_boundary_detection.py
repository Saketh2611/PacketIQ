"""Tests for boundary detection."""

import numpy as np
import pytest

from document_intelligence.stage1.boundary_baseline import BoundaryBaseline
from document_intelligence.stage1.page_features import PageFeatureBuilder
from document_intelligence.stage1.similarity import cosine_similarity, normalized_overlap, tokenize
import numpy as np


def test_cosine_similarity():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_normalized_overlap():
    assert normalized_overlap({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_boundary_thresholding(sample_pages):
    baseline = BoundaryBaseline(threshold=0.3)
    decisions = baseline.predict_pairs(sample_pages)
    assert len(decisions) == 2
    assert all(hasattr(d, "is_boundary") for d in decisions)


def test_embedding_baseline(sample_pages):
    baseline = BoundaryBaseline(threshold=0.5, mode="embedding_only")
    decisions = baseline.predict_embedding_only(sample_pages)
    assert len(decisions) == 2


def test_page_features(sample_pages):
    builder = PageFeatureBuilder()
    pair = builder.build_pair(sample_pages[0], sample_pages[1])
    assert len(pair.features) == 9
