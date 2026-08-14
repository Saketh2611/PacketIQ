"""Tests for page grouping."""

from document_intelligence.stage1.boundary_baseline import BoundaryDecision
from document_intelligence.stage1.grouping import decisions_to_groups


def test_grouping_basic():
    decisions = [
        BoundaryDecision(page_a=1, page_b=2, score=0.8, is_boundary=False),
        BoundaryDecision(page_a=2, page_b=3, score=0.2, is_boundary=True),
        BoundaryDecision(page_a=3, page_b=4, score=0.7, is_boundary=False),
    ]
    groups = decisions_to_groups(4, decisions, "test")
    assert len(groups) == 2
    assert groups[0].page_numbers == [1, 2]
    assert groups[1].page_numbers == [3, 4]


def test_single_page():
    groups = decisions_to_groups(1, [], "test")
    assert len(groups) == 1
    assert groups[0].page_numbers == [1]


def test_all_pages_same_document():
    decisions = [
        BoundaryDecision(page_a=1, page_b=2, score=0.9, is_boundary=False),
        BoundaryDecision(page_a=2, page_b=3, score=0.9, is_boundary=False),
    ]
    groups = decisions_to_groups(3, decisions, "test")
    assert len(groups) == 1
    assert groups[0].page_numbers == [1, 2, 3]
