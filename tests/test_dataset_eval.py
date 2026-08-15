"""Tests for dataset-backed Stage 1 evaluation."""

from document_intelligence.evaluation.dataset_eval import build_stream_sample, evaluate_boundary_method
from document_intelligence.dataset.adapter import DocSplitAdapter


def _rows(stream_id: str, specs: list[tuple[int, int, str]]) -> list[dict]:
    return [
        {"stream_id": stream_id, "position": pos, "boundary": boundary, "page_text": text}
        for pos, boundary, text in specs
    ]


def test_build_stream_sample_boundary_pairs():
    adapter = DocSplitAdapter()
    rows = _rows(
        "s1",
        [
            (0, 1, "doc1-a"),
            (1, 0, "doc1-b"),
            (2, 1, "doc2-a"),
        ],
    )
    sample = build_stream_sample(adapter, "s1", rows)
    assert sample.true_groups == [[1, 2], [3]]
    assert sample.true_boundary_pairs == [(1, 2, False), (2, 3, True)]


def test_evaluate_boundary_method_macro_averages_per_stream():
    adapter = DocSplitAdapter()
    samples = [
        build_stream_sample(
            adapter,
            "s1",
            _rows("s1", [(0, 1, "a"), (1, 0, "b"), (2, 1, "c")]),
        ),
        build_stream_sample(
            adapter,
            "s2",
            _rows("s2", [(0, 1, "x"), (1, 0, "y")]),
        ),
    ]
    result = evaluate_boundary_method(samples, "baseline_rule")
    assert result["streams_evaluated"] == 2
    assert result["page_pairs_evaluated"] == 3
    assert 0.0 <= result["page_grouping_accuracy"] <= 1.0
