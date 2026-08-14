"""Tests for dataset schema adaptation."""

from document_intelligence.dataset.adapter import DocSplitAdapter


def test_stream_rows_to_annotation_uses_boundary_as_document_start():
    rows = [
        {"stream_id": "s1", "position": 0, "boundary": 1, "page_text": "a"},
        {"stream_id": "s1", "position": 1, "boundary": 0, "page_text": "b"},
        {"stream_id": "s1", "position": 2, "boundary": 1, "page_text": "c"},
        {"stream_id": "s1", "position": 3, "boundary": 0, "page_text": "d"},
    ]

    annotation = DocSplitAdapter().stream_rows_to_annotation(rows)

    assert annotation.packet_id == "s1"
    assert annotation.page_count == 4
    assert annotation.groups == [[1, 2], [3, 4]]


def test_stream_rows_to_annotation_supports_openpss_label_field():
    rows = [
        {"stream_id": "s2", "position": 1, "label": 1, "text": "a"},
        {"stream_id": "s2", "position": 2, "label": 1, "text": "b"},
        {"stream_id": "s2", "position": 3, "label": 0, "text": "c"},
    ]

    annotation = DocSplitAdapter().stream_rows_to_annotation(rows)

    assert annotation.groups == [[1], [2, 3]]


def test_normalize_stream_rows_maps_benchmark_columns():
    rows = [
        {
            "stream_id": "bench",
            "position": 0,
            "boundary": 1,
            "page_text": "first",
            "source": "openpss_short",
        },
        {
            "stream_id": "bench",
            "position": 1,
            "boundary": 0,
            "page_text": "second",
            "source": "openpss_short",
        },
    ]

    pages = DocSplitAdapter().normalize_stream_rows(rows)

    assert [p.page_number for p in pages] == [1, 2]
    assert [p.original_position for p in pages] == [0, 1]
    assert [p.text for p in pages] == ["first", "second"]
    assert [p.starts_new_document for p in pages] == [True, False]
    assert pages[0].source == "openpss_short"


def test_normalize_stream_rows_maps_openpss_columns_and_sorts_positions():
    rows = [
        {"stream_id": "train", "position": 2, "label": 0, "text": "second"},
        {"stream_id": "train", "position": 1, "label": 1, "text": "first"},
    ]

    pages = DocSplitAdapter().normalize_stream_rows(rows)

    assert [p.page_number for p in pages] == [1, 2]
    assert [p.original_position for p in pages] == [1, 2]
    assert [p.text for p in pages] == ["first", "second"]
    assert [p.starts_new_document for p in pages] == [True, False]
    assert pages[0].source is None


def test_stream_rows_to_pages_uses_normalized_text_and_metadata():
    rows = [
        {"stream_id": "bench", "position": 0, "boundary": 1, "page_text": "hello", "source": "src"},
    ]

    pages = DocSplitAdapter().stream_rows_to_pages(rows)

    assert pages[0].page_number == 1
    assert pages[0].text == "hello"
    assert pages[0].metadata["original_position"] == 0
    assert pages[0].metadata["starts_new_document"] is True
    assert pages[0].metadata["source"] == "src"
