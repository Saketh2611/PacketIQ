"""Convert pairwise boundary decisions into contiguous document groups."""

from __future__ import annotations

from dataclasses import dataclass, field

from document_intelligence.stage1.boundary_baseline import BoundaryDecision
from document_intelligence.utils.hashing import make_document_id


@dataclass
class DocumentGroup:
    document_id: str
    page_numbers: list[int]
    page_start: int
    page_end: int
    boundary_confidences: list[float] = field(default_factory=list)
    group_confidence: float = 1.0

    @property
    def page_count(self) -> int:
        return len(self.page_numbers)


def decisions_to_groups(
    page_count: int,
    decisions: list[BoundaryDecision],
    packet_id: str = "packet",
) -> list[DocumentGroup]:
    """Convert boundary decisions to contiguous page groups."""
    if page_count == 0:
        return []

    groups: list[list[int]] = [[1]]
    confidences: list[list[float]] = [[]]

    for i, decision in enumerate(decisions):
        if decision.is_boundary:
            groups.append([decision.page_b])
            confidences.append([])
        else:
            groups[-1].append(decision.page_b)
            confidences[-1].append(decision.score)

    result: list[DocumentGroup] = []
    for idx, (pages, confs) in enumerate(zip(groups, confidences)):
        avg_conf = sum(confs) / len(confs) if confs else 1.0
        result.append(
            DocumentGroup(
                document_id=make_document_id(packet_id, idx + 1),
                page_numbers=pages,
                page_start=pages[0],
                page_end=pages[-1],
                boundary_confidences=confs,
                group_confidence=avg_conf,
            )
        )
    return result
