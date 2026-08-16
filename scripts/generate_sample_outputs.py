#!/usr/bin/env python3
"""Generate sample PDF and run end-to-end pipeline for sample outputs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz

from document_intelligence.config.settings import get_settings
from document_intelligence.pipeline import DocumentIntelligencePipeline
from document_intelligence.utils.io import write_json


def create_sample_pdf(path: Path) -> None:
    """Create a multi-document sample PDF packet."""
    doc = fitz.open()
    pages_content = [
        ("INVOICE\n\nBill To: Acme Corporation\n123 Main Street\n\nInvoice #INV-2024-001\nDate: 2024-01-15",),
        ("INVOICE (continued)\n\nItem          Qty    Amount\nLaptop         2     50000\nMouse          5       500\n\nSubtotal: 50500",),
        ("INVOICE (continued)\n\nTax (4%): 2020\nTotal Amount: ₹52,340\n\nPayment Terms: Net 30 days",),
        ("CURRICULUM VITAE\n\nJohn Doe\nSoftware Engineer\n\nExperience:\n- Senior Developer at TechCo (2020-2024)\n- Junior Developer at StartupInc (2018-2020)",),
        ("RESUME (continued)\n\nEducation:\n- BS Computer Science, State University (2018)\n\nSkills:\nPython, Machine Learning, FastAPI",),
        ("PASSPORT\n\nType: P\nCountry Code: EXA\nPassport No: AB1234567\n\nSurname: DOE\nGiven Names: JOHN\nNationality: EXAMPLE\nDate of Birth: 01 JAN 1990",),
    ]
    for content in pages_content:
        page = doc.new_page()
        page.insert_text((72, 72), content[0], fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    print(f"Created sample PDF: {path}")


def main() -> None:
    settings = get_settings()
    samples_dir = settings.outputs_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = samples_dir / "sample_packet.pdf"
    create_sample_pdf(pdf_path)

    pipeline = DocumentIntelligencePipeline()

    # Stage 1
    stage1 = pipeline.run_stage1(pdf_path)
    write_json(samples_dir / "stage1_output.json", stage1)
    print(f"Stage 1: {len(stage1['documents'])} documents")

    # Stage 2
    structured = pipeline.run_stage2(stage1, pdf_path)
    structured_dir = samples_dir / "structured"
    structured_dir.mkdir(exist_ok=True)
    for doc in structured:
        write_json(structured_dir / f"{doc.document_id}.json", doc.model_dump())
    write_json(samples_dir / "structured_document.json", structured[0].model_dump() if structured else {})
    print(f"Stage 2: {len(structured)} structured documents")

    # Stage 3
    index_stats = pipeline.build_index(structured)
    retrieval = pipeline.query("What is the total amount on the invoice?")
    write_json(samples_dir / "retrieval_output.json", retrieval)
    print(f"Stage 3: indexed {index_stats['chunks_indexed']} chunks, retrieval returned {len(retrieval['results'])} results")

    # Failure case example
    failure = {
        "case": "empty_query",
        "query": "",
        "result": pipeline.query(""),
        "note": "Empty query returns no results with an empty_query validation warning",
    }
    write_json(samples_dir / "failure_case_empty_query.json", failure)
    print(f"Sample outputs saved to {samples_dir}")


if __name__ == "__main__":
    main()
