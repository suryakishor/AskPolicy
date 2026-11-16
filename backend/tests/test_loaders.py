"""Tests for PDF, TXT, and DOCX extraction and document discovery."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.errors import IngestionError
from docx import Document as DocxFile
from ingestion.loaders import discover_documents, load_document


def build_pdf(pages: list[str]) -> bytes:
    """Create a tiny valid PDF with extractable text, without extra dependencies."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + index * 2} 0 R" for index in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, text in enumerate(pages):
        page_obj = 4 + index * 2
        content_obj = page_obj + 1
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj} 0 R >>"
            ).encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{object_number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_pdf_extraction_keeps_page_numbers(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    pdf_path = root / "handbook.pdf"
    pdf_path.write_bytes(
        build_pdf(
            [
                "AskPolicy leave policy allows twenty days of annual leave.",
                "Travel policy covers economy class flights for approved trips.",
            ]
        )
    )

    result = load_document(pdf_path, root)

    assert result.pdf_pages == 2
    assert result.empty is False
    assert [doc.metadata["page_number"] for doc in result.documents] == [1, 2]
    assert "twenty days" in result.documents[0].page_content
    assert result.documents[0].metadata["file_name"] == "handbook.pdf"
    assert result.documents[0].metadata["file_type"] == "pdf"
    assert result.documents[0].metadata["source"] == "handbook.pdf"


def test_pdf_skips_blank_pages_without_inventing_numbers(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    pdf_path = root / "partial.pdf"
    pdf_path.write_bytes(build_pdf(["", "Second page has content."]))

    result = load_document(pdf_path, root)

    assert result.pdf_pages == 2
    assert len(result.documents) == 1
    # The surviving document is genuinely page 2, not renumbered to page 1.
    assert result.documents[0].metadata["page_number"] == 2


def test_txt_extraction_handles_utf8_and_legacy_encodings(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    nested = root / "hr"
    nested.mkdir(parents=True)

    utf8_path = root / "leave.txt"
    utf8_path.write_text("Leave policy: 20 days.\n\nSubmit in advance.", encoding="utf-8")

    legacy_path = nested / "benefits.txt"
    legacy_path.write_bytes("Café allowance policy.".encode("cp1252"))

    utf8_result = load_document(utf8_path, root)
    legacy_result = load_document(legacy_path, root)

    assert "20 days" in utf8_result.documents[0].page_content
    assert "\n\n" in utf8_result.documents[0].page_content
    assert legacy_result.documents[0].page_content == "Café allowance policy."
    # Nested folders are preserved as a portable relative source path.
    assert legacy_result.documents[0].metadata["source"] == "hr/benefits.txt"
    assert "page_number" not in utf8_result.documents[0].metadata


def test_docx_extraction_includes_paragraphs_and_tables(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    docx_path = root / "onboarding.docx"

    document = DocxFile()
    document.add_paragraph("Employee onboarding procedure")
    document.add_paragraph("Complete the orientation checklist in week one.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Allowance"
    table.cell(0, 1).text = "Amount"
    table.cell(1, 0).text = "Home office"
    table.cell(1, 1).text = "500"
    document.save(docx_path)

    result = load_document(docx_path, root)

    assert result.empty is False
    content = result.documents[0].page_content
    assert "Employee onboarding procedure" in content
    assert "Home office | 500" in content
    assert "page_number" not in result.documents[0].metadata
    assert result.documents[0].metadata["file_type"] == "docx"


def test_empty_documents_are_flagged(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    empty_txt = root / "empty.txt"
    empty_txt.write_text("   \n\n  ", encoding="utf-8")

    result = load_document(empty_txt, root)

    assert result.empty is True
    assert result.documents == []


def test_discovery_ignores_unsupported_and_hidden_files(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    (root / "nested").mkdir(parents=True)
    (root / ".hidden").mkdir()
    (root / "policy.txt").write_text("policy", encoding="utf-8")
    (root / "nested" / "handbook.PDF").write_bytes(build_pdf(["page"]))
    (root / "notes.md").write_text("ignored", encoding="utf-8")
    (root / ".hidden" / "secret.txt").write_text("ignored", encoding="utf-8")
    (root / ".hidden.txt").write_text("ignored", encoding="utf-8")

    discovered = discover_documents(root)

    assert [path.name for path in discovered] == ["handbook.PDF", "policy.txt"]


def test_corrupt_pdf_raises_clear_error(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    broken = root / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")

    with pytest.raises(IngestionError, match="Could not read PDF"):
        load_document(broken, root)
