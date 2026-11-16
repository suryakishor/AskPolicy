"""Text extraction for PDF, TXT, and DOCX documents.

PDF files yield one document per page so citations can point at real pages;
DOCX and TXT yield one document for the whole file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.errors import IngestionError
from app.core.logging import get_logger
from docx import Document as DocxFile
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document
from pypdf import PdfReader

from ingestion.metadata import SUPPORTED_EXTENSIONS, base_metadata

logger = get_logger(__name__)

# Text encodings tried in order for .txt files.
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


@dataclass
class LoadResult:
    """Documents extracted from one file plus page statistics."""

    documents: list[Document] = field(default_factory=list)
    pdf_pages: int = 0
    empty: bool = False


def discover_documents(documents_root: Path) -> list[Path]:
    """Recursively find supported, non-hidden documents in sorted order."""
    if not documents_root.exists():
        return []
    files = [
        path
        for path in documents_root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(documents_root).parts)
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.as_posix().lower())


def load_document(path: Path, documents_root: Path) -> LoadResult:
    """Extract text from a supported file based on its extension.

    Raises:
        IngestionError: The file is unreadable or unsupported.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, documents_root)
    if suffix == ".txt":
        return _load_txt(path, documents_root)
    if suffix == ".docx":
        return _load_docx(path, documents_root)
    raise IngestionError(f"Unsupported document format: {path.name}")


def _load_pdf(path: Path, documents_root: Path) -> LoadResult:
    """Extract one ``Document`` per non-empty PDF page."""
    try:
        reader = PdfReader(str(path))
        # Try the common empty-password case; otherwise fail loudly.
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestionError(f"PDF is password protected: {path.name}")
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Could not read PDF '{path.name}': {exc}") from exc

    base = base_metadata(path, documents_root)
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise IngestionError(
                f"Could not extract page {page_number} from '{path.name}': {exc}"
            ) from exc
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        documents.append(
            Document(
                page_content=cleaned,
                metadata={**base, "page_number": page_number},
            )
        )

    if not documents:
        logger.warning("PDF contains no extractable text: %s", path.name)
    return LoadResult(
        documents=documents,
        pdf_pages=len(reader.pages),
        empty=not documents,
    )


def _load_txt(path: Path, documents_root: Path) -> LoadResult:
    """Read a plain text file using the first encoding that works."""
    text = ""
    for encoding in TEXT_ENCODINGS:
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise IngestionError(f"Could not read TXT '{path.name}': {exc}") from exc
    else:
        raise IngestionError(f"Could not decode '{path.name}' with supported encodings.")

    cleaned = _clean_text(text)
    if not cleaned:
        logger.warning("TXT document is empty: %s", path.name)
        return LoadResult(empty=True)

    return LoadResult(
        documents=[
            Document(
                page_content=cleaned,
                metadata=base_metadata(path, documents_root),
            )
        ]
    )


def _load_docx(path: Path, documents_root: Path) -> LoadResult:
    """Extract paragraphs and tables from a Word document in document order."""
    try:
        docx = DocxFile(str(path))
    except Exception as exc:
        raise IngestionError(f"Could not read DOCX '{path.name}': {exc}") from exc

    blocks: list[str] = []
    for block in _iter_docx_blocks(docx):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                blocks.append(text)
        else:
            # Include table content; policies often store pay scales and leave
            # tables inside Word tables.
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    blocks.append(" | ".join(cells))

    cleaned = _clean_text("\n\n".join(blocks))
    if not cleaned:
        logger.warning("DOCX document contains no text: %s", path.name)
        return LoadResult(empty=True)

    return LoadResult(
        documents=[
            Document(
                page_content=cleaned,
                metadata=base_metadata(path, documents_root),
            )
        ]
    )


def _iter_docx_blocks(docx):
    """Yield paragraphs and tables in the order they appear in the document."""
    body = docx.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, docx)
        elif child.tag == qn("w:tbl"):
            yield Table(child, docx)


def _clean_text(text: str) -> str:
    """Collapse runs of spaces while preserving blank-line boundaries."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized = "\n".join(lines)
    # Collapse three or more blank lines, which PDF extraction often creates.
    while "\n\n\n" in normalized:
        normalized = normalized.replace("\n\n\n", "\n\n")
    return normalized.strip()
