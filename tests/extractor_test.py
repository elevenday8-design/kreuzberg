from __future__ import annotations

from pathlib import Path

import pytest

from kreuzberg._extractors import (
    _extract_content_with_pandoc,
    _extract_file_with_pandoc,
    _extract_image_with_tesseract,
    _extract_pdf_file,
    _extract_pdf_with_pdfium2,
    _extract_pdf_with_tesseract,
)
from kreuzberg.exceptions import ParsingError


async def test_extract_pdf_with_pdfium2(searchable_pdf: Path) -> None:
    result = _extract_pdf_with_pdfium2(searchable_pdf)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_pdf_with_tesseract(scanned_pdf: Path) -> None:
    result = await _extract_pdf_with_tesseract(scanned_pdf)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_pdf_file(searchable_pdf: Path) -> None:
    result = await _extract_pdf_file(searchable_pdf)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_pdf_file_non_searchable(non_searchable_pdf: Path) -> None:
    result = await _extract_pdf_file(non_searchable_pdf)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_pdf_file_invalid() -> None:
    with pytest.raises(FileNotFoundError):
        await _extract_pdf_file(Path("/invalid/path.pdf"))


async def test_extract_content_with_pandoc(docx_document: Path) -> None:
    content = docx_document.read_bytes()
    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    result = await _extract_content_with_pandoc(content, mime_type)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_file_with_pandoc(docx_document: Path) -> None:
    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    result = await _extract_file_with_pandoc(docx_document, mime_type)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_file_with_pandoc_invalid() -> None:
    with pytest.raises(ParsingError):
        await _extract_file_with_pandoc(
            "/invalid/path.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


async def test_extract_image_with_tesseract(ocr_image: Path) -> None:
    result = await _extract_image_with_tesseract(ocr_image)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_image_with_tesseract_invalid() -> None:
    with pytest.raises(ParsingError):
        await _extract_image_with_tesseract("/invalid/path.jpg")
