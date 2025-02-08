from __future__ import annotations

from pathlib import Path

import pytest

from kreuzberg._extractors import (
    extract_content_with_pandoc,
    extract_file_with_pandoc,
    extract_pdf_file,
    extract_pdf_with_pdfium2,
    extract_pdf_with_tesseract,
)
from kreuzberg._tesseract import process_image_with_tesseract
from kreuzberg.exceptions import OCRError, ParsingError


async def test_extract_pdf_with_pdfium2(searchable_pdf: Path) -> None:
    result = await extract_pdf_with_pdfium2(searchable_pdf)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_pdf_with_tesseract(scanned_pdf: Path) -> None:
    result = await extract_pdf_with_tesseract(scanned_pdf)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_pdf_file(searchable_pdf: Path) -> None:
    result = await extract_pdf_file(searchable_pdf)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_pdf_file_non_searchable(non_searchable_pdf: Path) -> None:
    result = await extract_pdf_file(non_searchable_pdf)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_pdf_file_invalid() -> None:
    with pytest.raises(FileNotFoundError):
        await extract_pdf_file(Path("/invalid/path.pdf"))


async def test_extract_content_with_pandoc(docx_document: Path) -> None:
    content = docx_document.read_bytes()
    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    result = await extract_content_with_pandoc(content, mime_type)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_file_with_pandoc(docx_document: Path) -> None:
    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    result = await extract_file_with_pandoc(docx_document, mime_type)
    assert isinstance(result, str)
    assert result.strip()


async def test_extract_file_with_pandoc_invalid() -> None:
    with pytest.raises(ParsingError):
        await extract_file_with_pandoc(
            "/invalid/path.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


async def test_process_image_with_tesseract(ocr_image: Path) -> None:
    result = await process_image_with_tesseract(ocr_image)
    assert isinstance(result, str)
    assert result.strip()


async def test_process_image_with_tesseract_invalid() -> None:
    with pytest.raises(OCRError):
        await process_image_with_tesseract("/invalid/path.jpg")
