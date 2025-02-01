from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kreuzberg._mime_types import MARKDOWN_MIME_TYPE, PDF_MIME_TYPE, PLAIN_TEXT_MIME_TYPE
from kreuzberg.exceptions import ValidationError
from kreuzberg.extraction import extract_bytes, extract_file

if TYPE_CHECKING:
    from pathlib import Path


async def test_extract_bytes_pdf(searchable_pdf: Path) -> None:
    content = searchable_pdf.read_bytes()
    result = await extract_bytes(content, PDF_MIME_TYPE)
    assert result.mime_type == PLAIN_TEXT_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip()


async def test_extract_bytes_image(ocr_image: Path) -> None:
    content = ocr_image.read_bytes()
    mime_type = "image/jpeg"
    result = await extract_bytes(content, mime_type)
    assert result.mime_type == PLAIN_TEXT_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip()


async def test_extract_bytes_pandoc(docx_document: Path) -> None:
    content = docx_document.read_bytes()
    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    result = await extract_bytes(content, mime_type)
    assert result.mime_type == MARKDOWN_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip()


async def test_extract_bytes_plain_text() -> None:
    content = b"This is a plain text file."
    result = await extract_bytes(content, PLAIN_TEXT_MIME_TYPE)
    assert result.mime_type == PLAIN_TEXT_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip() == "This is a plain text file."


async def test_extract_bytes_markdown(markdown_document: Path) -> None:
    content = markdown_document.read_bytes()
    result = await extract_bytes(content, MARKDOWN_MIME_TYPE)
    assert result.mime_type == MARKDOWN_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip()


async def test_extract_bytes_invalid_mime() -> None:
    with pytest.raises(ValidationError, match="Unsupported mime type"):
        await extract_bytes(b"some content", "application/unknown")


async def test_extract_file_pdf(searchable_pdf: Path) -> None:
    result = await extract_file(searchable_pdf, PDF_MIME_TYPE)
    assert result.mime_type == PLAIN_TEXT_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip()


async def test_extract_file_image(ocr_image: Path) -> None:
    mime_type = "image/jpeg"
    result = await extract_file(ocr_image, mime_type)
    assert result.mime_type == PLAIN_TEXT_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip()


async def test_extract_file_pandoc(docx_document: Path) -> None:
    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    result = await extract_file(docx_document, mime_type)
    assert result.mime_type == MARKDOWN_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip()


async def test_extract_file_plain_text(tmp_path: Path) -> None:
    text_file = tmp_path / "sample.txt"
    text_file.write_text("This is a plain text file.")
    result = await extract_file(text_file, PLAIN_TEXT_MIME_TYPE)
    assert result.mime_type == PLAIN_TEXT_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip() == "This is a plain text file."


async def test_extract_file_markdown(markdown_document: Path) -> None:
    result = await extract_file(markdown_document, MARKDOWN_MIME_TYPE)
    assert result.mime_type == MARKDOWN_MIME_TYPE
    assert isinstance(result.content, str)
    assert result.content.strip()


async def test_extract_file_invalid_mime() -> None:
    with pytest.raises(ValidationError, match="Unsupported mime type"):
        await extract_file("/invalid/path.txt", "application/unknown")


async def test_extract_file_not_exists() -> None:
    with pytest.raises(ValidationError, match="The file does not exist"):
        await extract_file("/invalid/path.txt", PLAIN_TEXT_MIME_TYPE)
