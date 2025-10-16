from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import anyio
import fitz  # type: ignore[import-untyped]

from kreuzberg._extractors._base import Extractor
from kreuzberg._extractors._pdf import PDFExtractor
from kreuzberg._types import ExtractionResult, normalize_metadata

from .config import requires_full_pdf_extractor

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from fitz import Document


class PyMuPDFPDFExtractor(Extractor):
    """Lightweight PDF extractor using PyMuPDF for text-only extraction."""

    SUPPORTED_MIME_TYPES = {"application/pdf"}

    def _needs_fallback(self) -> bool:
        return requires_full_pdf_extractor(self.config)

    def _fallback(self) -> PDFExtractor:
        config = self.config

        image_ocr_enabled = False
        if config.image_ocr_config is not None:
            image_ocr_enabled = getattr(config.image_ocr_config, "enabled", False)
        image_ocr_enabled = image_ocr_enabled or config.ocr_extracted_images

        if config.ocr_config is None and config.ocr_backend in (None, "tesseract"):
            config = replace(config, ocr_backend="nas")

        if image_ocr_enabled and config.image_ocr_backend is None:
            config = replace(config, image_ocr_backend="nas")

        return PDFExtractor(mime_type=self.mime_type, config=config)

    def _extract_with_loader(self, loader: Callable[[], Document]) -> ExtractionResult:
        document = loader()
        try:
            text_parts: list[str] = []
            for page in document:
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text.rstrip())

            metadata = dict(document.metadata or {})
            metadata.setdefault("source_format", "pdf")
        finally:
            document.close()

        result = ExtractionResult(
            content="\n\n".join(part for part in text_parts if part).strip(),
            mime_type="text/plain",
            metadata=normalize_metadata(metadata),
        )
        return self._apply_quality_processing(result)

    def _extract_bytes_blocking(self, content: bytes) -> ExtractionResult:
        return self._extract_with_loader(lambda: fitz.open(stream=content, filetype="pdf"))

    def _extract_path_blocking(self, path: Path) -> ExtractionResult:
        return self._extract_with_loader(lambda: fitz.open(path=str(path)))

    def extract_bytes_sync(self, content: bytes) -> ExtractionResult:
        if self._needs_fallback():
            return self._fallback().extract_bytes_sync(content)

        return self._extract_bytes_blocking(content)

    def extract_path_sync(self, path: Path) -> ExtractionResult:
        if self._needs_fallback():
            return self._fallback().extract_path_sync(path)

        return self._extract_path_blocking(path)

    async def extract_bytes_async(self, content: bytes) -> ExtractionResult:
        if self._needs_fallback():
            return await self._fallback().extract_bytes_async(content)

        return await anyio.to_thread.run_sync(self._extract_bytes_blocking, content)

    async def extract_path_async(self, path: Path) -> ExtractionResult:
        if self._needs_fallback():
            return await self._fallback().extract_path_async(path)

        return await anyio.to_thread.run_sync(self._extract_path_blocking, path)
