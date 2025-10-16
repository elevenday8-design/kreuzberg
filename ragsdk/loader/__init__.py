from __future__ import annotations

from kreuzberg._registry import ExtractorRegistry

from .config import requires_full_pdf_extractor
from .pymupdf_pdf_extractor import PyMuPDFPDFExtractor

ExtractorRegistry.add_extractor(PyMuPDFPDFExtractor)

__all__ = ["PyMuPDFPDFExtractor", "requires_full_pdf_extractor"]
