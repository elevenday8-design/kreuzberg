from __future__ import annotations

from functools import lru_cache, wraps

import kreuzberg._ocr as _ocr_module
from kreuzberg._registry import ExtractorRegistry

from .config import requires_full_pdf_extractor
from .nas_ocr_backend import NASOCRBackend
from .pymupdf_pdf_extractor import PyMuPDFPDFExtractor

ExtractorRegistry.add_extractor(PyMuPDFPDFExtractor)


def _patch_ocr_backend() -> None:
    original_get_backend = _ocr_module.get_ocr_backend

    @lru_cache
    @wraps(original_get_backend)
    def patched_get_backend(backend: str):
        if backend == "nas":
            return NASOCRBackend()
        return original_get_backend(backend)

    if hasattr(original_get_backend, "cache_clear"):
        original_get_backend.cache_clear()

    setattr(_ocr_module, "NASOCRBackend", NASOCRBackend)
    if hasattr(_ocr_module, "__all__"):
        updated_all = list(getattr(_ocr_module, "__all__", []))
        if "NASOCRBackend" not in updated_all:
            updated_all.append("NASOCRBackend")
        setattr(_ocr_module, "__all__", updated_all)

    _ocr_module.get_ocr_backend = patched_get_backend  # type: ignore[assignment]


_patch_ocr_backend()

__all__ = ["PyMuPDFPDFExtractor", "requires_full_pdf_extractor", "NASOCRBackend"]
