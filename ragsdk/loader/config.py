from __future__ import annotations

from kreuzberg._types import ExtractionConfig


def requires_full_pdf_extractor(config: ExtractionConfig) -> bool:
    """Determine whether the full Kreuzberg PDF extractor is required.

    The lightweight PyMuPDF extractor only handles plain text extraction. When the
    caller enables features that depend on OCR, table extraction, or image
    processing, we defer to the default Kreuzberg PDF extractor so all requested
    capabilities remain available.
    """

    if config.force_ocr:
        return True

    if config.extract_tables or config.extract_tables_from_ocr:
        return True

    if config.extract_images:
        return True

    image_ocr_enabled = False
    if config.image_ocr_config is not None:
        image_ocr_enabled = getattr(config.image_ocr_config, "enabled", False)
    image_ocr_enabled = image_ocr_enabled or config.ocr_extracted_images

    if image_ocr_enabled:
        return True

    if config.image_ocr_backend is not None:
        return True

    return False
