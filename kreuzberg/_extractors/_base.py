from __future__ import annotations

import io
import logging
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import asdict
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, ClassVar

from PIL import Image

from kreuzberg._ocr import get_ocr_backend
from kreuzberg._types import (
    EasyOCRConfig,
    ExtractedImage,
    ExtractionResult,
    ImageOCRResult,
    PaddleOCRConfig,
    TesseractConfig,
    normalize_metadata,
)
from kreuzberg._utils._quality import calculate_quality_score, clean_extracted_text
from kreuzberg._utils._sync import run_taskgroup_batched

if TYPE_CHECKING:
    from pathlib import Path

    from kreuzberg._types import ExtractionConfig

MAX_TOTAL_IMAGE_SIZE = 100 * 1024 * 1024
MAX_SINGLE_IMAGE_SIZE = 50 * 1024 * 1024

logger = logging.getLogger(__name__)


class Extractor(ABC):
    __slots__ = ("config", "mime_type")

    SUPPORTED_MIME_TYPES: ClassVar[set[str]]

    def __init__(self, mime_type: str, config: ExtractionConfig) -> None:
        self.mime_type = mime_type
        self.config = config

    @abstractmethod
    async def extract_bytes_async(self, content: bytes) -> ExtractionResult: ...

    @abstractmethod
    async def extract_path_async(self, path: Path) -> ExtractionResult: ...

    @abstractmethod
    def extract_bytes_sync(self, content: bytes) -> ExtractionResult: ...

    @abstractmethod
    def extract_path_sync(self, path: Path) -> ExtractionResult: ...

    @classmethod
    def supports_mimetype(cls, mime_type: str) -> bool:
        return mime_type in cls.SUPPORTED_MIME_TYPES or any(
            mime_type.startswith(supported_type) for supported_type in cls.SUPPORTED_MIME_TYPES
        )

    def _apply_quality_processing(self, result: ExtractionResult) -> ExtractionResult:
        if not self.config.enable_quality_processing:
            return result

        if not result.content:
            return result

        cleaned_content = clean_extracted_text(result.content)

        quality_score = calculate_quality_score(cleaned_content, dict(result.metadata) if result.metadata else None)

        enhanced_metadata = (dict(result.metadata) if result.metadata else {}) | {"quality_score": quality_score}

        deduplicated_images = self._deduplicate_images(result.images) if result.images else []

        return ExtractionResult(
            content=cleaned_content,
            mime_type=result.mime_type,
            metadata=normalize_metadata(enhanced_metadata),
            tables=result.tables,
            chunks=result.chunks,
            images=deduplicated_images,
            image_ocr_results=result.image_ocr_results,
            entities=result.entities,
            keywords=result.keywords,
            detected_languages=result.detected_languages,
            document_type=result.document_type,
            document_type_confidence=result.document_type_confidence,
            layout=result.layout,
        )

    def _check_image_memory_limits(self, images: list[ExtractedImage]) -> list[ExtractedImage]:
        """Filter images based on memory safety limits."""
        if not images:
            return []

        total_size = sum(len(img.data) for img in images)
        if total_size > MAX_TOTAL_IMAGE_SIZE:
            logger.warning(
                "Total image size %d bytes exceeds limit of %d bytes, filtering large images",
                total_size,
                MAX_TOTAL_IMAGE_SIZE,
            )

            sorted_images = sorted(images, key=lambda x: len(x.data))
            filtered_images = []
            current_size = 0

            for img in sorted_images:
                img_size = len(img.data)
                if img_size > MAX_SINGLE_IMAGE_SIZE:
                    logger.warning(
                        "Skipping image %s: size %d bytes exceeds single image limit of %d bytes",
                        img.filename or "unknown",
                        img_size,
                        MAX_SINGLE_IMAGE_SIZE,
                    )
                    continue

                if current_size + img_size <= MAX_TOTAL_IMAGE_SIZE:
                    filtered_images.append(img)
                    current_size += img_size
                else:
                    logger.warning("Skipping image %s: would exceed total memory limit", img.filename or "unknown")

            return filtered_images

        filtered = []
        for img in images:
            img_size = len(img.data)
            if img_size > MAX_SINGLE_IMAGE_SIZE:
                logger.warning(
                    "Skipping image %s: size %d bytes exceeds limit of %d bytes",
                    img.filename or "unknown",
                    img_size,
                    MAX_SINGLE_IMAGE_SIZE,
                )
            else:
                filtered.append(img)

        return filtered

    def _deduplicate_images(self, images: list[ExtractedImage]) -> list[ExtractedImage]:
        if not self.config.deduplicate_images or not images:
            return images

        seen_hashes = set()
        unique_images = []

        for img in images:
            img_hash = zlib.crc32(img.data) & 0xFFFFFFFF
            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                unique_images.append(img)
            else:
                logger.debug("Filtered duplicate image: %s", img.filename)

        if len(unique_images) < len(images):
            logger.info("Deduplicated %d images to %d unique", len(images), len(unique_images))

        return unique_images

    async def _process_images_with_ocr(self, images: list[ExtractedImage]) -> list[ImageOCRResult]:  # noqa: C901, PLR0915
        if not images or not self.config.ocr_extracted_images:
            return []

        images = self._deduplicate_images(images)
        images = self._check_image_memory_limits(images)

        backend_name = self.config.image_ocr_backend or self.config.ocr_backend
        if backend_name is None:
            return []

        def _config_for_backend() -> dict[str, object]:
            if self.config.ocr_config is not None:
                if backend_name == "tesseract" and isinstance(self.config.ocr_config, TesseractConfig):
                    cfg = asdict(TesseractConfig())
                    cfg.update(asdict(self.config.ocr_config))
                    cfg["use_cache"] = self.config.use_cache
                    return cfg
                if backend_name == "easyocr" and isinstance(self.config.ocr_config, EasyOCRConfig):
                    cfg = asdict(EasyOCRConfig())
                    cfg.update(asdict(self.config.ocr_config))
                    cfg["use_cache"] = self.config.use_cache
                    return cfg
                if backend_name == "paddleocr" and isinstance(self.config.ocr_config, PaddleOCRConfig):
                    cfg = asdict(PaddleOCRConfig())
                    cfg.update(asdict(self.config.ocr_config))
                    cfg["use_cache"] = self.config.use_cache
                    return cfg

            if backend_name == "tesseract":
                cfg = asdict(TesseractConfig())
            elif backend_name == "easyocr":
                cfg = asdict(EasyOCRConfig())
            else:
                cfg = asdict(PaddleOCRConfig())
            cfg["use_cache"] = self.config.use_cache
            return cfg

        cfg = _config_for_backend()
        backend = get_ocr_backend(backend_name)

        results: list[ImageOCRResult] = []
        min_w, min_h = self.config.image_ocr_min_dimensions
        max_w, max_h = self.config.image_ocr_max_dimensions

        tasks = []

        async def _ocr_one(target: ExtractedImage) -> ImageOCRResult:
            try:
                start = time.time()
                pil_img = Image.open(io.BytesIO(target.data))
                ocr_res = await backend.process_image(pil_img, **cfg)
                duration = time.time() - start
                return ImageOCRResult(image=target, ocr_result=ocr_res, confidence_score=None, processing_time=duration)
            except Exception as e:  # pragma: no cover  # noqa: BLE001
                return ImageOCRResult(
                    image=target,
                    ocr_result=ExtractionResult(content="", mime_type="text/plain", metadata={}),
                    skipped_reason=f"OCR failed: {e}",
                )

        for img in images:
            fmt = img.format.lower()
            if fmt not in self.config.image_ocr_formats:
                results.append(
                    ImageOCRResult(
                        image=img,
                        ocr_result=ExtractionResult(content="", mime_type="text/plain", metadata={}),
                        skipped_reason=f"Unsupported format: {img.format}",
                    )
                )
                continue

            if img.dimensions is not None:
                w, h = img.dimensions
                if w < min_w or h < min_h:
                    results.append(
                        ImageOCRResult(
                            image=img,
                            ocr_result=ExtractionResult(content="", mime_type="text/plain", metadata={}),
                            skipped_reason=f"Too small: {w}x{h}",
                        )
                    )
                    continue
                if w > max_w or h > max_h:
                    results.append(
                        ImageOCRResult(
                            image=img,
                            ocr_result=ExtractionResult(content="", mime_type="text/plain", metadata={}),
                            skipped_reason=f"Too large: {w}x{h}",
                        )
                    )
                    continue

            tasks.append(_ocr_one(img))

        if tasks:
            batch = max(1, min(len(tasks), cpu_count()))
            results.extend(await run_taskgroup_batched(*tasks, batch_size=batch))

        return results
