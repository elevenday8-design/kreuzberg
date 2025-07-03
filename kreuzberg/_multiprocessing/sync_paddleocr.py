"""Pure synchronous PaddleOCR without any async overhead."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from kreuzberg._mime_types import PLAIN_TEXT_MIME_TYPE
from kreuzberg._ocr._paddleocr import PaddleOCRConfig
from kreuzberg._types import ExtractionResult
from kreuzberg._utils._string import normalize_spaces
from kreuzberg.exceptions import MissingDependencyError, OCRError


def _get_paddleocr_instance(config: PaddleOCRConfig) -> Any:
    """Get a PaddleOCR instance with the given configuration."""
    try:
        import paddleocr
    except ImportError as e:
        raise MissingDependencyError("PaddleOCR is not installed. Install it with: pip install paddleocr") from e

    # Handle device configuration
    use_gpu = False
    if hasattr(config, "device"):
        if config.device and config.device.lower() != "cpu":
            use_gpu = True
    elif hasattr(config, "use_gpu"):
        use_gpu = config.use_gpu

    kwargs = {
        "use_angle_cls": config.use_angle_cls,
        "lang": config.language,
        "det_algorithm": config.det_algorithm,
        "rec_algorithm": config.rec_algorithm,
        "cls_algorithm": config.rec_algorithm,  # Use rec_algorithm as fallback
        "use_gpu": use_gpu,
        "show_log": False,  # Disable logging for sync mode
    }

    # Add optional parameters if they exist in config
    optional_params = [
        "det_model_dir",
        "rec_model_dir",
        "cls_model_dir",
        "det_db_thresh",
        "det_db_box_thresh",
        "det_db_unclip_ratio",
        "use_dilation",
        "det_db_score_mode",
        "det_east_score_thresh",
        "det_east_cover_thresh",
        "det_east_nms_thresh",
        "det_fce_box_thresh",
        "det_pse_thresh",
        "det_pse_box_thresh",
        "det_pse_min_area",
        "det_pse_box_type",
        "det_pse_scale",
        "scales",
        "alpha",
        "beta",
        "fourier_degree",
        "rec_batch_num",
        "max_text_length",
        "rec_char_dict_path",
        "use_space_char",
        "vis_font_path",
        "drop_score",
        "crop_res_save_dir",
        "save_crop_res",
        "cls_batch_num",
        "cls_thresh",
        "enable_mkldnn",
        "cpu_threads",
        "use_pdserving",
        "warmup",
        "use_mp",
        "total_process_num",
        "process_id",
        "benchmark",
        "save_log_path",
        "show_log",
        "use_onnx",
        "use_tensorrt",
        "precision",
        "gpu_id",
        "gpu_mem",
    ]

    for param in optional_params:
        if hasattr(config, param):
            value = getattr(config, param)
            if value is not None:
                kwargs[param] = value

    return paddleocr.PaddleOCR(**kwargs)


def process_image_sync_pure(
    image_path: str | Path,
    config: PaddleOCRConfig | None = None,
) -> ExtractionResult:
    """Process an image with PaddleOCR using pure sync implementation.

    This bypasses all async overhead and calls PaddleOCR directly.

    Args:
        image_path: Path to the image file.
        config: PaddleOCR configuration.

    Returns:
        Extraction result.
    """
    cfg = config or PaddleOCRConfig()

    try:
        ocr_instance = _get_paddleocr_instance(cfg)

        # PaddleOCR returns a list of results for each detected text region
        # Each result is [bbox, (text, confidence)]
        results = ocr_instance.ocr(str(image_path), cls=cfg.use_angle_cls)

        if not results or not results[0]:
            return ExtractionResult(
                content="",
                mime_type=PLAIN_TEXT_MIME_TYPE,
                metadata={},
                chunks=[],
            )

        # Extract text and confidence from results
        texts = []
        confidences = []

        for line in results[0]:
            if line:
                bbox, (text, confidence) = line
                texts.append(text)
                confidences.append(confidence)

        # Join all text with newlines
        content = "\n".join(texts)
        content = normalize_spaces(content)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        metadata = {"confidence": avg_confidence} if confidences else {}

        return ExtractionResult(
            content=content,
            mime_type=PLAIN_TEXT_MIME_TYPE,
            metadata=metadata,  # type: ignore[arg-type]
            chunks=[],
        )

    except Exception as e:
        raise OCRError(f"PaddleOCR processing failed: {e}") from e


def process_image_bytes_sync_pure(
    image_bytes: bytes,
    config: PaddleOCRConfig | None = None,
) -> ExtractionResult:
    """Process image bytes with PaddleOCR using pure sync implementation.

    Args:
        image_bytes: Image data as bytes.
        config: PaddleOCR configuration.

    Returns:
        Extraction result.
    """
    import io

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_image:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.save(tmp_image.name, format="PNG")
        image_path = tmp_image.name

    try:
        return process_image_sync_pure(image_path, config)
    finally:
        image_file = Path(image_path)
        if image_file.exists():
            image_file.unlink()


def process_batch_images_sync_pure(
    image_paths: list[str | Path],
    config: PaddleOCRConfig | None = None,
) -> list[ExtractionResult]:
    """Process a batch of images sequentially with pure sync implementation.

    Args:
        image_paths: List of image file paths.
        config: PaddleOCR configuration.

    Returns:
        List of extraction results.
    """
    results = []
    for image_path in image_paths:
        result = process_image_sync_pure(image_path, config)
        results.append(result)
    return results


def process_batch_images_threaded(
    image_paths: list[str | Path],
    config: PaddleOCRConfig | None = None,
    max_workers: int | None = None,
) -> list[ExtractionResult]:
    """Process a batch of images using threading.

    Args:
        image_paths: List of image file paths.
        config: PaddleOCR configuration.
        max_workers: Maximum number of threads.

    Returns:
        List of extraction results in same order as input.
    """
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers is None:
        max_workers = min(len(image_paths), mp.cpu_count())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_image_sync_pure, path, config): i for i, path in enumerate(image_paths)
        }

        results: list[ExtractionResult] = [None] * len(image_paths)  # type: ignore[list-item]
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:  # noqa: BLE001
                results[index] = ExtractionResult(
                    content=f"Error: {e}",
                    mime_type=PLAIN_TEXT_MIME_TYPE,
                    metadata={"error": str(e)},  # type: ignore[typeddict-unknown-key]
                    chunks=[],
                )

    return results
