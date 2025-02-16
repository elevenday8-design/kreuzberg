from __future__ import annotations

from typing import Final

import cv2
import numpy as np
from PIL import Image, ImageOps

DEFAULT_DPI: Final[int] = 72
TARGET_DPI: Final[int] = 300
BINARIZATION_THRESHOLD: Final[int] = 0
BINARIZATION_MAX_VALUE: Final[int] = 255


def resize_for_ocr(image: Image.Image) -> Image.Image:
    """Resize the image to ensure sufficient DPI for OCR.

    Args:
        image: Input Pillow image.

    Returns:
        The resized image.
    """
    width, height = image.size
    scale_factor = TARGET_DPI / DEFAULT_DPI
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess the input image for OCR.

    Args:
        image: Input Pillow image.

    Returns:
        The preprocessed version of the input image.
    """
    grayscale = ImageOps.grayscale(image)
    _, binarized_image = cv2.threshold(
        np.array(grayscale), BINARIZATION_THRESHOLD, BINARIZATION_MAX_VALUE, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return resize_for_ocr(Image.fromarray(binarized_image))
