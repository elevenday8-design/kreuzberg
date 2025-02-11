"""Configuration for Kreuzberg text extraction."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass


@dataclass
class Config:
    """Global configuration for text extraction.

    Attributes:
        max_concurrent_ocr: Maximum number of concurrent Tesseract processes.
            If None, defaults to CPU count / 2 (minimum 1).
            This helps prevent resource exhaustion while still allowing parallelism.
    """

    max_concurrent_ocr: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_concurrent_ocr is not None:
            if not isinstance(self.max_concurrent_ocr, int):
                raise ValueError("max_concurrent_ocr must be an integer")
            if self.max_concurrent_ocr < 1:
                raise ValueError("max_concurrent_ocr must be at least 1")

    @property
    def concurrent_limit(self) -> int:
        """Get the actual concurrent process limit to use."""
        if self.max_concurrent_ocr is not None:
            return self.max_concurrent_ocr
        # Default to CPU count / 2, minimum 1
        # This accounts for Tesseract's internal threading
        return max(multiprocessing.cpu_count() // 2, 1)


# Default configuration
default_config = Config()
