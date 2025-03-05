from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kreuzberg import ExtractionResult, PSMMode


@dataclass
class ExtractionConfig:
    force_ocr: bool
    language: str
    max_processes: int
    psm: PSMMode | None = None


class BaseExtractor(ABC):
    @abstractmethod
    async def extract_bytes_async(self, content: bytes, config: ExtractionConfig) -> ExtractionResult: ...

    @abstractmethod
    async def extract_path_async(self, path: Path, config: ExtractionConfig) -> ExtractionResult: ...

    @abstractmethod
    def extract_bytes_sync(self, content: bytes, config: ExtractionConfig) -> ExtractionResult: ...

    @abstractmethod
    def extract_path_sync(self, path: Path, config: ExtractionConfig) -> ExtractionResult: ...
