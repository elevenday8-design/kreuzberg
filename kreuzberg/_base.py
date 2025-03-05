from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kreuzberg._constants import DEFAULT_MAX_PROCESSES

if TYPE_CHECKING:
    from pathlib import Path

    from kreuzberg import ExtractionResult, PSMMode


@dataclass
class ExtractionConfig:
    force_ocr: bool = False
    language: str = "eng"
    max_processes: int = DEFAULT_MAX_PROCESSES
    psm: PSMMode | None = None


class BaseExtractor(ABC):
    __slots__ = ("config",)

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    @abstractmethod
    async def extract_bytes_async(self, content: bytes) -> ExtractionResult: ...

    @abstractmethod
    async def extract_path_async(self, path: Path) -> ExtractionResult: ...

    @abstractmethod
    def extract_bytes_sync(self, content: bytes) -> ExtractionResult: ...

    @abstractmethod
    def extract_path_sync(self, path: Path) -> ExtractionResult: ...
