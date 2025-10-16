"""Convenience loader utilities built on top of :mod:`kreuzberg` extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from kreuzberg._types import ExtractionConfig, ExtractionResult, ExtractedImage
from kreuzberg.extraction import (
    extract_bytes,
    extract_bytes_sync,
    extract_file,
    extract_file_sync,
)


@dataclass(slots=True)
class LoaderOutput:
    """Structured representation of a document returned by :class:`KreuzbergLoader`.

    Attributes:
        text: The textual content returned by the extractor.
        mime_type: MIME type of the extracted text representation.
        metadata: Metadata dictionary returned by the extractor. A shallow copy is
            stored so callers can mutate it freely without affecting cached results.
        images: Collection of extracted images that belong to the document.
        source: Optional source path used for extraction. ``None`` for in-memory
            extractions.
        raw_result: Full :class:`kreuzberg._types.ExtractionResult` for advanced
            consumers that require access to lower level details.
    """

    text: str
    mime_type: str
    metadata: dict[str, Any]
    images: list[ExtractedImage]
    source: Path | None
    raw_result: ExtractionResult


def _ensure_chunking_disabled(config: ExtractionConfig | None) -> ExtractionConfig:
    """Return a config instance that does not request chunked content."""

    if config is None:
        return ExtractionConfig(chunk_content=False)

    if config.chunk_content:
        return replace(config, chunk_content=False)

    return config


class KreuzbergLoader:
    """Wrapper around :mod:`kreuzberg` extraction utilities.

    The loader guarantees that raw chunk generation is disabled so downstream
    callers can perform project specific chunking with :class:`ragsdk.splitter`
    utilities.
    """

    def __init__(self, *, config: ExtractionConfig | None = None) -> None:
        self._base_config = _ensure_chunking_disabled(config)

    async def load_file(
        self,
        path: str | Path,
        *,
        mime_type: str | None = None,
        config: ExtractionConfig | None = None,
    ) -> LoaderOutput:
        """Asynchronously extract a document from ``path``."""

        resolved_config = _ensure_chunking_disabled(config) if config else self._base_config
        result = await extract_file(path, mime_type=mime_type, config=resolved_config)
        return self._to_output(result, source=Path(path))

    def load_file_sync(
        self,
        path: str | Path,
        *,
        mime_type: str | None = None,
        config: ExtractionConfig | None = None,
    ) -> LoaderOutput:
        """Synchronously extract a document from ``path``."""

        resolved_config = _ensure_chunking_disabled(config) if config else self._base_config
        result = extract_file_sync(path, mime_type=mime_type, config=resolved_config)
        return self._to_output(result, source=Path(path))

    async def load_bytes(
        self,
        content: bytes,
        *,
        mime_type: str,
        config: ExtractionConfig | None = None,
    ) -> LoaderOutput:
        """Asynchronously extract a document from in-memory ``content``."""

        resolved_config = _ensure_chunking_disabled(config) if config else self._base_config
        result = await extract_bytes(content, mime_type=mime_type, config=resolved_config)
        return self._to_output(result, source=None)

    def load_bytes_sync(
        self,
        content: bytes,
        *,
        mime_type: str,
        config: ExtractionConfig | None = None,
    ) -> LoaderOutput:
        """Synchronously extract a document from in-memory ``content``."""

        resolved_config = _ensure_chunking_disabled(config) if config else self._base_config
        result = extract_bytes_sync(content, mime_type=mime_type, config=resolved_config)
        return self._to_output(result, source=None)

    def _to_output(self, result: ExtractionResult, *, source: Path | None) -> LoaderOutput:
        metadata: dict[str, Any]
        metadata = dict(result.metadata or {})
        images = list(result.images or [])
        return LoaderOutput(
            text=result.content,
            mime_type=result.mime_type,
            metadata=metadata,
            images=images,
            source=source,
            raw_result=result,
        )
