"""High-level orchestration helpers for loading and chunking documents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from kreuzberg._types import ExtractionConfig

from ragsdk.loader.kreuzberg_loader import KreuzbergLoader, LoaderOutput
from ragsdk.splitter import SplitDocument, TextSplitter


def _ensure_splitter(splitter: TextSplitter | None) -> TextSplitter:
    return splitter or TextSplitter()


def build_chunks(
    source: str | Path | bytes,
    *,
    mime_type: str | None = None,
    loader: KreuzbergLoader | None = None,
    splitter: TextSplitter | None = None,
    loader_config: ExtractionConfig | None = None,
) -> SplitDocument:
    """Load a document and split it into chunks using sensible defaults.

    Args:
        source: A file path or in-memory byte string. ``mime_type`` is required when
            ``source`` is bytes.
        mime_type: Optional mime type for file inputs. Required for in-memory bytes.
        loader: Optional custom :class:`KreuzbergLoader` to use.
        splitter: Optional :class:`TextSplitter` implementation.
        loader_config: Optional :class:`ExtractionConfig` overrides applied when a
            loader instance is created internally.

    Returns:
        :class:`SplitDocument` containing chunks, metadata and image references.
    """

    resolved_splitter = _ensure_splitter(splitter)
    resolved_loader = loader or KreuzbergLoader(config=loader_config)
    loader_kwargs = {"config": loader_config} if loader is not None and loader_config is not None else {}

    if isinstance(source, (str, Path)):
        document = resolved_loader.load_file_sync(source, mime_type=mime_type, **loader_kwargs)
    else:
        if mime_type is None:
            raise ValueError("mime_type must be provided when source is bytes")
        document = resolved_loader.load_bytes_sync(source, mime_type=mime_type, **loader_kwargs)

    return resolved_splitter.split(document)


def build_chunks_from_loader(
    documents: Iterable[LoaderOutput],
    *,
    splitter: TextSplitter | None = None,
) -> list[SplitDocument]:
    """Split pre-loaded documents using ``splitter`` or the default implementation."""

    resolved_splitter = _ensure_splitter(splitter)
    return resolved_splitter.split_many(documents)


def build_chunks_from_files(
    file_paths: Sequence[str | Path],
    *,
    loader: KreuzbergLoader | None = None,
    splitter: TextSplitter | None = None,
    loader_config: ExtractionConfig | None = None,
) -> list[SplitDocument]:
    """Convenience wrapper around :func:`build_chunks` for multiple files."""

    resolved_splitter = _ensure_splitter(splitter)
    resolved_loader = loader or KreuzbergLoader(config=loader_config)
    loader_kwargs = {"config": loader_config} if loader is not None and loader_config is not None else {}

    outputs = [resolved_loader.load_file_sync(path, **loader_kwargs) for path in file_paths]
    return resolved_splitter.split_many(outputs)

