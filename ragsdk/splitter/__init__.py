"""Text splitting utilities that operate on :mod:`ragsdk.loader` outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping

from kreuzberg._chunker import get_chunker
from kreuzberg._constants import DEFAULT_MAX_CHARACTERS, DEFAULT_MAX_OVERLAP
from kreuzberg._types import ExtractedImage

from ragsdk.loader.kreuzberg_loader import LoaderOutput


@dataclass(slots=True)
class SplitParameters:
    """Chunking parameters resolved for a particular document."""

    max_characters: int = DEFAULT_MAX_CHARACTERS
    overlap_characters: int = DEFAULT_MAX_OVERLAP


@dataclass(slots=True)
class TextChunk:
    """Represents a chunk of text derived from a loader output."""

    text: str
    metadata: dict[str, Any]
    index: int
    mime_type: str
    source: Path | None


@dataclass(slots=True)
class SplitDocument:
    """Container with chunked text and associated multimodal artefacts.

    Downstream retrievers should treat ``chunks`` as the ordered textual units to be
    indexed. ``images`` carries the extracted image artefacts and is intentionally
    separated so that future enrichment stages can manipulate images without
    modifying the text chunks. ``metadata`` contains document-level information that
    was present during extraction and should be merged with any additional retriever
    specific metadata prior to indexing.
    """

    chunks: list[TextChunk]
    images: list[ExtractedImage]
    metadata: dict[str, Any]
    mime_type: str
    source: Path | None


ChunkGenerator = Callable[[LoaderOutput, SplitParameters], Iterable[str]]
ParameterResolver = Callable[[LoaderOutput], SplitParameters]


def _default_parameter_resolver(_: LoaderOutput) -> SplitParameters:
    return SplitParameters()


class TextSplitter:
    """Utility that converts :class:`LoaderOutput` into text chunks."""

    def __init__(
        self,
        *,
        parameter_resolver: ParameterResolver | None = None,
        chunker_overrides: Mapping[str, ChunkGenerator] | None = None,
    ) -> None:
        self._parameter_resolver = parameter_resolver or _default_parameter_resolver
        self._chunker_overrides = dict(chunker_overrides or {})

    def with_override(self, mime_type: str, generator: ChunkGenerator) -> "TextSplitter":
        """Return a new splitter with an additional MIME specific override."""

        overrides: MutableMapping[str, ChunkGenerator] = dict(self._chunker_overrides)
        overrides[mime_type] = generator
        return TextSplitter(
            parameter_resolver=self._parameter_resolver,
            chunker_overrides=overrides,
        )

    def split(self, document: LoaderOutput) -> SplitDocument:
        params = self._parameter_resolver(document)
        metadata = dict(document.metadata)
        chunks = self._generate_chunks(document, params)
        text_chunks = [
            TextChunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": index},
                index=index,
                mime_type=document.mime_type,
                source=document.source,
            )
            for index, chunk_text in enumerate(chunks)
        ]

        return SplitDocument(
            chunks=text_chunks,
            images=list(document.images),
            metadata=metadata,
            mime_type=document.mime_type,
            source=document.source,
        )

    def split_many(self, documents: Iterable[LoaderOutput]) -> list[SplitDocument]:
        return [self.split(document) for document in documents]

    def _generate_chunks(self, document: LoaderOutput, params: SplitParameters) -> list[str]:
        override = self._chunker_overrides.get(document.mime_type)
        if override is not None:
            return list(override(document, params))

        chunker = get_chunker(
            mime_type=document.mime_type,
            max_characters=params.max_characters,
            overlap_characters=params.overlap_characters,
        )
        return list(chunker.chunks(document.text))


__all__ = [
    "SplitParameters",
    "TextChunk",
    "SplitDocument",
    "TextSplitter",
]

