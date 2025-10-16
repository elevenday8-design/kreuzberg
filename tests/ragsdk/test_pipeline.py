from __future__ import annotations

import pytest

from kreuzberg._types import ExtractionConfig, ExtractedImage

from ragsdk.loader import KreuzbergLoader
from ragsdk.splitter import SplitParameters, TextSplitter
from ragsdk.pipeline import build_chunks, build_chunks_from_loader


def test_loader_disables_chunking() -> None:
    loader = KreuzbergLoader(config=ExtractionConfig(chunk_content=True))

    document = loader.load_bytes_sync(b"Hello world", mime_type="text/plain")

    assert document.raw_result.chunks == []
    assert document.text == "Hello world"


def test_text_splitter_uses_parameter_resolver(tmp_path) -> None:
    sample_text = "One two three four five six"
    path = tmp_path / "example.txt"
    path.write_text(sample_text, encoding="utf-8")

    loader = KreuzbergLoader()
    document = loader.load_file_sync(path)

    splitter = TextSplitter(
        parameter_resolver=lambda _: SplitParameters(max_characters=10, overlap_characters=0)
    )

    split_document = splitter.split(document)

    assert len(split_document.chunks) >= 2
    assert [chunk.metadata["chunk_index"] for chunk in split_document.chunks] == list(
        range(len(split_document.chunks))
    )
    assert split_document.images == []


def test_build_chunks_from_loader_preserves_images(tmp_path) -> None:
    path = tmp_path / "image_doc.txt"
    path.write_text("content", encoding="utf-8")

    loader = KreuzbergLoader()
    document = loader.load_file_sync(path)

    # Attach an image reference to ensure the pipeline keeps multimodal data.
    image = ExtractedImage(data=b"image-bytes", format="png", filename="image.png")
    document.raw_result.images.append(image)
    document.images.append(image)

    split_documents = build_chunks_from_loader([document])

    assert len(split_documents) == 1
    assert split_documents[0].images == [image]
    assert split_documents[0].chunks[0].text


def test_build_chunks_requires_mime_for_bytes() -> None:
    with pytest.raises(ValueError):
        build_chunks(b"hello")


def test_build_chunks_creates_document(tmp_path) -> None:
    path = tmp_path / "document.txt"
    text = "alpha beta gamma delta"
    path.write_text(text, encoding="utf-8")

    result = build_chunks(path)

    assert result.metadata == {}
    assert result.chunks[0].text
    assert result.images == []

