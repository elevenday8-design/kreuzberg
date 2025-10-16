# RAG SDK Pipelines

The `ragsdk` namespace exposes light-weight helpers that compose Kreuzberg's
extraction capabilities with project specific chunking. The new helpers are
designed to keep textual and visual artefacts separate so multimodal retrieval
systems can reason about each modality independently.

## Loader contract

`ragsdk.loader.KreuzbergLoader` wraps the core `extract_file` and
`extract_bytes` functions while forcing `ExtractionConfig(chunk_content=False)`.
This ensures downstream chunking strategies always operate on raw text instead
of pre-generated chunks.

The loader returns a `LoaderOutput` data structure with:

* `text`: extracted textual content.
* `metadata`: shallow copy of document level metadata.
* `images`: a list of `ExtractedImage` objects.
* `mime_type` and `source`: retained for traceability.
* `raw_result`: access to the underlying `ExtractionResult` for advanced use
  cases.

## Chunking contract

`ragsdk.splitter.TextSplitter` accepts `LoaderOutput` instances and produces a
`SplitDocument`:

* `chunks`: ordered text chunks with a `chunk_index` value embedded in their
  metadata.
* `images`: the original `ExtractedImage` references from the loader stage.
* `metadata`: document-level metadata preserved for retrievers.

Retrievers should index text using the chunk list while treating `images` as a
parallel modality that can be enriched (for example by additional vision
models) before indexing.

Custom projects can override chunking behaviour via `TextSplitter`'s
`parameter_resolver` or MIME-specific overrides.

## Pipeline orchestration

`ragsdk.pipeline.build_chunks` combines loader and splitter steps. It accepts
either a file path or in-memory bytes and returns a `SplitDocument`. The
`build_chunks_from_loader` and `build_chunks_from_files` helpers make it easy to
process pre-loaded documents or multiple file paths, while still leaving room
for custom loaders or splitters.

These utilities provide a safe baseline for multimodal retrieval pipelines by
ensuring text and image artefacts remain separate until retrievers decide how
to index or post-process them.

