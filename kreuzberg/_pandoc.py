from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, cast

from anyio import Path as AsyncPath

from kreuzberg._string import normalize_spaces
from kreuzberg._sync import run_sync
from kreuzberg.exceptions import MissingDependencyError, ParsingError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

try:  # pragma: no cover
    from typing import NotRequired  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import NotRequired

version_ref: Final[dict[str, bool]] = {"checked": False}


# Block-level node types in Pandoc AST
BLOCK_HEADER: Final = "Header"  # Header with level, attributes and inline content
BLOCK_PARA: Final = "Para"  # Paragraph containing inline content
BLOCK_CODE: Final = "CodeBlock"  # Code block with attributes and string content
BLOCK_QUOTE: Final = "BlockQuote"  # Block quote containing blocks
BLOCK_LIST: Final = "BulletList"  # Bullet list containing items (blocks)
BLOCK_ORDERED: Final = "OrderedList"  # Numbered list with attrs and items

# Inline-level node types in Pandoc AST
INLINE_STR: Final = "Str"  # Plain text string
INLINE_SPACE: Final = "Space"  # Single space
INLINE_EMPH: Final = "Emph"  # Emphasized text (contains inlines)
INLINE_STRONG: Final = "Strong"  # Strong/bold text (contains inlines)
INLINE_LINK: Final = "Link"  # Link with text and target
INLINE_IMAGE: Final = "Image"  # Image with alt text and source
INLINE_CODE: Final = "Code"  # Inline code span
INLINE_MATH: Final = "Math"  # Math expression

# Metadata node types in Pandoc AST
META_MAP: Final = "MetaMap"  # Key-value mapping of metadata
META_LIST: Final = "MetaList"  # List of metadata values
META_INLINES: Final = "MetaInlines"  # Inline content in metadata
META_STRING: Final = "MetaString"  # Plain string in metadata
META_BLOCKS: Final = "MetaBlocks"  # Block content in metadata

# Node content field name
CONTENT_FIELD: Final = "c"
TYPE_FIELD: Final = "t"

# Valid node types
NodeType = Literal[
    # Block types
    "Header",
    "Para",
    "CodeBlock",
    "BlockQuote",
    "BulletList",
    "OrderedList",
    # Inline types
    "Str",
    "Space",
    "Emph",
    "Strong",
    "Link",
    "Image",
    "Code",
    "Math",
    # Meta types
    "MetaMap",
    "MetaList",
    "MetaInlines",
    "MetaString",
    "MetaBlocks",
]

PandocFormat = Literal[
    "application/csl+json",
    "application/docbook+xml",
    "application/epub+zip",
    "application/rtf",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/x-biblatex",
    "application/x-bibtex",
    "application/x-endnote+xml",
    "application/x-fictionbook+xml",
    "application/x-ipynb+json",
    "application/x-jats+xml",
    "application/x-latex",
    "application/x-opml+xml",
    "application/x-research-info-systems",
    "application/x-typst",
    "application/xml",
    "text/csv",
    "text/tab-separated-values",
    "text/troff",
    "text/x-commonmark",
    "text/x-dokuwiki",
    "text/x-gfm",
    "text/x-markdown",
    "text/x-markdown-extra",
    "text/x-mdoc",
    "text/x-multimarkdown",
    "text/x-org",
    "text/x-pod",
    "text/x-rst",
]

PANDOC_MIME_TYPE_EXT_MAP: Final[Mapping[PandocFormat, str]] = {
    "application/csl+json": "json",
    "application/docbook+xml": "xml",
    "application/epub+zip": "epub",
    "application/rtf": "rtf",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/x-biblatex": "bib",
    "application/x-bibtex": "bib",
    "application/x-endnote+xml": "xml",
    "application/x-fictionbook+xml": "fb2",
    "application/x-ipynb+json": "ipynb",
    "application/x-jats+xml": "xml",
    "application/x-latex": "tex",
    "application/x-opml+xml": "opml",
    "application/x-research-info-systems": "ris",
    "application/x-typst": "typst",
    "application/xml": "xml",
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "text/troff": "man",
    "text/x-commonmark": "md",
    "text/x-dokuwiki": "txt",
    "text/x-gfm": "md",
    "text/x-markdown": "md",
    "text/x-markdown-extra": "md",
    "text/x-mdoc": "mdoc",
    "text/x-multimarkdown": "md",
    "text/x-org": "org",
    "text/x-pod": "pod",
    "text/x-rst": "rst",
}


class Metadata(TypedDict, total=False):
    """Document metadata extracted from Pandoc document.

    All fields are optional but will only be included if they contain non-empty values.
    Any field that would be empty or None is omitted from the dictionary.
    """

    title: NotRequired[str]
    """Document title."""
    subtitle: NotRequired[str]
    """Document subtitle."""
    abstract: NotRequired[str | list[str]]
    """Document abstract, summary or description."""
    authors: NotRequired[list[str]]
    """List of document authors."""
    date: NotRequired[str]
    """Document date as string to preserve original format."""
    subject: NotRequired[str]
    """Document subject or topic."""
    description: NotRequired[str]
    """Extended description."""
    keywords: NotRequired[list[str]]
    """Keywords or tags."""
    categories: NotRequired[list[str]]
    """Categories or classifications."""
    version: NotRequired[str]
    """Version identifier."""
    language: NotRequired[str]
    """Document language code."""
    references: NotRequired[list[str]]
    """Reference entries."""
    citations: NotRequired[list[str]]
    """Citation identifiers."""
    copyright: NotRequired[str]
    """Copyright information."""
    license: NotRequired[str]
    """License information."""
    identifier: NotRequired[str]
    """Document identifier."""
    publisher: NotRequired[str]
    """Publisher name."""
    contributors: NotRequired[list[str]]
    """Additional contributors."""
    creator: NotRequired[str]
    """Document creator."""
    institute: NotRequired[str | list[str]]
    """Institute or organization."""


@dataclass
class PandocResult:
    """Result of a pandoc conversion including content and metadata."""

    content: str
    """The processed markdown content."""
    metadata: Metadata
    """Document metadata extracted from the source."""


def _extract_inline_text(node: dict[str, Any]) -> str | None:
    if not isinstance(node, dict):
        return None

    if node_type := node.get(TYPE_FIELD):
        if node_type == INLINE_STR:
            return node.get(CONTENT_FIELD)
        if node_type == INLINE_SPACE:
            return " "
        if node_type in (INLINE_EMPH, INLINE_STRONG):
            return _extract_inlines(node.get(CONTENT_FIELD, []))
    return None


def _extract_inlines(nodes: list[dict[str, Any]]) -> str | None:
    if not isinstance(nodes, list):
        return None

    texts = [text for node in nodes if (text := _extract_inline_text(node))]
    result = "".join(texts).strip()
    return result if result else None


def _extract_meta_value(node: Any) -> str | list[str] | None:
    if isinstance(node, list):
        results: list[str] = []
        for value in [value for item in node if (value := _extract_meta_value(item))]:
            if isinstance(value, list):
                results.extend(value)
            else:
                results.append(value)
        return results

    if not isinstance(node, dict) or CONTENT_FIELD not in node or TYPE_FIELD not in node:
        return None

    content = node[CONTENT_FIELD]
    node_type = node[TYPE_FIELD]

    if not content or node_type not in {
        META_STRING,
        META_INLINES,
        META_LIST,
        META_BLOCKS,
    }:
        return None

    if node_type == META_STRING and isinstance(content, str):
        return content

    if isinstance(content, list) and (content := [v for v in content if isinstance(v, dict)]):
        if node_type == META_INLINES:
            return _extract_inlines(cast(list[dict[str, Any]], content))

        if node_type == META_LIST:
            results = []
            for value in [value for item in content if (value := _extract_meta_value(item))]:
                if isinstance(value, list):
                    results.extend(value)
                else:
                    results.append(value)
            return results

        if blocks := [block for block in content if block.get(TYPE_FIELD) == BLOCK_PARA]:
            block_texts = []
            for block in blocks:
                block_content = block.get(CONTENT_FIELD, [])
                if isinstance(block_content, list) and (text := _extract_inlines(block_content)):
                    block_texts.append(text)
            return block_texts if block_texts else None

    return None


def _extract_metadata(raw_meta: dict[str, Any]) -> Metadata:
    """Extract all non-empty metadata values from Pandoc AST metadata."""
    meta: Metadata = {}

    for key, value in raw_meta.items():
        if extracted := _extract_meta_value(value):
            meta[key] = extracted  # type: ignore[literal-required]

    citations = [
        cite["citationId"]
        for block in raw_meta.get("blocks", [])
        if block.get(TYPE_FIELD) == "Cite"
        for cite in block.get(CONTENT_FIELD, [[{}]])[0]
        if isinstance(cite, dict)
    ]
    if citations:
        meta["citations"] = citations

    return meta


def _get_extension_from_mime_type(mime_type: str) -> str:
    if mime_type not in PANDOC_MIME_TYPE_EXT_MAP or not any(
        mime_type.startswith(value) for value in PANDOC_MIME_TYPE_EXT_MAP
    ):
        raise ValidationError(
            f"Unsupported mime type: {mime_type}",
            context={"mime_type": mime_type, "supported_mimetypes": ",".join(sorted(PANDOC_MIME_TYPE_EXT_MAP))},
        )

    return PANDOC_MIME_TYPE_EXT_MAP.get(cast(PandocFormat, mime_type)) or next(
        PANDOC_MIME_TYPE_EXT_MAP[k] for k in PANDOC_MIME_TYPE_EXT_MAP if k.startswith(mime_type)
    )


async def validate_pandoc_version() -> None:
    """Validate that Pandoc is installed and is version 3 or above.

    Raises:
        MissingDependencyError: If Pandoc is not installed or is below version 3.
    """
    try:
        if version_ref["checked"]:
            return

        result = await run_sync(subprocess.run, ["pandoc", "--version"], capture_output=True)
        version = result.stdout.decode().split("\n")[0].split()[1]
        if not version.startswith("3."):
            raise MissingDependencyError("Pandoc version 3 or above is required.")

        version_ref["checked"] = True

    except FileNotFoundError as e:
        raise MissingDependencyError("Pandoc is not installed.") from e


async def extract_metadata(input_file: str | PathLike[str], *, mime_type: str) -> Metadata:
    """Extract metadata from a document using pandoc.

    Args:
        input_file: The path to the file to process.
        mime_type: The mime type of the file.

    Raises:
        ParsingError: If Pandoc fails to extract metadata.

    Returns:
        Dictionary containing document metadata.
    """
    extension = _get_extension_from_mime_type(mime_type)

    with NamedTemporaryFile(suffix=".json") as metadata_file:
        try:
            command = [
                "pandoc",
                str(input_file),
                f"--from={extension}",
                "--to=json",
                "--standalone",
                "--quiet",
                "--output",
                metadata_file.name,
            ]

            result = await run_sync(
                subprocess.run,
                command,
                capture_output=True,
            )

            if result.returncode != 0:
                raise ParsingError(
                    "Failed to extract file data", context={"file": str(input_file), "error": result.stderr.decode()}
                )

            json_data = json.loads(await AsyncPath(metadata_file.name).read_text())
            return _extract_metadata(json_data)

        except (RuntimeError, OSError, json.JSONDecodeError) as e:
            raise ParsingError(
                "Failed to extract file data", context={"file": str(input_file), "error": result.stderr.decode()}
            ) from e


async def process_file(
    input_file: str | PathLike[str], *, mime_type: str, extra_args: list[str] | None = None
) -> PandocResult:
    """Process a single file using Pandoc and convert to markdown.

    Args:
        input_file: The path to the file to process.
        mime_type: The mime type of the file.
        extra_args: Additional Pandoc command line arguments.

    Raises:
        ParsingError: If Pandoc fails to process the file.

    Returns:
        PandocResult containing processed content and metadata.
    """
    await validate_pandoc_version()

    metadata = await extract_metadata(input_file, mime_type=mime_type)
    extension = _get_extension_from_mime_type(mime_type)

    with NamedTemporaryFile(suffix=".md") as output_file:
        try:
            command = [
                "pandoc",
                str(input_file),
                f"--from={extension}",
                "--to=markdown",
                "--standalone",
                "--wrap=preserve",
                "--quiet",
                "--output",
                output_file.name,
            ]

            if extra_args:
                command.extend(extra_args)

            result = await run_sync(
                subprocess.run,
                command,
                capture_output=True,
            )

            if result.returncode != 0:
                ParsingError(
                    "Failed to extract file data", context={"file": str(input_file), "error": result.stderr.decode()}
                )

            content = normalize_spaces(await AsyncPath(output_file.name).read_text())
            return PandocResult(
                content=content,
                metadata=metadata,
            )

        except (RuntimeError, OSError) as e:
            raise ParsingError(
                "Failed to extract file data", context={"file": str(input_file), "error": result.stderr.decode()}
            ) from e


async def process_content(content: bytes, *, mime_type: str, extra_args: list[str] | None = None) -> PandocResult:
    """Process content using Pandoc and convert to markdown.

    Args:
        content: The content to process.
        mime_type: The mime type of the content.
        extra_args: Additional Pandoc command line arguments.

    Returns:
        PandocResult containing processed content and metadata.
    """
    extension = _get_extension_from_mime_type(mime_type)

    with NamedTemporaryFile(suffix=f".{extension}") as input_file:
        await AsyncPath(input_file.name).write_bytes(content)
        return await process_file(input_file.name, mime_type=mime_type, extra_args=extra_args)
