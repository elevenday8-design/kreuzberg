from __future__ import annotations

import io
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

import anyio

from kreuzberg._extractors._base import Extractor
from kreuzberg._mime_types import PLAIN_TEXT_MIME_TYPE
from kreuzberg._types import ExtractionResult, normalize_metadata
from kreuzberg._utils._string import normalize_spaces, safe_decode

try:  # pragma: no cover - optional dependency
    import ezdxf  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled at runtime
    ezdxf = None  # type: ignore[assignment]


class DXFExtractor(Extractor):
    """Prototype extractor for DXF CAD drawings."""

    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        "image/vnd.dxf",
        "application/dxf",
        "application/x-dxf",
        "application/vnd.autocad.dxf",
    }

    _SUMMARY_ENTITY_LIMIT = 25

    def _summarize_with_ezdxf(self, text: str) -> tuple[str, dict[str, Any]]:
        assert ezdxf is not None  # for type checkers
        stream = io.StringIO(text)
        metadata: dict[str, Any] = {"source_format": "dxf", "mode": "ezdxf"}

        try:
            document = ezdxf.read(stream)
        except Exception as exc:  # pragma: no cover - ezdxf not available during tests
            metadata["parse_error"] = str(exc)
            metadata["mode"] = "fallback"
            return self._summarize_fallback(text), metadata

        layer_names = sorted(document.layers.names()) if document.layers else []
        if layer_names:
            metadata["layers"] = layer_names

        entity_counts = Counter(entity.dxftype() for entity in document.modelspace())
        if entity_counts:
            metadata["entity_counts"] = dict(entity_counts)

        summary_lines = ["DXF drawing summary"]
        if layer_names:
            summary_lines.append(f"Layers ({len(layer_names)}): {', '.join(layer_names[:8])}")
            if len(layer_names) > 8:
                summary_lines.append(f"… {len(layer_names) - 8} more layers")

        if entity_counts:
            summary_lines.append("Entity distribution:")
            for entity_name, count in entity_counts.most_common(self._SUMMARY_ENTITY_LIMIT):
                summary_lines.append(f"- {entity_name}: {count}")
            if len(entity_counts) > self._SUMMARY_ENTITY_LIMIT:
                summary_lines.append(
                    f"… {len(entity_counts) - self._SUMMARY_ENTITY_LIMIT} additional entity types"
                )
        else:
            summary_lines.append("No entities detected in model space")

        return "\n".join(summary_lines), metadata

    def _summarize_fallback(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections: list[str] = []
        for index, token in enumerate(lines[:-1]):
            if token.upper() == "SECTION":
                sections.append(lines[index + 1])

        summary_lines = ["DXF fallback summary"]
        if sections:
            unique_sections = []
            seen = set()
            for section in sections:
                key = section.upper()
                if key not in seen:
                    seen.add(key)
                    unique_sections.append(section)
            summary_lines.append(f"Sections ({len(unique_sections)}): {', '.join(unique_sections)}")
        preview = lines[: self._SUMMARY_ENTITY_LIMIT]
        if preview:
            summary_lines.append("Preview:")
            summary_lines.extend(f"- {line}" for line in preview)
        return "\n".join(summary_lines)

    def _summarize_content(self, content: bytes) -> tuple[str, dict[str, Any]]:
        text = safe_decode(content, errors="ignore")
        if ezdxf is not None:  # pragma: no branch - runtime behaviour only
            summary, metadata = self._summarize_with_ezdxf(text)
        else:
            summary = self._summarize_fallback(text)
            metadata = {
                "source_format": "dxf",
                "mode": "text_fallback",
                "warning": "ezdxf not installed – using text-level heuristics",
            }
        return summary, metadata

    def extract_bytes_sync(self, content: bytes) -> ExtractionResult:
        summary, metadata = self._summarize_content(content)
        result = ExtractionResult(
            content=normalize_spaces(summary),
            mime_type=PLAIN_TEXT_MIME_TYPE,
            metadata=normalize_metadata(metadata),
        )
        return self._apply_quality_processing(result)

    def extract_path_sync(self, path: Path) -> ExtractionResult:
        summary, metadata = self._summarize_content(path.read_bytes())
        result = ExtractionResult(
            content=normalize_spaces(summary),
            mime_type=PLAIN_TEXT_MIME_TYPE,
            metadata=normalize_metadata(metadata),
        )
        return self._apply_quality_processing(result)

    async def extract_bytes_async(self, content: bytes) -> ExtractionResult:
        return await anyio.to_thread.run_sync(self.extract_bytes_sync, content)

    async def extract_path_async(self, path: Path) -> ExtractionResult:
        return await anyio.to_thread.run_sync(self.extract_path_sync, path)
