from __future__ import annotations

import struct
from pathlib import Path
from typing import ClassVar, Iterable

import anyio

from kreuzberg._extractors._base import Extractor
from kreuzberg._mime_types import PLAIN_TEXT_MIME_TYPE
from kreuzberg._types import ExtractionResult, normalize_metadata
from kreuzberg._utils._string import normalize_spaces, safe_decode


class STLExtractor(Extractor):
    """Prototype extractor for STL mesh data."""

    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        "model/stl",
        "application/sla",
        "application/vnd.ms-pki.stl",
        "application/x-navistyle",
    }

    _ASCII_FACET_TOKEN = "facet normal"

    def _compute_bounds(self, vertices: Iterable[tuple[float, float, float]]):
        xs, ys, zs = zip(*vertices)
        return {
            "xmin": min(xs),
            "xmax": max(xs),
            "ymin": min(ys),
            "ymax": max(ys),
            "zmin": min(zs),
            "zmax": max(zs),
        }

    def _summarize_ascii(self, text: str) -> tuple[str, dict[str, object]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        metadata: dict[str, object] = {"source_format": "stl", "mode": "ascii"}

        name = lines[0].split(maxsplit=1)[1] if lines and lines[0].startswith("solid ") else ""
        if name:
            metadata["solid"] = name

        facet_count = sum(1 for line in lines if line.startswith(self._ASCII_FACET_TOKEN))
        metadata["facet_count"] = facet_count

        vertices: list[tuple[float, float, float]] = []
        for line in lines:
            if line.startswith("vertex "):
                try:
                    x_str, y_str, z_str = line.split()[1:4]
                    vertices.append((float(x_str), float(y_str), float(z_str)))
                except (ValueError, IndexError):  # pragma: no cover - defensive
                    continue

        if vertices:
            metadata["bounds"] = self._compute_bounds(vertices)

        summary_lines = ["STL mesh summary (ASCII)"]
        if name:
            summary_lines.append(f"Solid name: {name}")
        summary_lines.append(f"Facets detected: {facet_count}")
        if vertices:
            bounds = metadata["bounds"]
            summary_lines.append(
                "Bounds: x=({xmin:.3f}, {xmax:.3f}), y=({ymin:.3f}, {ymax:.3f}), z=({zmin:.3f}, {zmax:.3f})".format(
                    **bounds  # type: ignore[arg-type]
                )
            )
        else:
            summary_lines.append("No vertex coordinates detected")
        return "\n".join(summary_lines), metadata

    def _summarize_binary(self, content: bytes) -> tuple[str, dict[str, object]]:
        metadata: dict[str, object] = {"source_format": "stl", "mode": "binary"}
        if len(content) < 84:
            metadata["warning"] = "Binary STL shorter than header"
            return "Binary STL file is truncated", metadata

        header = content[:80].rstrip(b"\x00").decode("ascii", errors="ignore")
        triangle_count = struct.unpack("<I", content[80:84])[0]
        metadata["facet_count"] = triangle_count
        if header:
            metadata["header"] = header

        preview = header or "binary STL"
        summary_lines = ["STL mesh summary (binary)", f"Header preview: {preview[:60]}".rstrip(), f"Facets declared: {triangle_count}"]
        return "\n".join(summary_lines), metadata

    def _summarize_content(self, content: bytes) -> tuple[str, dict[str, object]]:
        stripped = content.lstrip()
        if stripped.startswith(b"solid"):
            text = safe_decode(content, errors="ignore")
            return self._summarize_ascii(text)
        return self._summarize_binary(content)

    def extract_bytes_sync(self, content: bytes) -> ExtractionResult:
        summary, metadata = self._summarize_content(content)
        result = ExtractionResult(
            content=normalize_spaces(summary),
            mime_type=PLAIN_TEXT_MIME_TYPE,
            metadata=normalize_metadata(metadata),
        )
        return self._apply_quality_processing(result)

    def extract_path_sync(self, path: Path) -> ExtractionResult:
        return self.extract_bytes_sync(path.read_bytes())

    async def extract_bytes_async(self, content: bytes) -> ExtractionResult:
        return await anyio.to_thread.run_sync(self.extract_bytes_sync, content)

    async def extract_path_async(self, path: Path) -> ExtractionResult:
        return await anyio.to_thread.run_sync(self.extract_path_sync, path)
