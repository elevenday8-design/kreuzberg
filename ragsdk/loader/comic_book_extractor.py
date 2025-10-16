from __future__ import annotations

import io
from collections import Counter
from pathlib import Path
from typing import ClassVar

import anyio

from kreuzberg._extractors._base import Extractor
from kreuzberg._mime_types import PLAIN_TEXT_MIME_TYPE
from kreuzberg._types import ExtractionResult, normalize_metadata
from kreuzberg._utils._string import normalize_spaces
from zipfile import BadZipFile, ZipFile


class ComicBookArchiveExtractor(Extractor):
    """Text-focused extractor for CBZ comic book archives."""

    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        "application/vnd.comicbook+zip",
        "application/x-cbz",
    }

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".bmp"}

    def _summarize_zip(self, zip_file: ZipFile) -> tuple[str, dict[str, object]]:
        infos = zip_file.infolist()
        total_uncompressed = sum(info.file_size for info in infos)
        metadata: dict[str, object] = {
            "source_format": "cbz",
            "file_count": len(infos),
            "uncompressed_bytes": total_uncompressed,
        }

        extension_counter = Counter()
        image_entries: list[str] = []
        for info in infos:
            filename = info.filename
            suffix = Path(filename).suffix.lower()
            if suffix:
                extension_counter[suffix] += 1
            if suffix in self._IMAGE_EXTENSIONS:
                image_entries.append(filename)

        if extension_counter:
            metadata["extension_counts"] = dict(extension_counter)
        if image_entries:
            metadata["image_count"] = len(image_entries)

        summary_lines = ["Comic book archive summary"]
        if image_entries:
            summary_lines.append(f"Image entries: {len(image_entries)}")
            preview = image_entries[:5]
            summary_lines.append("Preview:")
            summary_lines.extend(f"- {name}" for name in preview)
            if len(image_entries) > len(preview):
                summary_lines.append(f"… {len(image_entries) - len(preview)} more image files")
        else:
            summary_lines.append("No image entries detected – archive treated as metadata-only")

        non_image_entries = [info.filename for info in infos if Path(info.filename).suffix.lower() not in self._IMAGE_EXTENSIONS]
        if non_image_entries:
            preview = non_image_entries[:5]
            summary_lines.append("Non-image entries:")
            summary_lines.extend(f"- {name}" for name in preview)
            if len(non_image_entries) > len(preview):
                summary_lines.append(f"… {len(non_image_entries) - len(preview)} more supporting files")

        return "\n".join(summary_lines), metadata

    def _extract_bytes(self, content: bytes) -> tuple[str, dict[str, object]]:
        try:
            with ZipFile(io.BytesIO(content)) as archive:
                return self._summarize_zip(archive)
        except BadZipFile as exc:
            metadata = {"source_format": "cbz", "warning": f"Invalid ZIP structure: {exc}"}
            return "Failed to open CBZ archive", metadata

    def extract_bytes_sync(self, content: bytes) -> ExtractionResult:
        summary, metadata = self._extract_bytes(content)
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
