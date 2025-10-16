from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping

import httpx

from kreuzberg._ocr._base import OCRBackend
from kreuzberg._types import ExtractionResult


@dataclass(slots=True)
class NASOCRConfig:
    """Configuration for the Network Attached Storage OCR backend."""

    endpoint: str
    """HTTP endpoint of the NAS OCR service."""

    api_key: str | None = None
    """Optional API key that will be sent as a bearer token."""

    username: str | None = None
    """Optional username for basic authentication."""

    password: str | None = None
    """Optional password for basic authentication."""

    chunk_size: int = 1024 * 512
    """Size of streamed chunks when sending image data."""

    request_timeout: float = 60.0
    """Request timeout in seconds for NAS OCR requests."""

    extra_headers: Mapping[str, str] | None = None
    """Additional headers to include in each request."""

    use_cache: bool = True
    """Maintained for compatibility with ExtractionConfig."""


class NASOCRBackend(OCRBackend[NASOCRConfig]):
    """OCR backend that streams image bytes to a NAS OCR service."""

    supports_file_streaming: bool = True

    def __init__(self, config: NASOCRConfig | None = None) -> None:
        self._config = config

    async def process_image(self, image: Any, **kwargs: Any) -> ExtractionResult:
        config = self._resolve_config(kwargs)
        with self._image_to_temp_file(image) as path:
            return await self._process_file_async(path, config, self._detect_mime_type(path))

    async def process_file(self, path: Path, **kwargs: Any) -> ExtractionResult:
        config = self._resolve_config(kwargs)
        return await self._process_file_async(path, config, self._detect_mime_type(path))

    def process_image_sync(self, image: Any, **kwargs: Any) -> ExtractionResult:
        config = self._resolve_config(kwargs)
        with self._image_to_temp_file(image) as path:
            return self._process_file_sync(path, config, self._detect_mime_type(path))

    def process_file_sync(self, path: Path, **kwargs: Any) -> ExtractionResult:
        config = self._resolve_config(kwargs)
        return self._process_file_sync(path, config, self._detect_mime_type(path))

    def _resolve_config(self, overrides: MutableMapping[str, Any]) -> NASOCRConfig:
        provided = overrides.pop("nas_config", None)
        if isinstance(provided, NASOCRConfig):
            base = provided
        elif self._config is not None:
            base = self._config
        else:
            if "endpoint" not in overrides:
                raise ValueError("NAS OCR endpoint must be provided through NASOCRConfig or keyword arguments")
            base = NASOCRConfig(endpoint=str(overrides["endpoint"]))

        valid_fields = set(NASOCRConfig.__dataclass_fields__.keys())
        filtered: Dict[str, Any] = {k: overrides[k] for k in list(overrides.keys()) if k in valid_fields}

        if "endpoint" in filtered:
            filtered["endpoint"] = str(filtered["endpoint"])

        return replace(base, **filtered)

    async def _process_file_async(self, path: Path, config: NASOCRConfig, mime_type: str) -> ExtractionResult:
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.post(
                config.endpoint,
                headers=self._build_headers(config, mime_type),
                content=self._iter_file_chunks(path, config.chunk_size),
                auth=self._build_auth(config),
            )
        response.raise_for_status()
        return self._parse_response(response.json(), default_mime="text/plain")

    def _process_file_sync(self, path: Path, config: NASOCRConfig, mime_type: str) -> ExtractionResult:
        with httpx.Client(timeout=config.request_timeout) as client:
            response = client.post(
                config.endpoint,
                headers=self._build_headers(config, mime_type),
                content=self._iter_file_chunks(path, config.chunk_size),
                auth=self._build_auth(config),
            )
        response.raise_for_status()
        return self._parse_response(response.json(), default_mime="text/plain")

    def _build_headers(self, config: NASOCRConfig, mime_type: str) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": mime_type}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        if config.extra_headers:
            headers.update(dict(config.extra_headers))
        return headers

    def _build_auth(self, config: NASOCRConfig) -> tuple[str, str] | None:
        if config.username and config.password:
            return (config.username, config.password)
        return None

    def _parse_response(self, payload: Mapping[str, Any], default_mime: str) -> ExtractionResult:
        content = str(payload.get("content") or payload.get("text") or "")
        mime_type = str(payload.get("mime_type") or default_mime)
        metadata = payload.get("metadata")
        metadata_dict: Dict[str, Any]
        if isinstance(metadata, Mapping):
            metadata_dict = dict(metadata)
        else:
            metadata_dict = {}

        result = ExtractionResult(content=content, mime_type=mime_type, metadata=metadata_dict)

        for field in ("chunks", "tables", "images", "image_ocr_results", "entities", "keywords"):
            value = payload.get(field)
            if value is not None:
                setattr(result, field, value)

        for field in ("detected_languages", "document_type", "document_type_confidence"):
            if field in payload:
                setattr(result, field, payload[field])

        if "layout" in payload:
            setattr(result, "layout", payload["layout"])

        return result

    @staticmethod
    def _iter_file_chunks(path: Path, chunk_size: int) -> Iterator[bytes]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    @staticmethod
    def _detect_mime_type(path: Path) -> str:
        import mimetypes

        mime_type, _ = mimetypes.guess_type(path.name)
        return mime_type or "application/octet-stream"

    @contextmanager
    def _image_to_temp_file(self, image: Any) -> Iterator[Path]:
        from PIL import Image
        import tempfile

        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.open(image)

        suffix = f".{(pil_image.format or 'png').lower()}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        pil_image.save(temp_path)

        try:
            yield temp_path
        finally:
            temp_path.unlink(missing_ok=True)
            pil_image.close()


__all__ = ["NASOCRBackend", "NASOCRConfig"]
