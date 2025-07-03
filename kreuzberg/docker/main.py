from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
import structlog
from litestar import Litestar, post
from litestar.contrib.opentelemetry import OpenTelemetryConfig
from litestar.exceptions import HTTPException
from litestar.status_codes import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR
from structlog.contextvars import bind_contextvars, clear_contextvars

from kreuzberg import ExtractionConfig, extract_bytes
from kreuzberg.exceptions import KreuzbergError

if TYPE_CHECKING:
    from litestar.datastructures import UploadFile as LitestarUploadFile

# Configure logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()


class ExtractionRequest(msgspec.Struct):
    """Represents a request to extract content from a file."""

    file: LitestarUploadFile


@post("/extract")
async def extract_from_file(data: ExtractionRequest) -> dict[str, Any]:
    """Extracts text content from an uploaded file."""
    clear_contextvars()
    bind_contextvars(filename=data.file.filename, content_type=data.file.content_type)

    try:
        logger.info("Receiving file")
        file_bytes = await data.file.read()

        config = ExtractionConfig(ocr_backend="tesseract")
        result = await extract_bytes(file_bytes, mime_type=data.file.content_type, config=config)

        logger.info("Successfully extracted content")
        return result.to_dict()

    except KreuzbergError as e:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Extraction failed: {e}") from e
    except Exception as e:
        logger.exception("An unexpected server error occurred")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred."
        ) from e


@post("/health")
async def health_check() -> dict[str, str]:
    """A simple health check endpoint."""
    return {"status": "ok"}


# Configure OpenTelemetry
opentelemetry_config = OpenTelemetryConfig(
    service_name="kreuzberg-api",
)

app = Litestar(
    route_handlers=[extract_from_file, health_check],
    middleware=[opentelemetry_config.middleware],
)
