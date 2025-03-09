from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from PIL import Image

from kreuzberg._ocr._tesseract import (
    TesseractBackend,
)
from kreuzberg._types import ExtractionResult, PSMMode
from kreuzberg.exceptions import MissingDependencyError, OCRError

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def backend() -> TesseractBackend:
    return TesseractBackend()


@pytest.fixture
def mock_run_process(mocker: MockerFixture) -> Mock:
    def run_sync(command: list[str], **kwargs: Any) -> Mock:
        result = Mock()
        result.stdout = b"tesseract 5.0.0"
        result.returncode = 0
        result.stderr = b""

        if "--version" in command and command[0].endswith("tesseract"):
            return result

        if "test_process_file_error" in str(kwargs.get("cwd")):
            result.returncode = 1
            result.stderr = b"Error processing file"
            raise OCRError("Error processing file")

        if "test_process_file_runtime_error" in str(kwargs.get("cwd")):
            raise RuntimeError("Command failed")

        if len(command) >= 3 and command[0].endswith("tesseract"):
            output_file = command[2]
            if "test_process_image_with_tesseract_invalid_input" in str(kwargs.get("cwd")):
                result.returncode = 1
                result.stderr = b"Error processing file"
                raise OCRError("Error processing file")

            if not all(arg in command for arg in ["--oem", "1", "--loglevel", "OFF", "-c", "thresholding_method=1"]):
                result.returncode = 1
                result.stderr = b"Missing required tesseract arguments"
                return result

            Path(f"{output_file}.txt").write_text("Sample OCR text")
            result.returncode = 0
            return result

        return result

    return mocker.patch("kreuzberg._ocr._tesseract.run_process", side_effect=run_sync)


@pytest.fixture
def mock_run_process_invalid(mocker: MockerFixture) -> Mock:
    def run_sync(command: list[str], **kwargs: Any) -> Mock:
        result = Mock()
        result.stdout = b"tesseract 4.0.0"
        result.returncode = 0
        result.stderr = b""
        return result

    return mocker.patch("kreuzberg._ocr._tesseract.run_process", side_effect=run_sync)


@pytest.fixture
def mock_run_process_error(mocker: MockerFixture) -> Mock:
    def run_sync(command: list[str], **kwargs: Any) -> Mock:
        raise FileNotFoundError

    return mocker.patch("kreuzberg._ocr._tesseract.run_process", side_effect=run_sync)


@pytest.mark.anyio
async def test_validate_tesseract_version(backend: TesseractBackend, mock_run_process: Mock) -> None:
    await backend._validate_tesseract_version()
    mock_run_process.assert_called_with(["tesseract", "--version"])


@pytest.fixture(autouse=True)
def reset_version_ref(mocker: MockerFixture) -> None:
    mocker.patch("kreuzberg._ocr._tesseract.version_ref", {"checked": False})


@pytest.mark.anyio
async def test_validate_tesseract_version_invalid(
    backend: TesseractBackend, mock_run_process_invalid: Mock, reset_version_ref: None
) -> None:
    with pytest.raises(MissingDependencyError, match="Tesseract version 5 or above is required"):
        await backend._validate_tesseract_version()


@pytest.mark.anyio
async def test_validate_tesseract_version_missing(
    backend: TesseractBackend, mock_run_process_error: Mock, reset_version_ref: None
) -> None:
    with pytest.raises(MissingDependencyError, match="Tesseract is not installed"):
        await backend._validate_tesseract_version()


@pytest.mark.anyio
async def test_process_file(backend: TesseractBackend, mock_run_process: Mock, ocr_image: Path) -> None:
    result = await backend.process_file(ocr_image, language="eng", psm=PSMMode.AUTO)
    assert isinstance(result, ExtractionResult)
    assert result.content.strip() == "Sample OCR text"


@pytest.mark.anyio
async def test_process_file_with_options(backend: TesseractBackend, mock_run_process: Mock, ocr_image: Path) -> None:
    result = await backend.process_file(ocr_image, language="eng", psm=PSMMode.AUTO)
    assert isinstance(result, ExtractionResult)
    assert result.content.strip() == "Sample OCR text"


@pytest.mark.anyio
async def test_process_file_error(backend: TesseractBackend, mock_run_process: Mock, ocr_image: Path) -> None:
    mock_run_process.return_value.returncode = 1
    mock_run_process.return_value.stderr = b"Error processing file"
    mock_run_process.side_effect = None
    with pytest.raises(OCRError, match="OCR failed with a non-0 return code"):
        await backend.process_file(ocr_image, language="eng", psm=PSMMode.AUTO)


@pytest.mark.anyio
async def test_process_file_runtime_error(backend: TesseractBackend, mock_run_process: Mock, ocr_image: Path) -> None:
    mock_run_process.side_effect = RuntimeError()
    with pytest.raises(OCRError, match="Failed to OCR using tesseract"):
        await backend.process_file(ocr_image, language="eng", psm=PSMMode.AUTO)


@pytest.mark.anyio
async def test_process_image(backend: TesseractBackend, mock_run_process: Mock) -> None:
    image = Image.new("RGB", (100, 100))
    result = await backend.process_image(image, language="eng", psm=PSMMode.AUTO)
    assert isinstance(result, ExtractionResult)
    assert result.content.strip() == "Sample OCR text"


@pytest.mark.anyio
async def test_process_image_with_tesseract_pillow(backend: TesseractBackend, mock_run_process: Mock) -> None:
    image = Image.new("RGB", (100, 100))
    result = await backend.process_image(image)
    assert isinstance(result, ExtractionResult)
    assert result.content.strip() == "Sample OCR text"


@pytest.mark.anyio
async def test_integration_process_file(backend: TesseractBackend, ocr_image: Path) -> None:
    result = await backend.process_file(ocr_image, language="eng", psm=PSMMode.AUTO)
    assert isinstance(result, ExtractionResult)
    assert result.content.strip()


@pytest.mark.anyio
async def test_integration_process_image(backend: TesseractBackend, ocr_image: Path) -> None:
    image = Image.open(ocr_image)
    with image:
        result = await backend.process_image(image, language="eng", psm=PSMMode.AUTO)
        assert isinstance(result, ExtractionResult)
        assert result.content.strip()


@pytest.mark.anyio
async def test_process_file_linux(backend: TesseractBackend, mocker: MockerFixture) -> None:
    mocker.patch("sys.platform", "linux")

    mock_run = mocker.patch("kreuzberg._ocr._tesseract.run_process")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = b"test output"

    await backend.process_file(Path("test.png"), language="eng", psm=PSMMode.AUTO)

    mock_run.assert_called_once()
    assert mock_run.call_args[1]["env"] == {"OMP_THREAD_LIMIT": "1"}
