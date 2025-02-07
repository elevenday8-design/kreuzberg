from __future__ import annotations

import re
import subprocess
from asyncio import gather
from enum import Enum
from os import PathLike, environ
from sys import platform
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from PIL.Image import Image

from kreuzberg._sync import run_sync
from kreuzberg.exceptions import MissingDependencyError

version_ref = {"checked": False}

SupportedLanguages = Literal[
    "afr",
    "amh",
    "ara",
    "asm",
    "aze",
    "aze_cyrl",
    "bel",
    "ben",
    "bod",
    "bos",
    "bre",
    "bul",
    "cat",
    "ceb",
    "ces",
    "chi_sim",
    "chi_tra",
    "chr",
    "cos",
    "cym",
    "dan",
    "dan_frak",
    "deu",
    "deu_frak",
    "deu_latf",
    "dzo",
    "ell",
    "eng",
    "enm",
    "epo",
    "equ",
    "est",
    "eus",
    "fao",
    "fas",
    "fil",
    "fin",
    "fra",
    "frk",
    "frm",
    "fry",
    "gla",
    "gle",
    "glg",
    "grc",
    "guj",
    "hat",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "iku",
    "ind",
    "isl",
    "ita",
    "ita_old",
    "jav",
    "jpn",
    "kan",
    "kat",
    "kat_old",
    "kaz",
    "khm",
    "kir",
    "kmr",
    "kor",
    "kor_vert",
    "kur",
    "lao",
    "lat",
    "lav",
    "lit",
    "ltz",
    "mal",
    "mar",
    "mkd",
    "mlt",
    "mon",
    "mri",
    "msa",
    "mya",
    "nep",
    "nld",
    "nor",
    "oci",
    "ori",
    "osd",
    "pan",
    "pol",
    "por",
    "pus",
    "que",
    "ron",
    "rus",
    "san",
    "sin",
    "slk",
    "slk_frak",
    "slv",
    "snd",
    "spa",
    "spa_old",
    "sqi",
    "srp",
    "srp_latn",
    "sun",
    "swa",
    "swe",
    "syr",
    "tam",
    "tat",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tir",
    "ton",
    "tur",
    "uig",
    "ukr",
    "urd",
    "uzb",
    "uzb_cyrl",
    "vie",
    "yid",
    "yor",
]


class PSMMode(Enum):
    """Enum for Tesseract Page Segmentation Modes (PSM) with human-readable values."""

    OSD_ONLY = 0  # Orientation and script detection only.
    AUTO_OSD = 1  # Automatic page segmentation with orientation and script detection.
    AUTO_ONLY = 2  # Automatic page segmentation without OSD.
    AUTO = 3  # Fully automatic page segmentation (default).
    SINGLE_COLUMN = 4  # Assume a single column of text.
    SINGLE_BLOCK_VERTICAL = 5  # Assume a single uniform block of vertically aligned text.
    SINGLE_BLOCK = 6  # Assume a single uniform block of text.
    SINGLE_LINE = 7  # Treat the image as a single text line.
    SINGLE_WORD = 8  # Treat the image as a single word.
    CIRCLE_WORD = 9  # Treat the image as a single word in a circle.
    SINGLE_CHAR = 10  # Treat the image as a single character.


async def validate_tesseract_version() -> None:
    """Validate that Tesseract is installed and is version 5 or above.

    Raises:
        RuntimeError: If Tesseract is not installed or is below version 5.
    """
    try:
        if version_ref["checked"]:
            return

        result = await run_sync(subprocess.run, ["tesseract", "--version"], capture_output=True)
        version_match = re.search(r"tesseract\s+(\d+)", result.stdout)
        if not version_match or int(version_match.group(1)) < 5:
            raise RuntimeError("Tesseract version 5 or above is required.")

        version_ref["checked"] = True
    except FileNotFoundError as e:
        raise MissingDependencyError("Tesseract is not installed.") from e


async def process_file(input_file: str | PathLike, *, language: SupportedLanguages, psm: PSMMode, **kwargs: Any) -> str:
    """Process a single image file using Tesseract OCR.

    Args:
        input_file: The path to the image file to process.
        language: The language code for OCR.
        psm: Page segmentation mode.
        **kwargs: Additional Tesseract configuration options as key-value pairs.

    Returns:
        str: Extracted text from the image.
    """
    with NamedTemporaryFile(suffix=".txt") as output_file:
        try:
            command = [
                "tesseract",
                str(input_file),
                output_file.name,
                "-l",
                language,
                "--psm",
                str(psm.value),
            ]

            for key, value in kwargs.items():
                command.extend(["-c", f"{key}={value}"])

            platform_kwargs = {}
            if platform == "win32":
                kwargs["startupinfo"] = subprocess.STARTUPINFO(
                    dwFlags=subprocess.STARTF_USESHOWWINDOW, wShowWindow=subprocess.SW_HIDE
                )

            process = await run_sync(
                subprocess.Popen,
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environ,
                stdout=subprocess.PIPE,
                **platform_kwargs,
            )

            await run_sync(process.wait)
            return (await output_file.read_text()).strip()
        except RuntimeError:
            raise


async def process_image(image: Image, *, language: SupportedLanguages, psm: PSMMode, **kwargs: Any) -> str:
    """Process a single Pillow Image using Tesseract OCR.

    Args:
        image: The Pillow Image to process.
        language: The language code for OCR.
        psm: Page segmentation mode.
        **kwargs: Additional Tesseract configuration options as key-value pairs.


    Returns:
        str: Extracted text from the image.
    """
    with NamedTemporaryFile(suffix=".png") as image_file, NamedTemporaryFile(suffix=".txt") as output_file:
        await run_sync(image.save, image_file.name, format="PNG")
        return await process_file(image_file.name, language=language, psm=psm, **kwargs)


async def extract_text(
    image: Image | PathLike | str, *, language: SupportedLanguages = "eng", psm: PSMMode = PSMMode.AUTO, **kwargs: Any
) -> str:
    """Run Tesseract OCR asynchronously on a single Pillow Image or a list of Pillow Images.

    Args:
        image: A single Pillow Image or a list of Pillow Images to process.
        language: The language code for OCR (default: "eng").
        psm: Page segmentation mode (default: PSMMode.AUTO).
        **kwargs: Additional Tesseract configuration options as key-value pairs.

    Raises:
        ValueError: If the input is not a Pillow Image or a list of Pillow Images.
        RuntimeError: If Tesseract is not installed or is below version 5.
        subprocess.CalledProcessError: If Tesseract CLI execution fails.

    Returns:
        Extracted text as a string (for single image) or a list of strings (for multiple images).
    """
    await validate_tesseract_version()

    if isinstance(image, Image):
        return await process_image(image, language=language, psm=psm, **kwargs)

    if isinstance(image, (PathLike, str)):
        return await process_file(image, language=language, psm=psm, **kwargs)

    raise ValueError("Input must be one of: str, Pathlike or Pillow Image.")


async def batch_extract_text(
    images: list[Image | PathLike | str],
    *,
    language: SupportedLanguages = "eng",
    psm: PSMMode = PSMMode.AUTO,
    **kwargs: Any,
) -> list[str]:
    """Run Tesseract OCR asynchronously on a single Pillow Image or a list of Pillow Images.

    Args:
        images: A single Pillow Image or a list of Pillow Images to process.
        language: The language code for OCR (default: "eng").
        psm: Page segmentation mode (default: PSMMode.AUTO).
        **kwargs: Additional Tesseract configuration options as key-value pairs.

    Raises:
        ValueError: If the input is not a Pillow Image or a list of Pillow Images.
        RuntimeError: If Tesseract is not installed or is below version 5.
        subprocess.CalledProcessError: If Tesseract CLI execution fails.

    Returns:
        Extracted text as a string (for single image) or a list of strings (for multiple images).
    """
    await validate_tesseract_version()
    return await gather(*[extract_text(image, language=language, psm=psm, **kwargs) for image in images])
