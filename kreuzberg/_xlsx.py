from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from anyio import Path as AsyncPath
from anyio import create_task_group
from python_calamine import CalamineWorkbook

from kreuzberg import ExtractionResult, ParsingError
from kreuzberg._mime_types import MARKDOWN_MIME_TYPE
from kreuzberg._pandoc import process_file_with_pandoc
from kreuzberg._string import normalize_spaces
from kreuzberg._sync import run_sync


async def extract_xlsx_file(input_file: Path) -> ExtractionResult:
    """Extract text from an XLSX file by converting it to CSV and then to markdown.

    Args:
        input_file: The path to the XLSX file.

    Returns:
        The extracted text content.

    Raises:
        ParsingError: If the XLSX file could not be parsed.
    """
    try:
        workbook = await run_sync(CalamineWorkbook.from_path, str(input_file))

        results = cast(list[str], [None] * len(workbook.sheet_names))

        async def convert_sheet_to_text(sheet_name: str) -> None:
            nonlocal results
            values = await run_sync(workbook.get_sheet_by_name(sheet_name).to_python)

            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)

            for row in values:
                writer.writerow(row)

            csv_data = csv_buffer.getvalue()
            csv_buffer.close()

            with NamedTemporaryFile(suffix=".csv") as csv_file:
                await AsyncPath(csv_file.name).write_text(csv_data)
                result = await process_file_with_pandoc(csv_file.name, mime_type="text/csv")
                results[workbook.sheet_names.index(sheet_name)] = (
                    f"## {sheet_name}\n\n{normalize_spaces(result.content)}"
                )

        async with create_task_group() as tg:
            for sheet_name in workbook.sheet_names:
                tg.start_soon(convert_sheet_to_text, sheet_name)

        return ExtractionResult(
            content="\n\n".join(results),
            mime_type=MARKDOWN_MIME_TYPE,
            metadata={},
        )
    except Exception as e:
        raise ParsingError(
            "Could not extract text from XLSX",
            context={
                "error": str(e),
            },
        ) from e


async def extract_xlsx_content(content: bytes) -> ExtractionResult:
    """Extract text from an XLSX file content.

    Args:
        content: The XLSX file content.

    Returns:
        The extracted text content.
    """
    with NamedTemporaryFile(suffix=".xlsx") as xlsx_file:
        await AsyncPath(xlsx_file.name).write_bytes(content)
        return await extract_xlsx_file(Path(xlsx_file.name))
