from pathlib import Path
from tempfile import NamedTemporaryFile

from kreuzberg._sync import run_sync


async def create_temp_file(extension: str) -> Path:
    """Create a temporary file that is closed.

    Args:
        extension: The file extension.

    Returns:
        The temporary file path.
    """
    file = await run_sync(NamedTemporaryFile, suffix=extension, delete=False)
    await run_sync(file.close)
    return Path(file.name)
