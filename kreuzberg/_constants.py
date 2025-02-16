from __future__ import annotations

from multiprocessing import cpu_count
from typing import Final

DEFAULT_MAX_PROCESSES: Final[int] = max(cpu_count(), 1)
