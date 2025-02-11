from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Ref(Generic[T]):
    """A reference to a value that can be modified."""

    value: T | None = None
    """The value referenced by this Ref."""
