from __future__ import annotations

import pytest

from kreuzberg._string import safe_decode


@pytest.mark.parametrize(
    "byte_data, encoding, expected",
    [
        (b"hello", "utf-8", "hello"),
        (b"hello", None, "hello"),
        (b"caf\xc3\xa9", "utf-8", "café"),
        (b"caf\xe9", "latin-1", "café"),
        (b"", "utf-8", ""),
        (b"", None, ""),
    ],
)
def test_safe_decode(byte_data: bytes, encoding: str | None, expected: str) -> None:
    assert safe_decode(byte_data, encoding) == expected


@pytest.mark.parametrize(
    "byte_data, expected",
    [
        (b"caf\x81", "caf\x81"),
        (b"caf\xf0\x28\x8c\xbc", "caf"),
        (b"caf\xe9hello", "caféhello"),
    ],
)
def test_safe_decode_fallback(byte_data: bytes, expected: str) -> None:
    result = safe_decode(byte_data)
    assert result.startswith("caf")
    assert all(ord(char) < 128 or char.isprintable() for char in result)
