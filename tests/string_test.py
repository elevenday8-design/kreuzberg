from __future__ import annotations

import pytest

from kreuzberg import ParsingError
from kreuzberg._string import normalize_spaces, safe_decode


@pytest.mark.parametrize(
    ("byte_data", "encoding", "expected"),
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


def test_safe_decode_with_detected_encoding() -> None:
    text = "Hello 世界"
    byte_data = text.encode("utf-8")
    assert safe_decode(byte_data) == text


def test_safe_decode_with_invalid_encoding() -> None:
    byte_data = b"\xff\xfe"
    result = safe_decode(byte_data)
    assert isinstance(result, str)


def test_safe_decode_with_fallback_encodings() -> None:
    text = "Hello World"
    byte_data = text.encode("utf-8")
    assert safe_decode(byte_data) == text


def test_safe_decode_no_valid_encoding() -> None:
    byte_data = b"\xff\xfe\xff\xfe"

    def mock_decode(self: bytes, encoding: str, errors: str = "strict") -> str:
        raise UnicodeDecodeError(encoding, byte_data, 0, 1, "mock error")

    byte_data_mock = type("MockBytes", (bytes,), {"decode": mock_decode})(byte_data)

    with pytest.raises(ParsingError) as exc:
        safe_decode(byte_data_mock)
    assert "Could not decode byte string" in str(exc.value)


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("hello  world", "hello world"),
        ("  hello   world  ", "hello world"),
        ("\thello\t\tworld\n", "hello world"),
        ("hello      world", "hello world"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_spaces(input_text: str, expected: str) -> None:
    assert normalize_spaces(input_text) == expected
