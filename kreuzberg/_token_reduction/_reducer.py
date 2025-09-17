from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, TypedDict

from kreuzberg._token_reduction._stopwords import get_default_stopwords_manager
from kreuzberg.exceptions import ValidationError

if TYPE_CHECKING:
    from kreuzberg._types import TokenReductionConfig


class ReductionStats(TypedDict):
    """Statistics about token reduction operation."""

    character_reduction_ratio: float
    token_reduction_ratio: float
    original_characters: int
    reduced_characters: int
    original_tokens: int
    reduced_tokens: int


HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
REPEATED_PUNCTUATION_PATTERNS = (
    re.compile(r"([!?.]){2,}"),
    re.compile(r"(,){2,}"),
)
WHITESPACE_PATTERNS = (
    re.compile(r"\n{3,}"),
    re.compile(r"[ \t]+"),
)
MARKDOWN_PATTERNS = (
    re.compile(r"^\s*[-*+]\s"),
    re.compile(r"^\s*\d+\.\s"),
)
WORD_CLEAN_PATTERN = re.compile(r"[^\w]")
EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{3,}")


def _normalize_newlines(text: str) -> str:
    """Remove excessive newlines, keeping at most double newlines."""
    return EXCESSIVE_NEWLINES_PATTERN.sub("\n\n", text)


def _is_markdown_structural_line(line: str, in_code_block: bool) -> bool:
    """Check if a line contains markdown structural elements that should be preserved."""
    if in_code_block:
        return True

    stripped = line.strip()
    return (
        stripped.startswith("#")
        or "|" in line
        or MARKDOWN_PATTERNS[0].match(line) is not None
        or MARKDOWN_PATTERNS[1].match(line) is not None
    )


@lru_cache(maxsize=128)
def _get_cached_stopwords(language: str) -> set[str]:
    return get_default_stopwords_manager().get_stopwords(language)


@lru_cache(maxsize=64)
def _get_cached_custom_stopwords(language: str, custom_words_tuple: tuple[str, ...]) -> set[str]:
    """Get stopwords with custom words efficiently cached."""
    base_stopwords = _get_cached_stopwords(language)
    custom_words = set(custom_words_tuple)
    return base_stopwords | custom_words


def reduce_tokens(
    text: str,
    *,
    config: TokenReductionConfig,
    language: str | None = None,
) -> str:
    """Reduce tokens in text based on the specified configuration."""
    if config is None:
        raise ValidationError("Config cannot be None")

    if text is None:
        raise ValidationError("Text cannot be None")

    if not isinstance(text, str):
        raise ValidationError(f"Text must be a string, got {type(text).__name__}")

    if language is not None and not isinstance(language, str):
        raise ValidationError(f"Language must be a string or None, got {type(language).__name__}")

    if language is not None and len(language.strip()) == 0:
        raise ValidationError("Language cannot be empty or whitespace-only")

    if len(text) > 10_000_000:
        raise ValidationError(f"Text too large: {len(text)} characters (max 10,000,000)")

    if language is not None and not re.match(r"^[a-zA-Z0-9-]+$", language):
        raise ValidationError(f"Invalid language code format: {language}")

    if config.mode == "off":
        return text

    if not text.strip():
        return ""

    if config.mode == "light":
        return _apply_light_reduction(text, preserve_markdown=config.preserve_markdown)
    if config.mode == "moderate":
        return _apply_moderate_reduction(
            text,
            config=config,
            language=language,
        )

    return text


def _apply_light_reduction(text: str, *, preserve_markdown: bool) -> str:
    if preserve_markdown:
        return _apply_light_reduction_markdown_aware(text)
    return _apply_light_reduction_plain(text)


def _apply_light_reduction_plain(text: str) -> str:
    text = HTML_COMMENT_PATTERN.sub("", text)

    text = REPEATED_PUNCTUATION_PATTERNS[0].sub(r"\1", text)
    text = REPEATED_PUNCTUATION_PATTERNS[1].sub(r"\1", text)

    text = WHITESPACE_PATTERNS[0].sub("\n\n", text)
    text = WHITESPACE_PATTERNS[1].sub(" ", text)

    return text.strip()


def _apply_light_reduction_markdown_aware(text: str) -> str:
    lines = text.split("\n")
    processed_lines = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            processed_lines.append(line)
            continue

        if _is_markdown_structural_line(line, in_code_block):
            processed_lines.append(line)
            continue

        if line.strip():
            reduced = _apply_light_reduction_plain(line)
            processed_lines.append(reduced)
        else:
            processed_lines.append(line)

    result = "\n".join(processed_lines)
    return _normalize_newlines(result).strip()


def _apply_moderate_reduction(
    text: str,
    *,
    config: TokenReductionConfig,
    language: str | None = None,
) -> str:
    text = _apply_light_reduction(text, preserve_markdown=config.preserve_markdown)

    lang = language or config.language_hint or "en"

    manager = get_default_stopwords_manager()
    if not manager.has_language(lang):
        lang = "en"

    if not manager.has_language(lang):
        return text

    if config.custom_stopwords and lang in config.custom_stopwords:
        custom_words_tuple = tuple(sorted(config.custom_stopwords[lang]))
        stopwords = _get_cached_custom_stopwords(lang, custom_words_tuple)
    else:
        stopwords = _get_cached_stopwords(lang)

    if config.preserve_markdown:
        return _apply_stopword_reduction_markdown_aware(text, stopwords=stopwords)
    return _apply_stopword_reduction_plain(text, stopwords=stopwords)


def _apply_stopword_reduction_plain(text: str, *, stopwords: set[str]) -> str:
    words = text.split()
    filtered_words = []

    for word in words:
        clean_word = WORD_CLEAN_PATTERN.sub("", word.lower())

        if (
            clean_word not in stopwords
            or len(clean_word) <= 2
            or word.isupper()
            or any(char.isdigit() for char in word)
        ):
            filtered_words.append(word)

    return " ".join(filtered_words)


def _apply_stopword_reduction_markdown_aware(text: str, *, stopwords: set[str]) -> str:
    lines = text.split("\n")
    processed_lines = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            processed_lines.append(line)
            continue

        if _is_markdown_structural_line(line, in_code_block):
            processed_lines.append(line)
            continue

        if line.strip():
            reduced = _apply_stopword_reduction_plain(line, stopwords=stopwords)
            processed_lines.append(reduced)
        else:
            processed_lines.append(line)

    result = "\n".join(processed_lines)
    return _normalize_newlines(result).strip()


def get_reduction_stats(original: str, reduced: str) -> ReductionStats:
    """Get detailed statistics about the reduction."""
    if original is None:
        raise ValidationError("Original text cannot be None")

    if reduced is None:
        raise ValidationError("Reduced text cannot be None")

    if not isinstance(original, str):
        raise ValidationError(f"Original text must be a string, got {type(original).__name__}")

    if not isinstance(reduced, str):
        raise ValidationError(f"Reduced text must be a string, got {type(reduced).__name__}")

    original_chars = len(original)
    reduced_chars = len(reduced)
    original_tokens = len(original.split())
    reduced_tokens = len(reduced.split())

    char_reduction = (original_chars - reduced_chars) / original_chars if original_chars > 0 else 0.0
    token_reduction = (original_tokens - reduced_tokens) / original_tokens if original_tokens > 0 else 0.0

    return {
        "character_reduction_ratio": char_reduction,
        "token_reduction_ratio": token_reduction,
        "original_characters": original_chars,
        "reduced_characters": reduced_chars,
        "original_tokens": original_tokens,
        "reduced_tokens": reduced_tokens,
    }
