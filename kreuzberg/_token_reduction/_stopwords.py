from __future__ import annotations

from functools import cache
from pathlib import Path

import msgspec

from kreuzberg._utils._ref import Ref

_STOPWORDS_FILE = Path(__file__).parent / "stop_words.json"


@cache
def _load_default_stopwords() -> dict[str, set[str]]:
    with _STOPWORDS_FILE.open("rb") as f:
        data: dict[str, list[str]] = msgspec.json.decode(f.read())
    return {lang: set(words) for lang, words in data.items()}


def _create_default_manager() -> StopwordsManager:
    return StopwordsManager()


_default_manager_ref = Ref("default_stopwords_manager", _create_default_manager)


class StopwordsManager:
    def __init__(
        self,
        custom_stopwords: dict[str, list[str]] | None = None,
        stopwords_path: str | Path | None = None,
    ) -> None:
        self._custom_stopwords: dict[str, set[str]] = {}

        if custom_stopwords:
            self._custom_stopwords = {lang: set(words) for lang, words in custom_stopwords.items()}

        if stopwords_path:
            self._load_custom_stopwords_from_file(stopwords_path)

    def _load_custom_stopwords_from_file(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            msg = f"Stopwords file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open("rb") as f:
            data = msgspec.json.decode(f.read())

        if not isinstance(data, dict):
            msg = "Stopwords file must contain a JSON object"
            raise ValueError(msg)

        for lang, words in data.items():
            if not isinstance(words, list):
                msg = f"Stopwords for language '{lang}' must be a list"
                raise ValueError(msg)
            self._custom_stopwords[lang] = set(words)

    def _get_default_stopwords(self) -> dict[str, set[str]]:
        return _load_default_stopwords()

    def get_stopwords(self, language: str) -> set[str]:
        default_stopwords = self._get_default_stopwords()
        result = set()

        if language in default_stopwords:
            result.update(default_stopwords[language])

        if language in self._custom_stopwords:
            result.update(self._custom_stopwords[language])

        return result

    def has_language(self, language: str) -> bool:
        default_stopwords = self._get_default_stopwords()
        return language in default_stopwords or language in self._custom_stopwords

    def supported_languages(self) -> list[str]:
        default_stopwords = self._get_default_stopwords()
        all_langs = set(default_stopwords.keys())
        all_langs.update(self._custom_stopwords.keys())
        return sorted(all_langs)

    def add_custom_stopwords(self, language: str, words: list[str] | set[str]) -> None:
        if language not in self._custom_stopwords:
            self._custom_stopwords[language] = set()

        if isinstance(words, list):
            words = set(words)

        self._custom_stopwords[language].update(words)


def get_default_stopwords_manager() -> StopwordsManager:
    return _default_manager_ref.get()
