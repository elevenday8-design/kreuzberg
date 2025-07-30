from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from kreuzberg import MissingDependencyError
from kreuzberg import _entity_extraction as ee
from kreuzberg._entity_extraction import SpacyEntityExtractionConfig
from kreuzberg._types import Entity

SAMPLE_TEXT = "John Doe visited Berlin on 2023-01-01. Contact: john@example.com or +49-123-4567."


@pytest.mark.parametrize(
    "custom_patterns,expected",
    [
        (None, []),
        (frozenset([("INVOICE_ID", r"INV-\d+")]), []),
        (
            frozenset([("EMAIL", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")]),
            [Entity(type="EMAIL", text="john@example.com", start=48, end=64)],
        ),
    ],
)
def test_custom_entity_patterns(
    custom_patterns: frozenset[tuple[str, str]] | None, expected: list[Entity], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "spacy", MagicMock())
    entities = ee.extract_entities(SAMPLE_TEXT, entity_types=(), custom_patterns=custom_patterns)
    assert all(isinstance(e, Entity) for e in entities)
    if expected:
        assert any(e.type == "EMAIL" and e.text == "john@example.com" for e in entities)
    else:
        assert entities == expected


def test_extract_entities_with_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEnt:
        def __init__(self, label: str, text: str, start_char: int, end_char: int):
            self.label_ = label
            self.text = text
            self.start_char = start_char
            self.end_char = end_char

    class DummyDoc:
        def __init__(self, text: str):
            self.ents = [
                DummyEnt("PERSON", "John Doe", 0, 8),
                DummyEnt("GPE", "Berlin", 18, 24),
            ]

    class DummyNLP:
        max_length = 1000000

        def __call__(self, text: str) -> DummyDoc:
            return DummyDoc(text)

    def mock_load(_model_name: str) -> DummyNLP:
        return DummyNLP()

    mock_spacy = MagicMock()
    mock_spacy.load = mock_load
    monkeypatch.setitem(sys.modules, "spacy", mock_spacy)

    def mock_load_spacy_model(model_name: str, spacy_config: Any) -> DummyNLP:
        return DummyNLP()

    monkeypatch.setattr(ee, "_load_spacy_model", mock_load_spacy_model)

    result = ee.extract_entities(SAMPLE_TEXT, entity_types=["PERSON", "GPE"], languages=["en"])
    assert any(e.type == "PERSON" and e.text == "John Doe" for e in result)
    assert any(e.type == "GPE" and e.text == "Berlin" for e in result)
    assert all(isinstance(e, Entity) for e in result)


def test_extract_keywords_with_keybert(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyModel:
        def extract_keywords(self, _text: str, top_n: int = 10) -> list[tuple[str, float]]:
            if top_n == 2:
                return [("Berlin", 0.9), ("John Doe", 0.8)]
            return [("keyword", 0.5)] * top_n

    mock_keybert = MagicMock()
    mock_keybert.KeyBERT = DummyModel
    monkeypatch.setitem(sys.modules, "keybert", mock_keybert)

    result = ee.extract_keywords(SAMPLE_TEXT, keyword_count=2)
    assert result == [("Berlin", 0.9), ("John Doe", 0.8)]


def test_extract_entities_missing_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "spacy", None)
    with pytest.raises(MissingDependencyError):
        ee.extract_entities(SAMPLE_TEXT, entity_types=["PERSON"])


def test_extract_keywords_missing_keybert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keybert", None)
    with pytest.raises(MissingDependencyError):
        ee.extract_keywords(SAMPLE_TEXT, keyword_count=5)


def test_spacy_entity_extraction_config_defaults() -> None:
    """Test SpacyEntityExtractionConfig default values."""
    config = SpacyEntityExtractionConfig()
    assert config.language_models is not None
    assert isinstance(config.language_models, tuple)
    assert config.fallback_to_multilingual is True
    assert config.max_doc_length == 1000000
    assert config.batch_size == 1000


def test_spacy_entity_extraction_config_custom_models() -> None:
    """Test SpacyEntityExtractionConfig with custom language models."""
    custom_models = {"en": "en_core_web_lg", "fr": "fr_core_news_sm"}
    config = SpacyEntityExtractionConfig(language_models=custom_models)
    assert isinstance(config.language_models, tuple)
    assert dict(config.language_models) == custom_models


def test_spacy_entity_extraction_config_get_model_for_language() -> None:
    """Test get_model_for_language method."""
    config = SpacyEntityExtractionConfig()

    # Test exact match
    assert config.get_model_for_language("en") == "en_core_web_sm"
    assert config.get_model_for_language("de") == "de_core_news_sm"

    # Test base language fallback
    assert config.get_model_for_language("en-US") == "en_core_web_sm"
    assert config.get_model_for_language("de-DE") == "de_core_news_sm"

    # Test non-existent language
    assert config.get_model_for_language("xx") is None
    assert config.get_model_for_language("nonexistent") is None


def test_spacy_entity_extraction_config_get_fallback_model() -> None:
    """Test get_fallback_model method."""
    config_with_fallback = SpacyEntityExtractionConfig(fallback_to_multilingual=True)
    assert config_with_fallback.get_fallback_model() == "xx_ent_wiki_sm"

    config_without_fallback = SpacyEntityExtractionConfig(fallback_to_multilingual=False)
    assert config_without_fallback.get_fallback_model() is None


def test_spacy_entity_extraction_config_empty_models() -> None:
    """Test SpacyEntityExtractionConfig with empty language models."""
    config = SpacyEntityExtractionConfig(language_models={})
    assert config.get_model_for_language("en") is None


def test_spacy_entity_extraction_config_model_cache_dir() -> None:
    """Test SpacyEntityExtractionConfig with model cache directory."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        config = SpacyEntityExtractionConfig(model_cache_dir=temp_dir)
        assert str(config.model_cache_dir) == temp_dir


def test_select_spacy_model_fallback() -> None:
    """Test _select_spacy_model with fallback behavior."""
    config = SpacyEntityExtractionConfig(language_models={"en": "en_core_web_sm"}, fallback_to_multilingual=True)

    # Test normal selection
    model = ee._select_spacy_model(["en"], config)
    assert model == "en_core_web_sm"

    # Test fallback when language not found
    model = ee._select_spacy_model(["nonexistent"], config)
    assert model == "xx_ent_wiki_sm"  # Should return fallback

    # Test no fallback when disabled
    config_no_fallback = SpacyEntityExtractionConfig(
        language_models={"en": "en_core_web_sm"}, fallback_to_multilingual=False
    )
    model = ee._select_spacy_model(["nonexistent"], config_no_fallback)
    assert model is None


def test_extract_entities_empty_input() -> None:
    """Test extract_entities with empty input."""
    result = ee.extract_entities("", entity_types=["PERSON"])
    assert result == []


def test_extract_entities_no_entities_types() -> None:
    """Test extract_entities with no entity types specified."""
    result = ee.extract_entities(SAMPLE_TEXT, entity_types=())
    assert isinstance(result, list)


def test_extract_keywords_with_default_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test extract_keywords with default count."""

    class DummyModel:
        def extract_keywords(self, _text: str, top_n: int = 10) -> list[tuple[str, float]]:
            return [("keyword", 0.5)] * min(top_n, 3)

    mock_keybert = MagicMock()
    mock_keybert.KeyBERT = DummyModel
    monkeypatch.setitem(sys.modules, "keybert", mock_keybert)

    result = ee.extract_keywords(SAMPLE_TEXT)  # Default count is 10
    assert len(result) == 3  # But dummy only returns 3


def test_extract_keywords_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test extract_keywords with empty input."""

    class DummyModel:
        def extract_keywords(self, _text: str, top_n: int = 10) -> list[tuple[str, float]]:
            return []

    mock_keybert = MagicMock()
    mock_keybert.KeyBERT = DummyModel
    monkeypatch.setitem(sys.modules, "keybert", mock_keybert)

    result = ee.extract_keywords("")
    assert result == []
