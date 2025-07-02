from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from kreuzberg._types import Entity

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None

try:
    from keybert import KeyBERT
except ImportError:
    KeyBERT = None

logger = logging.getLogger(__name__)


def extract_entities(
    text: str,
    entity_types: Sequence[str] = ("PERSON", "ORGANIZATION", "LOCATION", "DATE", "EMAIL", "PHONE"),
    custom_patterns: frozenset[tuple[str, str]] | None = None,
    model: Any = None,
) -> list[Entity]:
    """Extract entities from text using custom regex patterns and/or a NER model.

    Args:
        text: The input text to extract entities from.
        entity_types: List of entity types to extract using the NER model.
        custom_patterns: Tuple mapping entity types to regex patterns for custom extraction.
        model: Pre-initialized NER model instance. If None, a default model is used if available.

    Returns:
        list[Entity]: A list of extracted Entity objects with type, text, start, and end positions.
    """
    entities: list[Entity] = []
    if custom_patterns:
        custom_patterns_dict = dict(custom_patterns)
        for ent_type, pattern in custom_patterns_dict.items():
            entities.extend(
                Entity(type=ent_type, text=match.group(), start=match.start(), end=match.end())
                for match in re.finditer(pattern, text)
            )
    if GLiNER is not None and entity_types:
        ner_model = model or GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        try:
            results = ner_model.predict_entities(text, list(entity_types))
            entities.extend(
                Entity(
                    type=ent["label"],
                    text=ent["text"],
                    start=ent["start"],
                    end=ent["end"],
                )
                for ent in results
            )
        except RuntimeError as e:
            logger.warning("NER model failed: %s. Falling back to regex extraction only.", e)
    return entities


def extract_keywords(
    text: str,
    keyword_count: int = 10,
    model: Any = None,
) -> list[tuple[str, float]]:
    """Extract keywords from text using the KeyBERT model.

    Args:
        text: The input text to extract keywords from.
        keyword_count: Number of top keywords to return. Defaults to 10.
        model: Pre-initialized KeyBERT model instance. If None, a default model is used if available.

    Returns:
        list[tuple[str, float]]: A list of tuples containing keywords and their relevance scores.
    """
    if KeyBERT is None:
        return []
    kw_model = model or KeyBERT()
    try:
        keywords = kw_model.extract_keywords(text, top_n=keyword_count)
        return [(kw, float(score)) for kw, score in keywords]
    except RuntimeError as e:
        logger.debug("Keyword extraction failed: %s, returning empty list.", e)
        return []
