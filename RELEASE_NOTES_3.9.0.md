# Release Notes - v3.9.0

## New Features

### Automatic Document Type Detection (#88)

Kreuzberg now includes automatic document classification capabilities, allowing you to identify document types (contracts, forms, invoices, receipts, reports) during extraction.

**Key features:**

- Text-based and vision-based classification modes
- Multi-language support via Google Translate integration
- Configurable confidence thresholds
- New configuration options in `ExtractionConfig`:
  - `auto_detect_document_type`: Enable/disable classification
  - `document_classification_mode`: Choose between "text" or "vision" mode
  - `type_confidence_threshold`: Set minimum confidence level
- New result fields in `ExtractionResult`:
  - `document_type`: The detected document category
  - `type_confidence`: Confidence score of the classification

**Installation:**

```bash
pip install "kreuzberg[auto-classify-document-type]"
```

**Usage:**

```python
from kreuzberg import extract_file, ExtractionConfig

config = ExtractionConfig(auto_detect_document_type=True, document_classification_mode="text")
result = await extract_file("invoice.pdf", config=config)
print(f"Type: {result.document_type}, Confidence: {result.type_confidence}")
```

### DeepSource Integration

Added `.deepsource.toml` configuration for automated code quality analysis.

## Bug Fixes

- Fixed PDF extraction when no OCR backend is available
- Updated entity extraction test to use frozenset of tuples for proper comparison
- Fixed config handling for dataclasses with `slots=True` by replacing `config.__dict__` with `asdict(config)`
- Resolved coverage configuration and cleanup issues in test runs

## Improvements

- Enhanced CI/CD pipeline with retry logic for flaky steps across all platforms
- Improved test coverage gathering and cleanup procedures
- Updated dependencies in `uv.lock`

## Documentation

- Added comprehensive guide for document classification feature
- Updated all relevant documentation sections to include the new feature
- Enhanced API reference with new configuration options

## Dependencies

- New optional dependency group: `auto-classify-document-type`
  - `deep-translator`: For multi-language support
  - `pandas`: For data processing in classification
