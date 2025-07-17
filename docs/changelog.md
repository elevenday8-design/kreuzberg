# Changelog

All notable changes to Kreuzberg will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.9.0] - 2025-01-17

### Added

- Automatic Document Type Detection (#88) - A new feature for classifying documents into categories (contract, form, invoice, receipt, report)
    - Integration with Google Translate for multi-language support
    - New optional dependency group `auto-classify-document-type` with `deep-translator` and `pandas`
    - Comprehensive tests and documentation
- DeepSource integration for code quality analysis

### Fixed

- PDF extraction handling when no OCR backend is available
- Entity extraction test updated to use frozenset of tuples
- Config handling for dataclasses with `slots=True` - replaced `config.__dict__` with `asdict(config)`
- Coverage configuration and cleanup issues

### Changed

- CI/CD: Added retry logic for flaky steps across all platforms
- Improved coverage gathering and cleanup in test runs
- Updated dependencies in `uv.lock`

## [3.8.2] - Previous Release

### Added

- Documentation site with comprehensive examples and API reference
- Improved configuration for all OCR backends
- Added hooks system for validation and post-processing
- Language detection feature with `auto_detect_language` configuration option
- New optional dependency group `langdetect` for automatic language detection

### Changed

- Refactored internal structure for better maintainability
- Updated extraction functions to use config object instead of kwargs
- Improved error messages and reporting

## Previous Versions

For a complete history of changes, please refer to the [GitHub releases page](https://github.com/strickvl/kreuzberg/releases).
