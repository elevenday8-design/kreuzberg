# Metadata Extraction

Kreuzberg automatically extracts rich metadata from documents including titles, authors, dates, and format-specific properties.

## How It Works

The [`ExtractionResult`](../api-reference/types.md#extractionresult) includes a `metadata` dictionary with available metadata properties. Each format uses specialized extractors:

- **PDF**: Uses [Playa](https://github.com/dhdaines/playa) to extract document information, structure, and properties
- **Office**: Uses Pandoc for Word, PowerPoint, and other formats
- **Images**: Extracts dimensions and format properties
- **HTML**: Extracts meta tags and structured data

## Metadata Fields

The [`Metadata`](../api-reference/types.md#metadata) dictionary is a `TypedDict` with optional fields. Only available fields are included in the results.

## Usage Example

Accessing document metadata:

```python
from kreuzberg import extract_file

# Extract from a file
result = await extract_file("document.pdf")

# Access metadata - use .get() with a default value for safety
title = result.metadata.get("title", "Untitled Document")
authors = result.metadata.get("authors", ["Unknown Author"])
created_date = result.metadata.get("created_at")

# Print available metadata
print("Available metadata fields:")
for key, value in result.metadata.items():
    print(f"{key}: {value}")
```

## PDF-Specific Metadata

For PDF documents, Kreuzberg extracts a rich set of metadata including:

- **Document information dictionary** values (title, author, subject, keywords, etc.)
- **Document structure** information including page count, dimensions, and outline
- **Font information** from document pages
- **PDF/A compliance** status
- **Encryption** status and permissions
- **Language information** when available in document structure

If a PDF document contains UTF-16BE encoded strings (often present in PDF metadata with a byte order mark `\xfe\xff`), Kreuzberg will automatically detect and decode these properly.

## Working with Multiple Document Types

When working with multiple document types, it's important to remember that different document formats may provide different metadata fields. Always use defensive programming (like using `.get()` with a default value) when accessing metadata fields:

```python
# Safe way to access metadata across different document types
author = result.metadata.get("authors", ["Unknown"])[0] if "authors" in result.metadata else "Unknown"
creation_date = result.metadata.get("created_at", "Unknown date")
```

## Viewing Available Metadata

To view all available metadata for a document:

```python
from kreuzberg import extract_file
import json

result = await extract_file("your_file.pdf")
print(json.dumps(result.metadata, indent=2, default=str))
```

This will print all available metadata fields for the document in a readable format.
