# Loader Regression Fixture Notes

Binary fixtures are not stored in the repository. Instead, this directory documents the reference samples that developers can
recreate locally when running regression tests for the lightweight extractors.

## CAD formats

Create the following ASCII files if you need to exercise the CAD extractors:

- `cad/simple_line.dxf`
  ```
  0
  SECTION
  2
  HEADER
  0
  ENDSEC
  0
  SECTION
  2
  TABLES
  0
  ENDSEC
  0
  SECTION
  2
  ENTITIES
  0
  LINE
  8
  Layer1
  10
  0.0
  20
  0.0
  30
  0.0
  11
  100.0
  21
  0.0
  31
  0.0
  0
  ENDSEC
  0
  EOF
  ```

- `cad/simple_mesh.stl`
  ```
  solid simple_mesh
    facet normal 0 0 1
      outer loop
        vertex 0 0 0
        vertex 1 0 0
        vertex 0 1 0
      endloop
    endfacet
  endsolid simple_mesh
  ```

## Specialized archives

The comic book extractor expects a CBZ archive that contains a few placeholder pages. Generate `archives/sample.cbz`
locally with:

```python
from pathlib import Path
from zipfile import ZipFile

root = Path('test/test_resources/archives')
root.mkdir(parents=True, exist_ok=True)
archive = root / 'sample.cbz'

with ZipFile(archive, 'w') as zf:
    for idx in range(1, 4):
        zf.writestr(f'page_{idx:02d}.txt', f'Placeholder page {idx}\n')
    zf.writestr('README.txt', 'Synthetic comic archive for regression tests.\n')
```
