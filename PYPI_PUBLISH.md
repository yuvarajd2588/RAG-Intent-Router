# PyPI Publish Instructions

This guide walks through building, verifying, and publishing the package to PyPI.

## 1. Prepare the environment

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel build twine pytest
```

## 2. Install the package locally

```bash
pip install -e .
```

## 3. Run tests

```bash
python -m pytest -q
```

## 4. Build distributions

```bash
python -m build
```

This creates both:
- source distribution: `dist/*.tar.gz`
- wheel: `dist/*.whl`

## 5. Verify the build artifacts

```bash
python -m twine check dist/*
```

## 6. Upload to PyPI (production)

```bash
twine upload dist/*
```

You will be prompted for your PyPI credentials.

If you prefer to provide them explicitly in the command line:

```bash
twine upload --repository pypi dist/*
```

## 7. Verify the published package

```bash
pip install --upgrade rag-intent-classifier
```

## Notes

- Keep the package version updated in `pyproject.toml` before each release.
- If you need to republish a version, increase the version number first.
- For a first-time publish, make sure your PyPI/TestPyPI account is configured with valid credentials.
