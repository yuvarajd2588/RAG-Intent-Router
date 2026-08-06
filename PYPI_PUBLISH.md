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

## 6. Upload to TestPyPI (recommended first)

```bash
twine upload --repository testpypi dist/*
```

You will be prompted for your TestPyPI credentials.

## 7. Install from TestPyPI to verify

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rag-intent-classifier
```

## 8. Upload to PyPI

```bash
twine upload dist/*
```

## 9. Verify the published package

```bash
pip install --upgrade rag-intent-classifier
```

## Notes

- Keep the package version updated in `pyproject.toml` before each release.
- If you need to republish a version, increase the version number first.
- For a first-time publish, make sure your PyPI/TestPyPI account is configured with valid credentials.
