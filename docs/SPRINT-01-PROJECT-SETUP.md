# Sprint 1: Project Setup & Package Skeleton

**Duration**: 1 day  
**Goal**: Create a clean, modern, pip-installable Python package structure.

## Tasks
- Add proper package metadata (name, version, description, author, license, classifiers)
- Define core dependencies (`torch`, `h5py>=3.9`, `numpy`) and optional dev dependencies
- Create package folder: `neuroinquisitor/` (under `src/` per setuptools layout)
- Add `neuroinquisitor/__init__.py` with `__version__` and main class export
- Add minimal `neuroinquisitor/core.py` (placeholder class only)
- Add `README.md` with installation instructions (`pip install -e .`)
- Set up folder structure: `examples/`, `tests/`, `docs/`
- Add GitHub Actions CI workflow (`.github/workflows/ci.yml`) that runs `pip install -e ".[dev]"` and `pytest` on push and pull request

## Testing
- Write a simple smoke test in `tests/test_import.py` that imports the package and checks `__version__`
- Run `pytest` to verify the test passes

## Definition of Done
- `pip install -e .` succeeds and installs the package correctly
- `import neuroinquisitor; print(neuroinquisitor.__version__)` works
- Repository has clean structure, proper `pyproject.toml`, and professional README
- CI workflow passes on push
- All tests pass
