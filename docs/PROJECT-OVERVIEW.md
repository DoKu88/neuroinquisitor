# neuroinquisitor Project – Phase 1 Overview

**Project Name**: neuroinquisitor  
**Repository**: `neuroinquisitor`  
**Goal**: Build a professional, pip-installable Python package for PyTorch weight observability.

## Objectives
- Build a clean, reusable, and **pip-installable** Python package that reliably snapshots and persistently saves all PyTorch model weights during training.
- Store weights in a local, high-performance HDF5 database (via `h5py`).
- Create a well-documented, type-hinted `NeuroInquisitor` class that can be dropped into any training loop with minimal code changes.
- Follow modern Python packaging best practices so the library can be installed via `pip install neuroinquisitor`.

## Goals
- Make weight saving automatic, safe, efficient, and production-ready.
- Support multiple snapshots over training without data loss or corruption.
- Provide easy read-back of saved weights for later analysis.
- Ensure the package is fully tested and follows PyPI standards.

## Deliverables (End of Phase 1)
- Fully functional, pip-installable neuroinquisitor package
- HDF5-based local database with hierarchical structure
- Working `snapshot()` method
- Context-manager support and safe file handling
- Complete working example notebook
- Comprehensive unit tests
- Proper package structure (`pyproject.toml`)

**Phase 1 Success Criteria**  
You can run `pip install -e .`, train any PyTorch model, call `observer.snapshot()` periodically, and end up with a usable `weights.h5` file containing all weight tensors from multiple epochs. All tests pass.

**Next Phase**  
Gradients, activations, statistics, PCA/UMAP, Captum integration, and full PyPI release.
