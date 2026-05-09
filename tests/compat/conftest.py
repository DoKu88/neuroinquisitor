"""Conftest for third-party compatibility tests.

These tests verify that NI output formats are directly consumable by Captum,
TorchLens, and TransformerLens without any type conversion or glue code.

Each test file in this directory requires one optional package.  When a
package is missing, the tests skip with an explicit install instruction rather
than silently.  This conftest prints a summary of missing packages at the
start of each session so skipped tests are always visible.

To run just one library's tests:
    pytest tests/compat/test_captum.py
    pytest tests/compat/test_torchlens.py
    pytest tests/compat/test_transformerlens.py

To run all compat tests:
    pytest tests/compat/
"""

from __future__ import annotations

import importlib

import pytest

_COMPAT_PACKAGES: dict[str, str] = {
    "captum": "pip install captum",
    "torchlens": "pip install torchlens",
    "transformer_lens": "pip install transformer-lens",
    "tensorboard": "pip install tensorboard",
    "fiftyone": "pip install fiftyone",
}


def pytest_sessionstart(session: pytest.Session) -> None:
    missing = [
        (pkg, cmd)
        for pkg, cmd in _COMPAT_PACKAGES.items()
        if importlib.util.find_spec(pkg) is None
    ]
    if missing:
        lines = ["", "=" * 60, "  compat/ — optional packages not installed:", ""]
        for pkg, cmd in missing:
            lines.append(f"  MISSING  {pkg}")
            lines.append(f"           install: {cmd}")
        lines += ["", "  Tests requiring these packages will be SKIPPED.", "=" * 60, ""]
        print("\n".join(lines))
