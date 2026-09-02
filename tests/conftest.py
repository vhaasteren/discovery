#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""
This is a configuration file for pytest containing customizations and fixtures.

In VSCode, Code Coverage is recorded in config.xml. Delete this file to reset reporting.
"""

from __future__ import annotations

from typing import List

import pytest
from _pytest.nodes import Item


def pytest_collection_modifyitems(items: list[Item]):
    for item in items:
        if "_int_" in item.nodeid:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def _restore_kernel_mode():
    """Snapshot the discovery kernel mode before each test and restore it after.

    Tests that switch modes (`ds.config(kernels=...)`) must not leak the last
    mode into later tests. This autouse net snapshots the module default
    (currently 'matrix'; metamath is opt-in) and restores it after every test,
    so the suite is mode-independent regardless of any per-test reset
    convention. Import is lazy so tests that never touch discovery pay nothing
    at collection time.
    """
    try:
        import discovery as ds
    except Exception:
        yield
        return
    saved = ds.config()
    try:
        yield
    finally:
        ds.config(kernels=saved)


@pytest.fixture
def unit_test_mocks(monkeypatch: None):
    """Include Mocks here to execute all commands offline and fast."""
    pass
