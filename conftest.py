# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Top-level conftest. Adds ``src/`` to sys.path so ``import assay``
resolves to the in-tree package whether or not the package has been
``pip install -e .``'d into the active venv.

This makes ``uv run pytest assay/tests/`` work from either workspace
root or the assay/ directory with no editable-install dance.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
