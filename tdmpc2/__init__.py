"""Top-level package for TDMPC2 utilities and CLI entrypoints.

This project stores its source code under ``tdmpc2/tdmpc2`` while expecting
imports such as ``tdmpc2.common`` to work from the repository root. We extend
``__path__`` here so subpackages in the nested directory are discoverable
without requiring extra ``PYTHONPATH`` tweaks.
"""

from __future__ import annotations

import os
import pkgutil

# Allow Python to discover modules in the nested ``tdmpc2/tdmpc2`` directory.
_INNER_PKG = os.path.join(os.path.dirname(__file__), "tdmpc2")
if os.path.isdir(_INNER_PKG):
    # ``pkgutil.extend_path`` keeps the default path while avoiding duplicates.
    __path__ = pkgutil.extend_path(__path__, __name__)
    if _INNER_PKG not in __path__:
        __path__.append(_INNER_PKG)

# Submodules such as ``tdmpc2.launch`` and ``tdmpc2.common`` can now be imported
# directly without additional path tweaks.
