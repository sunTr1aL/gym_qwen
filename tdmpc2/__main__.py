"""
Entrypoint for running tdmpc2 as a module (e.g., `python -m tdmpc2.train`).
This file enables proper package-relative imports when running train.py or train_ddp.py via `python -m`.
"""

import sys
from pathlib import Path

# Ensure relative imports work correctly by keeping the package context
if __name__ == '__main__':
    # When run as `python -m tdmpc2.train`, __main__.py is not invoked directly.
    # This file exists to support `python -m tdmpc2` discovery and to document
    # that the package is runnable.
    pass
