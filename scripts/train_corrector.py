#!/usr/bin/env python3
"""Entry point for offline corrector training from the repo root."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "tdmpc2"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_corrector import main


if __name__ == "__main__":
    main()
