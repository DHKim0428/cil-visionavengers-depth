#!/usr/bin/env python3
"""Compatibility wrapper for the old DA2-only training command.

New runs should use ``python scripts/train.py --config ...``.  This wrapper is
kept during the transition so older notes and queued commands still resolve to
the unified runner instead of maintaining a second training implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train import main


if __name__ == "__main__":
    main()
