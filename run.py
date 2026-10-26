#!/usr/bin/env python
"""Entry point: python run.py --case nesp

Adds src/ to the import path so the package runs straight from a clone,
without an install step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from powersys.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
