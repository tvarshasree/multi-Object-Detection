"""
Pytest configuration hook.
Guarantees the root project directory is always on Python's sys.path,
allowing test modules to cleanly import 'app' and 'scripts'.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
