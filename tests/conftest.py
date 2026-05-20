"""
pytest configuration for the CEINN repo.

Adds the repo root to sys.path so tests can `from utils.losses import ...`
without an editable install.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
