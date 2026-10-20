"""Test configuration.

The environment is set before `app.main` is imported anywhere, because settings
are read once at import and cached. In particular the history database is
redirected into a temporary directory so a test run never writes to the real
one.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

_TEST_DB = Path(tempfile.gettempdir()) / "visionai-test-history.db"
_TEST_DB.unlink(missing_ok=True)

os.environ.setdefault("VISIONAI_HISTORY_DB", str(_TEST_DB))
os.environ.setdefault("VISIONAI_PERSIST_UPLOADS", "false")
