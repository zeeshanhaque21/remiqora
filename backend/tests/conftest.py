"""Shared fixtures for the remix backend tests.

REMIQORA_DATA_DIR is pointed at a tmp dir *before* any app module is imported:
config.py reads it at import time and db.py opens the SQLite file lazily from
it, so a test run must not touch the developer's real data/.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Must run before "import app.*": config.py resolves DATA_DIR/LOG_DIR at import.
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="remiqora-tests-")
os.environ["REMIQORA_DATA_DIR"] = _TEST_DATA_DIR
os.environ["REMIQORA_LOG_DIR"] = str(Path(_TEST_DATA_DIR) / "logs")
# Keep telemetry pings off in tests too (config.setdefault would do it, but
# be explicit so a pre-set env var can't leak through).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# backend/ on sys.path so "import app" works regardless of pytest's rootdir.
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

from app import db  # noqa: E402
from app.config import DATA_DIR  # noqa: E402


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(autouse=True)
def clean_db():
    """Empty the tracks/projects tables between tests; the shared SQLite file
    lives for the whole session, so each test must start from a clean slate."""
    conn = db.get_db()
    conn.execute("DELETE FROM tracks")
    conn.execute("DELETE FROM projects")
    conn.commit()
    yield
    conn.execute("DELETE FROM tracks")
    conn.execute("DELETE FROM projects")
    conn.commit()


@pytest.fixture(autouse=True)
def reset_jobs():
    """Drop any in-memory pipeline/stems/midi job left by a previous test."""
    from app import stems
    from app.orchestrator.manager import manager
    from app.remix import pipeline

    pipeline._jobs.clear()
    stems._jobs.clear()
    manager.state.models["yue2"].status = manager.state.models["yue2"].status.__class__.STOPPED
    yield
    pipeline._jobs.clear()
    stems._jobs.clear()
