"""SQLite event store for cry detection events."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("/data/db/events.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  REAL    NOT NULL,
    duration   REAL    DEFAULT 0,
    confidence REAL    DEFAULT 0,
    source     TEXT    DEFAULT '',
    created_at TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp);
"""


@contextmanager
def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    try:
        yield conn
    finally:
        conn.close()
