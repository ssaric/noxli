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

CREATE TABLE IF NOT EXISTS sleep_sessions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time            REAL    NOT NULL,
    end_time              REAL    DEFAULT NULL,
    session_type          TEXT    DEFAULT 'nap',
    method                TEXT    DEFAULT NULL,
    time_to_sleep_minutes INTEGER DEFAULT NULL,
    notes                 TEXT    DEFAULT '',
    created_at            TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_sleep_start ON sleep_sessions (start_time);
CREATE INDEX IF NOT EXISTS idx_sleep_end   ON sleep_sessions (end_time);
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
