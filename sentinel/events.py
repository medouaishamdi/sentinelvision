"""SQLite event log: every run and every alert is stored and queryable."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    scene TEXT,
    source TEXT,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    summary TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    created_at TEXT NOT NULL,
    type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    frame INTEGER,
    time_s REAL,
    track_id INTEGER,
    object_class TEXT,
    zone TEXT,
    snapshot TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventStore:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def new_run(self, scene: str, source: str, model: str) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO runs (created_at, scene, source, model) VALUES (?, ?, ?, ?)",
                            (_now(), scene, source, model))
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, summary: dict, status: str = "done") -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET status = ?, summary = ? WHERE id = ?",
                      (status, json.dumps(summary), run_id))

    def add_events(self, run_id: int, events: list) -> None:
        if not events:
            return
        with self._conn() as c:
            c.executemany(
                "INSERT INTO events (run_id, created_at, type, severity, message, frame, time_s, track_id, "
                "object_class, zone, snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(run_id, _now(), e.type, e.severity, e.message, e.frame, e.time_s, e.track_id,
                  e.object_class, e.zone, e.snapshot) for e in events],
            )

    def runs(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "summary": json.loads(r["summary"]) if r["summary"] else None} for r in rows]

    def run(self, run_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return {**dict(r), "summary": json.loads(r["summary"]) if r["summary"] else None} if r else None

    def events(self, run_id: int | None = None, event_type: str | None = None,
               severity: str | None = None, limit: int = 500) -> list[dict]:
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if run_id is not None:
            sql += " AND run_id = ?"
            args.append(run_id)
        if event_type:
            sql += " AND type = ?"
            args.append(event_type)
        if severity:
            sql += " AND severity = ?"
            args.append(severity)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]
