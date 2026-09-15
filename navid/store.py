from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class QueueFull(Exception):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.db = root / "jobs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS refs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, path TEXT NOT NULL,
                    created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL,
                    created REAL NOT NULL, started REAL, finished REAL,
                    output TEXT, inference_s REAL, error TEXT);
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created);
            """)

        # Additive migration preserves existing tasks when updating the service.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "metrics" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN metrics TEXT")
            db.commit()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    def add_reference(self, ref_id: str, kind: str, path: Path) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO refs VALUES (?, ?, ?, ?)",
                       (ref_id, kind, str(path), time.time()))

    def reference(self, ref_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM refs WHERE id=?", (ref_id,)).fetchone()
            return dict(row) if row else None

    def enqueue(self, payload: dict, limit: int) -> str:
        job_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= limit:
                db.rollback()
                raise QueueFull
            db.execute("INSERT INTO jobs(id,status,payload,created) VALUES (?, 'queued', ?, ?)",
                       (job_id, json.dumps(payload), time.time()))
            db.commit()
        return job_id

    def claim(self) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                db.commit()
                return None
            db.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), row['id']))
            db.commit()
            return {"id": row["id"], **json.loads(row["payload"])}

    def finish(self, job_id: str, *, output: str | None = None,
               inference_s: float | None = None, error: str | None = None, metrics: dict | None = None) -> None:
        with self.connect() as db:
            db.execute("""UPDATE jobs SET status=?,finished=?,output=?,inference_s=?,error=?,metrics=?
                          WHERE id=? AND status='running'""",
                       ("failed" if error else "succeeded", time.time(), output, inference_s, error, json.dumps(metrics) if metrics else None, job_id))

    def get(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        payload = json.loads(item.pop("payload"))
        # Jobs from before profile support were always four-step. Never infer
        # historical NFE from the current process configuration after a restart.
        nfe = (payload.get("execution") or {}).get("nfe", 4)
        item.update(duration=payload["duration"], seed=payload["seed"], task="ref2va", nfe=nfe)
        item["execution"] = payload.get("execution")
        item["metrics"] = json.loads(item["metrics"]) if item.get("metrics") else None
        return item

    def fail_pending(self, reason: str) -> None:
        with self.connect() as db:
            db.execute("""UPDATE jobs SET status='failed',finished=?,error=?
                          WHERE status IN ('queued','running')""", (time.time(), reason))
