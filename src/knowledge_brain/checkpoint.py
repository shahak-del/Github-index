"""SQLite-backed checkpoint/state store.

Tracks, per Dropbox file, the last-indexed revision/content-hash and chunk
count (for idempotent reruns and dedup), plus a single persisted Dropbox
delta cursor for incremental sync.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_id TEXT PRIMARY KEY,
    path_display TEXT NOT NULL,
    rev TEXT NOT NULL,
    content_hash TEXT,
    chunk_count INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    last_indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class FileState:
    file_id: str
    path_display: str
    rev: str
    content_hash: str | None
    chunk_count: int
    embedding_model: str
    last_indexed_at: str


class CheckpointStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- Files -----------------------------------------------------------
    def get_file_state(self, file_id: str) -> FileState | None:
        row = self._conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        return FileState(**dict(row)) if row else None

    def upsert_file_state(self, state: FileState) -> None:
        self._conn.execute(
            """
            INSERT INTO files (file_id, path_display, rev, content_hash, chunk_count,
                                embedding_model, last_indexed_at)
            VALUES (:file_id, :path_display, :rev, :content_hash, :chunk_count,
                    :embedding_model, :last_indexed_at)
            ON CONFLICT(file_id) DO UPDATE SET
                path_display=excluded.path_display,
                rev=excluded.rev,
                content_hash=excluded.content_hash,
                chunk_count=excluded.chunk_count,
                embedding_model=excluded.embedding_model,
                last_indexed_at=excluded.last_indexed_at
            """,
            state.__dict__,
        )
        self._conn.commit()

    def remove_file_state(self, file_id: str) -> None:
        self._conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        self._conn.commit()

    def find_file_id_by_path(self, path_display: str) -> str | None:
        row = self._conn.execute(
            "SELECT file_id FROM files WHERE path_display = ?", (path_display,)
        ).fetchone()
        return row["file_id"] if row else None

    def all_files(self) -> list[FileState]:
        rows = self._conn.execute("SELECT * FROM files").fetchall()
        return [FileState(**dict(r)) for r in rows]

    def file_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS c FROM files").fetchone()["c"]

    def total_chunk_count(self) -> int:
        row = self._conn.execute("SELECT SUM(chunk_count) AS c FROM files").fetchone()
        return row["c"] or 0

    # -- Sync cursor -------------------------------------------------------
    def get_cursor(self) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM sync_state WHERE key = 'dropbox_cursor'"
        ).fetchone()
        return row["value"] if row else None

    def set_cursor(self, cursor: str) -> None:
        self._conn.execute(
            "INSERT INTO sync_state (key, value) VALUES ('dropbox_cursor', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (cursor,),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM sync_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO sync_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()


@contextmanager
def open_checkpoint(db_path: Path):
    store = CheckpointStore(db_path)
    try:
        yield store
    finally:
        store.close()
