"""Shared data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class DropboxEntry:
    """A file entry discovered in Dropbox (metadata only, read-only)."""

    path_lower: str
    path_display: str
    name: str
    file_id: str
    rev: str
    content_hash: str | None
    size: int
    server_modified: datetime

    @property
    def extension(self) -> str:
        return self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""

    @property
    def parent_folder(self) -> str:
        idx = self.path_display.rstrip("/").rfind("/")
        return self.path_display[:idx] if idx > 0 else "/"

    @property
    def inferred_project(self) -> str:
        """First path segment under the indexed root, used as a coarse project label."""
        parts = [p for p in self.path_display.strip("/").split("/") if p]
        return parts[0] if parts else "root"


@dataclass
class ExtractedDocument:
    entry: DropboxEntry
    text: str
    content_type: str
    parse_error: str | None = None


@dataclass
class Chunk:
    entry: DropboxEntry
    chunk_index: int
    chunk_count: int
    text: str
    content_hash: str  # hash of the *file* content this chunk was derived from


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]
    embedding_model: str


@dataclass
class FileOutcome:
    path_display: str
    status: str  # "indexed" | "skipped" | "failed" | "unchanged"
    chunks_written: int = 0
    reason: str | None = None


@dataclass
class RunSummary:
    files_discovered: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    files_unchanged: int = 0
    chunks_written: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        elapsed = None
        if self.finished_at:
            elapsed = (self.finished_at - self.started_at).total_seconds()
        return {
            "files_discovered": self.files_discovered,
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "files_unchanged": self.files_unchanged,
            "chunks_written": self.chunks_written,
            "elapsed_seconds": elapsed,
            "errors": self.errors[:20],
        }
