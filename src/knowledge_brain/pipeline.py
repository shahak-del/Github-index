"""Orchestration: doctor / inventory / sync / status / search."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

from qdrant_client.http import models as qm

from knowledge_brain.checkpoint import CheckpointStore, FileState
from knowledge_brain.chunker import make_chunks, point_id
from knowledge_brain.config import Settings
from knowledge_brain.dropbox_client import DropboxAuthTestFailed, DropboxClient
from knowledge_brain.embeddings import EmbeddingClient, build_embedding_client
from knowledge_brain.models import Chunk, DropboxEntry, RunSummary
from knowledge_brain.parsers import SUPPORTED_EXTENSIONS, content_type_for, extract_text
from knowledge_brain.qdrant_store import QdrantStore

logger = logging.getLogger("knowledge_brain.pipeline")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    severity: str = "required"  # "required" | "optional"


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks if c.severity == "required")


def _docker_available() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker not found; falling back to Python venv execution"
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5, check=True
        )
        return True, "docker CLI found and daemon reachable"
    except Exception:  # noqa: BLE001
        return False, "docker CLI found but daemon not reachable; falling back to Python venv execution"


def run_doctor(settings: Settings) -> DoctorReport:
    report = DoctorReport()

    docker_ok, docker_detail = _docker_available()
    report.checks.append(CheckResult("docker", docker_ok, docker_detail, severity="optional"))

    for name in ("dropbox_access_token", "qdrant_url", "qdrant_api_key"):
        report.checks.append(CheckResult(f"env:{name.upper()}", True, "present"))

    try:
        dbx = DropboxClient(settings.dropbox_access_token.get_secret_value())
        display_name = dbx.test_auth()
        report.checks.append(
            CheckResult("dropbox_auth", True, f"authenticated as account: {display_name}")
        )
    except DropboxAuthTestFailed as exc:
        report.checks.append(CheckResult("dropbox_auth", False, str(exc)))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(
            CheckResult("dropbox_auth", False, f"could not reach Dropbox: {type(exc).__name__}: {exc}")
        )

    try:
        store = QdrantStore(
            settings.qdrant_url.get_secret_value(), settings.qdrant_api_key.get_secret_value()
        )
        detail = store.test_connection()
        report.checks.append(CheckResult("qdrant_connection", True, detail))

        capacity = store.detect_capacity()
        report.checks.append(
            CheckResult(
                "qdrant_capacity",
                True,
                f"{capacity.source}: {capacity.detail}",
                severity="optional",
            )
        )
    except Exception as exc:  # noqa: BLE001
        report.checks.append(
            CheckResult("qdrant_connection", False, f"could not reach Qdrant: {type(exc).__name__}: {exc}")
        )

    try:
        client = build_embedding_client(
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else None,
        )
        report.checks.append(
            CheckResult(
                "embedding_model",
                True,
                f"{settings.embedding_provider}:{settings.embedding_model} "
                f"loaded, dim={client.dimension}",
                severity="optional",
            )
        )
    except Exception as exc:  # noqa: BLE001
        report.checks.append(
            CheckResult(
                "embedding_model",
                False,
                f"embedding client unavailable ({type(exc).__name__}: {exc}); "
                "required before real indexing",
                severity="optional",
            )
        )

    return report


@dataclass
class InventoryReport:
    total_files: int = 0
    total_bytes: int = 0
    by_extension: Counter = field(default_factory=Counter)
    supported_count: int = 0
    unsupported_count: int = 0
    projects: Counter = field(default_factory=Counter)
    oversized_count: int = 0
    _unsupported_bytes: int = 0

    def estimated_chunks(self, chunk_size: int, avg_bytes_per_char: float = 2.0) -> int:
        # Very rough estimate: assume supported files' bytes map ~1:1 to
        # extractable characters (binary/formatting overhead ignored), then
        # divide by chunk size.
        supported_bytes = self.total_bytes - self._unsupported_bytes
        if chunk_size <= 0:
            return 0
        return max(1, int(supported_bytes / avg_bytes_per_char / chunk_size))


def run_inventory(settings: Settings, dbx: DropboxClient | None = None) -> InventoryReport:
    dbx = dbx or DropboxClient(settings.dropbox_access_token.get_secret_value(), settings.dropbox_root_path)
    report = InventoryReport()
    for entry in dbx.iter_all_entries():
        report.total_files += 1
        report.total_bytes += entry.size
        report.by_extension[entry.extension or "(none)"] += 1
        report.projects[entry.inferred_project] += 1
        if entry.extension in SUPPORTED_EXTENSIONS:
            report.supported_count += 1
        else:
            report.unsupported_count += 1
            report._unsupported_bytes += entry.size
        if entry.size > settings.max_file_size_mb * 1024 * 1024:
            report.oversized_count += 1
    return report


def _entry_needs_indexing(entry: DropboxEntry, checkpoint: CheckpointStore) -> bool:
    state = checkpoint.get_file_state(entry.file_id)
    if state is None:
        return True
    if entry.content_hash and state.content_hash and entry.content_hash != state.content_hash:
        return True
    if state.rev != entry.rev:
        return True
    return False


def _process_one_file(
    entry: DropboxEntry, dbx: DropboxClient, settings: Settings
) -> tuple[DropboxEntry, str | None, str | None]:
    """Download + extract text for one entry. Returns (entry, text, error)."""
    if entry.extension not in SUPPORTED_EXTENSIONS:
        return entry, None, "unsupported extension"
    if entry.size > settings.max_file_size_mb * 1024 * 1024:
        return entry, None, f"file exceeds max_file_size_mb={settings.max_file_size_mb}"
    try:
        content = dbx.download_bytes(entry.path_lower)
    except Exception as exc:  # noqa: BLE001
        return entry, None, f"download failed: {type(exc).__name__}: {exc}"
    text, error = extract_text(entry.extension, content)
    return entry, text, error


def run_sync(
    settings: Settings,
    checkpoint: CheckpointStore,
    dry_run: bool = False,
    limit: int | None = None,
    incremental: bool | None = None,
    dbx: DropboxClient | None = None,
    qdrant: QdrantStore | None = None,
    embedder: EmbeddingClient | None = None,
) -> RunSummary:
    """Run an initial or incremental sync.

    ``incremental`` auto-detects from the presence of a stored cursor when
    left as ``None``. ``limit`` caps the number of *new/changed* files
    processed -- used for the small controlled sample run.
    """
    dbx = dbx or DropboxClient(settings.dropbox_access_token.get_secret_value(), settings.dropbox_root_path)
    summary = RunSummary()

    stored_cursor = checkpoint.get_cursor()
    is_incremental = incremental if incremental is not None else stored_cursor is not None

    if is_incremental and stored_cursor:
        changes = dbx.get_changes(stored_cursor)
        candidate_entries = changes.upserts
        deleted_paths = changes.deleted_paths
        new_cursor = changes.cursor
    else:
        candidate_entries = list(dbx.iter_all_entries())
        deleted_paths = []
        new_cursor = dbx.get_latest_cursor()

    summary.files_discovered = len(candidate_entries)

    needs_indexing: list[DropboxEntry] = []
    for entry in candidate_entries:
        if entry.extension not in SUPPORTED_EXTENSIONS:
            summary.files_skipped += 1
            continue
        if not _entry_needs_indexing(entry, checkpoint):
            summary.files_unchanged += 1
            continue
        needs_indexing.append(entry)

    truncated = limit is not None and limit < len(needs_indexing)
    to_process = needs_indexing[:limit] if limit is not None else needs_indexing
    summary.truncated_by_limit = truncated

    if dry_run:
        summary.finished_at = datetime.now(timezone.utc)
        return summary

    qdrant = qdrant or QdrantStore(
        settings.qdrant_url.get_secret_value(), settings.qdrant_api_key.get_secret_value()
    )

    # Deletions (only relevant for incremental syncs). Always applied in
    # full, regardless of --limit, since removals are cheap/idempotent and
    # unrelated to the file-processing cap.
    for path in deleted_paths:
        file_id = checkpoint.find_file_id_by_path(path)
        if file_id:
            qdrant.delete_by_file_id(settings.qdrant_collection, file_id)
            checkpoint.remove_file_state(file_id)

    # IMPORTANT: never advance the cursor past work this run didn't actually
    # do. If --limit truncated the batch, the cursor stays put so the next
    # `sync` call re-fetches the same candidates (already-indexed ones are
    # then skipped as "unchanged" in O(1) checkpoint lookups) until a run
    # completes untruncated -- otherwise a sample run would silently mark
    # the rest of the archive as "already synced" and an incremental sync
    # would never pick it up.
    if truncated:
        summary.cursor_advanced = False
        if not to_process:
            summary.finished_at = datetime.now(timezone.utc)
            return summary
    elif not to_process:
        checkpoint.set_cursor(new_cursor)
        summary.cursor_advanced = True
        summary.finished_at = datetime.now(timezone.utc)
        return summary

    embedder = embedder or build_embedding_client(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else None,
    )
    qdrant.ensure_collection(settings.qdrant_collection, embedder.dimension)

    with ThreadPoolExecutor(max_workers=settings.download_concurrency) as pool:
        futures = {pool.submit(_process_one_file, e, dbx, settings): e for e in to_process}
        for future in as_completed(futures):
            entry, text, error = future.result()
            if error and text is None:
                if error == "unsupported extension":
                    summary.files_skipped += 1
                else:
                    summary.files_failed += 1
                    summary.errors.append(f"{entry.path_display}: {error}")
                continue

            chunks = make_chunks(
                entry, text or "", entry.content_hash or entry.rev,
                settings.chunk_size_chars, settings.chunk_overlap_chars,
            )
            if not chunks:
                summary.files_skipped += 1
                continue

            _embed_and_upsert(chunks, embedder, qdrant, settings)

            prev_state = checkpoint.get_file_state(entry.file_id)
            if prev_state and prev_state.chunk_count > len(chunks):
                qdrant.delete_stale_chunks(settings.qdrant_collection, entry.file_id, len(chunks))

            checkpoint.upsert_file_state(
                FileState(
                    file_id=entry.file_id,
                    path_display=entry.path_display,
                    rev=entry.rev,
                    content_hash=entry.content_hash,
                    chunk_count=len(chunks),
                    embedding_model=f"{settings.embedding_provider}:{settings.embedding_model}",
                    last_indexed_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            summary.files_processed += 1
            summary.chunks_written += len(chunks)

    if not truncated:
        checkpoint.set_cursor(new_cursor)
        summary.cursor_advanced = True
    summary.finished_at = datetime.now(timezone.utc)
    return summary


def _embed_and_upsert(
    chunks: list[Chunk], embedder: EmbeddingClient, qdrant: QdrantStore, settings: Settings
) -> None:
    texts = [c.text for c in chunks]
    vectors = embedder.embed_batched(texts, settings.embedding_batch_size)
    points = []
    for chunk, vector in zip(chunks, vectors):
        entry = chunk.entry
        payload = {
            "source": settings.source_name,
            "file_id": entry.file_id,
            "rev": entry.rev,
            "path": entry.path_display,
            "parent_folder": entry.parent_folder,
            "project": entry.inferred_project,
            "filename": entry.name,
            "content_type": content_type_for(entry.extension),
            "extension": entry.extension,
            "modified_at": entry.server_modified.isoformat(),
            "content_hash": chunk.content_hash,
            "chunk_index": chunk.chunk_index,
            "chunk_count": chunk.chunk_count,
            "embedding_model": f"{settings.embedding_provider}:{settings.embedding_model}",
            "text": chunk.text,
        }
        pid = point_id(settings.source_name, entry.file_id, chunk.chunk_index)
        points.append(qm.PointStruct(id=pid, vector=vector, payload=payload))
    qdrant.upsert_batched(settings.qdrant_collection, points, settings.qdrant_upsert_batch_size)


@dataclass
class StatusReport:
    files_indexed: int
    chunks_indexed: int
    points_in_qdrant: int | None
    has_cursor: bool
    collection: str


def run_status(settings: Settings, checkpoint: CheckpointStore, qdrant: QdrantStore | None = None) -> StatusReport:
    points = None
    qdrant = qdrant or QdrantStore(
        settings.qdrant_url.get_secret_value(), settings.qdrant_api_key.get_secret_value()
    )
    try:
        if qdrant.collection_exists(settings.qdrant_collection):
            points = qdrant.count(settings.qdrant_collection)
    except Exception:  # noqa: BLE001
        points = None

    return StatusReport(
        files_indexed=checkpoint.file_count(),
        chunks_indexed=checkpoint.total_chunk_count(),
        points_in_qdrant=points,
        has_cursor=checkpoint.get_cursor() is not None,
        collection=settings.qdrant_collection,
    )


def run_search(
    settings: Settings,
    query: str,
    limit: int = 5,
    embedder: EmbeddingClient | None = None,
    qdrant: QdrantStore | None = None,
):
    embedder = embedder or build_embedding_client(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else None,
    )
    qdrant = qdrant or QdrantStore(
        settings.qdrant_url.get_secret_value(), settings.qdrant_api_key.get_secret_value()
    )
    vector = embedder.embed([query])[0]
    return qdrant.search(settings.qdrant_collection, vector, limit=limit)
