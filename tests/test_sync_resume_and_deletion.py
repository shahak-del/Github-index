from pathlib import Path

from knowledge_brain.checkpoint import CheckpointStore
from knowledge_brain.config import Settings
from knowledge_brain.dropbox_client import ChangeSet
from knowledge_brain.pipeline import run_sync
from tests.conftest import FakeDropboxClient, FakeEmbeddingClient, FakeQdrantStore, make_entry


def _settings(state_dir: Path) -> Settings:
    # _env_file=None: fully isolate tests from any real .env in the repo
    # root (never read real secrets in a test process).
    return Settings(
        _env_file=None,
        DROPBOX_ACCESS_TOKEN="x",
        QDRANT_URL="http://localhost",
        QDRANT_API_KEY="x",
        STATE_DIR=state_dir,
        LOG_DIR=state_dir / "logs",
        CHUNK_SIZE_CHARS=50,
        CHUNK_OVERLAP_CHARS=10,
    )


def test_rerun_is_idempotent_no_duplicate_points(tmp_path: Path):
    settings = _settings(tmp_path)
    entry1 = make_entry(path="/Projects/Alpha/notes.txt", file_id="id:1", rev="r1", content_hash="h1")
    entry2 = make_entry(path="/Projects/Alpha/second.md", file_id="id:2", rev="r1", content_hash="h1")
    contents = {
        entry1.path_lower: b"hello world " * 30,
        entry2.path_lower: b"second file text " * 30,
    }
    dbx = FakeDropboxClient([entry1, entry2], contents)
    qdrant = FakeQdrantStore()
    embedder = FakeEmbeddingClient()
    checkpoint = CheckpointStore(settings.checkpoint_db_path)

    summary1 = run_sync(settings, checkpoint, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=False)
    assert summary1.files_processed == 2
    assert summary1.chunks_written > 0
    points_after_first = qdrant.count(settings.qdrant_collection)
    assert points_after_first == summary1.chunks_written
    assert checkpoint.get_cursor() == "cursor-1"

    # Simulate a restart: same entries, nothing changed on Dropbox's side.
    summary2 = run_sync(settings, checkpoint, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=False)
    assert summary2.files_processed == 0
    assert summary2.files_unchanged == 2
    assert summary2.chunks_written == 0
    assert qdrant.count(settings.qdrant_collection) == points_after_first

    checkpoint.close()


def test_dry_run_makes_no_writes(tmp_path: Path):
    settings = _settings(tmp_path)
    entry1 = make_entry(path="/a.txt", file_id="id:1")
    dbx = FakeDropboxClient([entry1], {entry1.path_lower: b"hello"})
    qdrant = FakeQdrantStore()
    embedder = FakeEmbeddingClient()
    checkpoint = CheckpointStore(settings.checkpoint_db_path)

    summary = run_sync(
        settings, checkpoint, dry_run=True, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=False
    )
    assert summary.files_discovered == 1
    assert summary.files_processed == 0
    assert checkpoint.file_count() == 0
    assert checkpoint.get_cursor() is None  # dry-run must not persist state
    assert settings.qdrant_collection not in qdrant.collections

    checkpoint.close()


def test_limit_truncation_does_not_advance_cursor(tmp_path: Path):
    """A --limit sample run must not mark the whole archive as synced.

    If it advanced the stored cursor to "latest" after only processing a
    handful of files, a later incremental sync would never pick up the
    files that were skipped by the limit -- they'd look like they existed
    before the cursor and were already handled.
    """
    settings = _settings(tmp_path)
    entries = [
        make_entry(path=f"/Projects/Alpha/f{i}.txt", file_id=f"id:{i}", rev="r1", content_hash="h1")
        for i in range(5)
    ]
    contents = {e.path_lower: f"content for file {i} ".encode() * 10 for i, e in enumerate(entries)}
    dbx = FakeDropboxClient(entries, contents)
    qdrant = FakeQdrantStore()
    embedder = FakeEmbeddingClient()
    checkpoint = CheckpointStore(settings.checkpoint_db_path)

    # Sample run: only 2 of the 5 discovered files get processed.
    summary = run_sync(
        settings, checkpoint, limit=2, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=False
    )
    assert summary.files_processed == 2
    assert summary.truncated_by_limit is True
    assert summary.cursor_advanced is False
    assert checkpoint.get_cursor() is None
    assert checkpoint.file_count() == 2

    # A second sample run continues from where the first left off instead
    # of re-processing the same 2 files or skipping the remaining 3.
    summary2 = run_sync(
        settings, checkpoint, limit=2, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=False
    )
    assert summary2.files_unchanged == 2  # the first 2, already indexed
    assert summary2.files_processed == 2  # 2 more of the remaining 3
    assert summary2.cursor_advanced is False
    assert checkpoint.file_count() == 4

    # A final untruncated run finishes the rest and only now advances the cursor.
    summary3 = run_sync(
        settings, checkpoint, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=False
    )
    assert summary3.files_processed == 1
    assert summary3.truncated_by_limit is False
    assert summary3.cursor_advanced is True
    assert checkpoint.get_cursor() == "cursor-1"
    assert checkpoint.file_count() == 5

    checkpoint.close()


def test_incremental_sync_handles_deletion(tmp_path: Path):
    settings = _settings(tmp_path)
    entry1 = make_entry(path="/Projects/Alpha/keep.txt", file_id="id:1", rev="r1", content_hash="h1")
    entry2 = make_entry(path="/Projects/Alpha/remove.txt", file_id="id:2", rev="r1", content_hash="h1")
    contents = {
        entry1.path_lower: b"keep this content " * 10,
        entry2.path_lower: b"remove this content " * 10,
    }
    dbx = FakeDropboxClient([entry1, entry2], contents)
    qdrant = FakeQdrantStore()
    embedder = FakeEmbeddingClient()
    checkpoint = CheckpointStore(settings.checkpoint_db_path)

    run_sync(settings, checkpoint, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=False)
    assert checkpoint.file_count() == 2
    points_before = qdrant.count(settings.qdrant_collection)
    assert points_before > 0

    # Now simulate an incremental sync where Dropbox reports entry2 deleted.
    dbx.next_change_set = ChangeSet(
        upserts=[], deleted_paths=[entry2.path_display], cursor="cursor-2", has_more=False
    )
    summary = run_sync(settings, checkpoint, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=True)

    assert checkpoint.file_count() == 1
    assert checkpoint.get_file_state("id:2") is None
    assert checkpoint.get_file_state("id:1") is not None
    # Only entry2's points should be gone; entry1's remain.
    remaining_file_ids = {p.payload["file_id"] for p in qdrant.points[settings.qdrant_collection].values()}
    assert remaining_file_ids == {"id:1"}
    assert checkpoint.get_cursor() == "cursor-2"

    checkpoint.close()


def test_incremental_dry_run_does_not_apply_deletion(tmp_path: Path):
    settings = _settings(tmp_path)
    entry1 = make_entry(path="/a.txt", file_id="id:1", rev="r1", content_hash="h1")
    dbx = FakeDropboxClient([entry1], {entry1.path_lower: b"hello world " * 10})
    qdrant = FakeQdrantStore()
    embedder = FakeEmbeddingClient()
    checkpoint = CheckpointStore(settings.checkpoint_db_path)

    run_sync(settings, checkpoint, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=False)
    assert checkpoint.file_count() == 1

    dbx.next_change_set = ChangeSet(
        upserts=[], deleted_paths=[entry1.path_display], cursor="cursor-2", has_more=False
    )
    summary = run_sync(
        settings, checkpoint, dry_run=True, dbx=dbx, qdrant=qdrant, embedder=embedder, incremental=True
    )
    # Dry-run must not delete anything or move the cursor.
    assert checkpoint.file_count() == 1
    assert checkpoint.get_cursor() == "cursor-1"

    checkpoint.close()
