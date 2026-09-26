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
