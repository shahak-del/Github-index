from pathlib import Path

from knowledge_brain.checkpoint import CheckpointStore, FileState


def test_file_state_round_trip(tmp_path: Path):
    store = CheckpointStore(tmp_path / "cp.sqlite3")
    assert store.get_file_state("id:1") is None

    state = FileState(
        file_id="id:1",
        path_display="/a/b.txt",
        rev="rev1",
        content_hash="hash1",
        chunk_count=3,
        embedding_model="local:x",
        last_indexed_at="2024-01-01T00:00:00+00:00",
    )
    store.upsert_file_state(state)

    fetched = store.get_file_state("id:1")
    assert fetched is not None
    assert fetched.rev == "rev1"
    assert fetched.chunk_count == 3

    # Update in place.
    state.rev = "rev2"
    state.chunk_count = 5
    store.upsert_file_state(state)
    fetched = store.get_file_state("id:1")
    assert fetched.rev == "rev2"
    assert fetched.chunk_count == 5
    assert store.file_count() == 1
    store.close()


def test_remove_file_state(tmp_path: Path):
    store = CheckpointStore(tmp_path / "cp.sqlite3")
    store.upsert_file_state(
        FileState("id:1", "/a.txt", "r1", "h1", 1, "local:x", "2024-01-01T00:00:00+00:00")
    )
    assert store.file_count() == 1
    store.remove_file_state("id:1")
    assert store.file_count() == 0
    assert store.get_file_state("id:1") is None
    store.close()


def test_cursor_get_set(tmp_path: Path):
    store = CheckpointStore(tmp_path / "cp.sqlite3")
    assert store.get_cursor() is None
    store.set_cursor("cursor-a")
    assert store.get_cursor() == "cursor-a"
    store.set_cursor("cursor-b")
    assert store.get_cursor() == "cursor-b"
    store.close()


def test_find_file_id_by_path(tmp_path: Path):
    store = CheckpointStore(tmp_path / "cp.sqlite3")
    store.upsert_file_state(
        FileState("id:1", "/a/b.txt", "r1", "h1", 1, "local:x", "2024-01-01T00:00:00+00:00")
    )
    assert store.find_file_id_by_path("/a/b.txt") == "id:1"
    assert store.find_file_id_by_path("/missing.txt") is None
    store.close()


def test_persists_across_reopen(tmp_path: Path):
    path = tmp_path / "cp.sqlite3"
    store = CheckpointStore(path)
    store.upsert_file_state(
        FileState("id:1", "/a.txt", "r1", "h1", 2, "local:x", "2024-01-01T00:00:00+00:00")
    )
    store.set_cursor("cursor-a")
    store.close()

    reopened = CheckpointStore(path)
    assert reopened.file_count() == 1
    assert reopened.get_cursor() == "cursor-a"
    reopened.close()
