from pathlib import Path

from knowledge_brain.checkpoint import CheckpointStore
from knowledge_brain.config import Settings
from knowledge_brain.pipeline import run_sync
from tests.conftest import FakeDropboxClient, FakeEmbeddingClient, FakeQdrantStore, make_entry

REQUIRED_PAYLOAD_FIELDS = {
    "source", "file_id", "rev", "path", "parent_folder", "project", "filename",
    "content_type", "extension", "modified_at", "content_hash", "chunk_index",
    "chunk_count", "embedding_model", "text",
}


def test_indexed_points_carry_full_traceback_metadata(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        DROPBOX_ACCESS_TOKEN="x",
        QDRANT_URL="http://localhost",
        QDRANT_API_KEY="x",
        STATE_DIR=tmp_path,
        LOG_DIR=tmp_path / "logs",
        CHUNK_SIZE_CHARS=50,
        CHUNK_OVERLAP_CHARS=10,
    )
    entry = make_entry(path="/Projects/Alpha/notes.txt", file_id="id:1", rev="r1", content_hash="h1")
    dbx = FakeDropboxClient([entry], {entry.path_lower: b"hello world " * 20})
    qdrant = FakeQdrantStore()
    checkpoint = CheckpointStore(settings.checkpoint_db_path)

    run_sync(settings, checkpoint, dbx=dbx, qdrant=qdrant, embedder=FakeEmbeddingClient(), incremental=False)

    points = list(qdrant.points[settings.qdrant_collection].values())
    assert points, "expected at least one point to be written"
    for point in points:
        assert REQUIRED_PAYLOAD_FIELDS.issubset(point.payload.keys())
        assert point.payload["project"] == "Projects"
        assert point.payload["filename"] == "notes.txt"
        assert point.payload["file_id"] == "id:1"

    checkpoint.close()
