from __future__ import annotations

from datetime import datetime, timezone

import pytest

from knowledge_brain.dropbox_client import ChangeSet
from knowledge_brain.embeddings import EmbeddingClient
from knowledge_brain.models import DropboxEntry


def make_entry(
    path: str = "/Projects/Alpha/notes.txt",
    file_id: str = "id:1",
    rev: str = "rev1",
    content_hash: str = "hash1",
    size: int = 100,
) -> DropboxEntry:
    return DropboxEntry(
        path_lower=path.lower(),
        path_display=path,
        name=path.rsplit("/", 1)[-1],
        file_id=file_id,
        rev=rev,
        content_hash=content_hash,
        size=size,
        server_modified=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class FakeEmbeddingClient(EmbeddingClient):
    """Deterministic fixed-dimension embeddings, no model/network needed."""

    def __init__(self, dimension: int = 8):
        self.model_name = "fake-embedder"
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            h = sum(ord(c) for c in text) or 1
            vectors.append([((h * (i + 1)) % 97) / 97 for i in range(self._dimension)])
        return vectors


class FakeQdrantStore:
    """In-memory stand-in for QdrantStore, same public surface used by pipeline."""

    def __init__(self):
        self.collections: dict[str, dict] = {}
        self.points: dict[str, dict] = {}

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def ensure_collection(self, name: str, dimension: int) -> None:
        if name in self.collections:
            if self.collections[name]["dimension"] != dimension:
                raise ValueError("dimension mismatch")
            return
        self.collections[name] = {"dimension": dimension}
        self.points[name] = {}

    def count(self, name: str) -> int:
        return len(self.points.get(name, {}))

    def upsert_batched(self, name: str, points, batch_size: int) -> None:
        for p in points:
            self.points.setdefault(name, {})[p.id] = p

    def delete_by_file_id(self, name: str, file_id: str) -> None:
        store = self.points.get(name, {})
        for pid in [pid for pid, p in store.items() if p.payload.get("file_id") == file_id]:
            del store[pid]

    def delete_stale_chunks(self, name: str, file_id: str, keep_below_index: int) -> None:
        store = self.points.get(name, {})
        for pid in [
            pid
            for pid, p in store.items()
            if p.payload.get("file_id") == file_id and p.payload.get("chunk_index", 0) >= keep_below_index
        ]:
            del store[pid]

    def search(self, name: str, vector, limit: int = 5):
        return []


class FakeDropboxClient:
    """Stand-in for DropboxClient. Never touches the network."""

    def __init__(self, entries: list[DropboxEntry], contents: dict[str, bytes]):
        self._entries = entries
        self._contents = contents
        self.latest_cursor = "cursor-1"
        self.next_change_set: ChangeSet | None = None

    def iter_all_entries(self):
        yield from self._entries

    def get_latest_cursor(self) -> str:
        return self.latest_cursor

    def get_changes(self, cursor: str) -> ChangeSet:
        assert self.next_change_set is not None, "test must set next_change_set"
        return self.next_change_set

    def download_bytes(self, path_lower: str) -> bytes:
        return self._contents[path_lower]


@pytest.fixture
def fake_embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def fake_qdrant() -> FakeQdrantStore:
    return FakeQdrantStore()
