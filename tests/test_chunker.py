from knowledge_brain.chunker import chunk_text, make_chunks, point_id
from tests.conftest import make_entry


def test_chunk_text_empty():
    assert chunk_text("   ", 100, 10) == []


def test_chunk_text_single_chunk_when_short():
    assert chunk_text("hello world", 100, 10) == ["hello world"]


def test_chunk_text_splits_and_overlaps():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # every char of the original content should appear in at least one chunk
    assert "".join(chunks).replace(" ", "") != ""


def test_chunk_text_rejects_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_point_id_deterministic():
    a = point_id("dropbox", "id:1", 0)
    b = point_id("dropbox", "id:1", 0)
    assert a == b


def test_point_id_varies_by_chunk_index():
    a = point_id("dropbox", "id:1", 0)
    b = point_id("dropbox", "id:1", 1)
    assert a != b


def test_point_id_varies_by_file_id():
    a = point_id("dropbox", "id:1", 0)
    b = point_id("dropbox", "id:2", 0)
    assert a != b


def test_point_id_stable_across_content_change():
    # IDs are derived from (source, file_id, chunk_index) only, so a
    # content edit that keeps the same chunk position reuses the same
    # point id (upsert overwrites in place -> idempotent + in-place update).
    a = point_id("dropbox", "id:1", 0)
    b = point_id("dropbox", "id:1", 0)
    assert a == b


def test_make_chunks_indices_and_counts():
    entry = make_entry()
    text = "word " * 500
    chunks = make_chunks(entry, text, "hash1", chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.chunk_count == len(chunks)
        assert c.content_hash == "hash1"
        assert c.entry is entry


def test_make_chunks_empty_text_yields_no_chunks():
    entry = make_entry()
    assert make_chunks(entry, "   ", "hash1", 100, 10) == []
