"""Text chunking and deterministic point-ID generation.

Point IDs are derived from ``(source, file_id, chunk_index)`` only -- never
from content -- so that re-indexing the *same* chunk position of the *same*
file always produces the same Qdrant point ID (upsert overwrites in place,
making reruns idempotent). Content changes are detected separately via the
Dropbox content hash stored in the checkpoint DB and in each point's
payload.
"""

from __future__ import annotations

import uuid

from knowledge_brain.models import Chunk, DropboxEntry

_POINT_ID_NAMESPACE = uuid.UUID("6a3f0d1a-6b8e-4b9a-9f2f-9b1a9c9d7e10")


def point_id(source_name: str, file_id: str, chunk_index: int) -> str:
    """Deterministic UUID5 string suitable as a Qdrant point ID."""
    key = f"{source_name}:{file_id}:{chunk_index}"
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, key))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows of ~``chunk_size`` characters.

    Breaks on the nearest preceding whitespace when possible to avoid
    cutting mid-word. Empty/whitespace-only input yields no chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)
    step = chunk_size - overlap

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            boundary = text.rfind(" ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start += step
        if start <= 0:
            start = end

    return chunks


def make_chunks(
    entry: DropboxEntry,
    text: str,
    content_hash: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    pieces = chunk_text(text, chunk_size, overlap)
    total = len(pieces)
    return [
        Chunk(
            entry=entry,
            chunk_index=i,
            chunk_count=total,
            text=piece,
            content_hash=content_hash,
        )
        for i, piece in enumerate(pieces)
    ]
