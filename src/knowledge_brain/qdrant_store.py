"""Qdrant collection management, indexing, and search.

Collection creation is deferred until the embedding model's vector
dimensionality is known (see :func:`ensure_collection`), per the
requirement that we must not guess a vector size ahead of time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger("knowledge_brain.qdrant")

PAYLOAD_INDEX_FIELDS: dict[str, qm.PayloadSchemaType] = {
    "project": qm.PayloadSchemaType.KEYWORD,
    "path": qm.PayloadSchemaType.KEYWORD,
    "source": qm.PayloadSchemaType.KEYWORD,
    "content_type": qm.PayloadSchemaType.KEYWORD,
    "file_id": qm.PayloadSchemaType.KEYWORD,
    "modified_at": qm.PayloadSchemaType.DATETIME,
}

# Conservative default used for capacity planning when the cluster does not
# expose exact tier limits through the public API (typical for managed
# free/small Qdrant Cloud clusters). Override via QDRANT_ASSUMED_CAPACITY_MB.
DEFAULT_ASSUMED_CAPACITY_MB = 1024


@dataclass
class CapacityEstimate:
    assumed_disk_mb: int
    source: str  # "reported" | "assumed_default"
    detail: str


class QdrantStore:
    def __init__(self, url: str, api_key: str):
        self._url = url
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)

    # -- Preflight -------------------------------------------------------
    def test_connection(self) -> str:
        collections = self.client.get_collections()
        return f"connected, {len(collections.collections)} collection(s) visible"

    def detect_capacity(self, assumed_mb: int = DEFAULT_ASSUMED_CAPACITY_MB) -> CapacityEstimate:
        """Best-effort cluster capacity detection.

        Qdrant Cloud's public data-plane API does not expose the account's
        tier disk/RAM limit. We try the (best-effort, may be blocked)
        ``/telemetry`` endpoint for real numbers and otherwise fall back to
        a conservative assumed small/free-tier limit so the pipeline never
        assumes unlimited capacity.
        """
        try:
            telemetry = self.client.http.service_api.telemetry()
            detail = str(getattr(telemetry, "result", telemetry))[:500]
            return CapacityEstimate(assumed_disk_mb=assumed_mb, source="reported", detail=detail)
        except Exception as exc:  # noqa: BLE001
            return CapacityEstimate(
                assumed_disk_mb=assumed_mb,
                source="assumed_default",
                detail=(
                    "cluster does not expose telemetry/tier limits via the API "
                    f"({type(exc).__name__}); assuming a small/free-tier cluster "
                    f"({assumed_mb} MB) for safety"
                ),
            )

    # -- Collection lifecycle -------------------------------------------
    def collection_exists(self, name: str) -> bool:
        try:
            self.client.get_collection(name)
            return True
        except (UnexpectedResponse, ValueError):
            return False

    def ensure_collection(self, name: str, dimension: int) -> None:
        if self.collection_exists(name):
            info = self.client.get_collection(name)
            existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_size != dimension:
                raise ValueError(
                    f"Collection {name!r} already exists with vector size "
                    f"{existing_size}, but the configured embedding model "
                    f"produces {dimension}-dim vectors. Use a new collection "
                    "name or re-embed/migrate before switching models."
                )
            return

        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
        )
        self._ensure_payload_indexes(name)

    def _ensure_payload_indexes(self, name: str) -> None:
        for field_name, schema in PAYLOAD_INDEX_FIELDS.items():
            try:
                self.client.create_payload_index(
                    collection_name=name, field_name=field_name, field_schema=schema
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not create payload index for %s: %s", field_name, exc)

    def count(self, name: str) -> int:
        return self.client.count(name, exact=True).count

    # -- Writes ------------------------------------------------------------
    def upsert_batched(self, name: str, points: list[qm.PointStruct], batch_size: int) -> None:
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=name, points=batch, wait=True)

    def delete_by_file_id(self, name: str, file_id: str) -> None:
        self.client.delete(
            collection_name=name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="file_id", match=qm.MatchValue(value=file_id))]
                )
            ),
        )

    def delete_stale_chunks(self, name: str, file_id: str, keep_below_index: int) -> None:
        """Remove leftover chunks with index >= keep_below_index for a file
        whose content shrank on re-index (avoids orphaned trailing chunks)."""
        self.client.delete(
            collection_name=name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="file_id", match=qm.MatchValue(value=file_id)),
                        qm.FieldCondition(
                            key="chunk_index", range=qm.Range(gte=keep_below_index)
                        ),
                    ]
                )
            ),
        )

    # -- Reads ----------------------------------------------------------
    def search(self, name: str, vector: list[float], limit: int = 5):
        return self.client.query_points(collection_name=name, query=vector, limit=limit).points
