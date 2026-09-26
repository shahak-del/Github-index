"""Pluggable multilingual embedding clients.

Providers:
  * ``local``  -- sentence-transformers, runs on-box, no API key needed.
                  Default model: intfloat/multilingual-e5-small (Hebrew + English).
  * ``openai`` -- OpenAI-compatible embeddings endpoint. Requires
                  EMBEDDING_API_KEY.
  * ``cohere`` -- Cohere multilingual embeddings. Requires EMBEDDING_API_KEY.

The vector dimensionality is only known once the underlying model is
loaded (or after a first live API call), which is why collection creation
must happen *after* an :class:`EmbeddingClient` is constructed -- see
``qdrant_store.ensure_collection``.
"""

from __future__ import annotations

import abc
import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("knowledge_brain.embeddings")


class EmbeddingClient(abc.ABC):
    model_name: str

    @property
    @abc.abstractmethod
    def dimension(self) -> int: ...

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_batched(self, texts: list[str], batch_size: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors.extend(self.embed(batch))
        return vectors


class LocalEmbeddingClient(EmbeddingClient):
    """sentence-transformers, multilingual, no network/API key required
    (after the model weights are cached locally)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Probe with a single short input to learn the dimensionality.
            self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self.model_name, "input": texts},
            timeout=60,
        )
        if resp.status_code == 429:
            raise RuntimeError("rate limited")
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]


class CohereEmbeddingClient(EmbeddingClient):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.cohere.com/v1"):
        self.model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            f"{self._base_url}/embed",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self.model_name, "texts": texts, "input_type": "search_document"},
            timeout=60,
        )
        if resp.status_code == 429:
            raise RuntimeError("rate limited")
        resp.raise_for_status()
        return resp.json()["embeddings"]


def build_embedding_client(
    provider: str, model_name: str, api_key: str | None
) -> EmbeddingClient:
    provider = provider.lower()
    if provider == "local":
        return LocalEmbeddingClient(model_name)
    if provider == "openai":
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for provider 'openai'")
        return OpenAIEmbeddingClient(model_name, api_key)
    if provider == "cohere":
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for provider 'cohere'")
        return CohereEmbeddingClient(model_name, api_key)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")
