# Knowledge Brain — Dropbox → Qdrant

A resumable, read-only Dropbox indexer that extracts text from a large
mixed-format archive, chunks it, embeds it with a multilingual model
(Hebrew + English), and writes searchable vectors + metadata into Qdrant.

See `README_CODEX.md` for the original task specification.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Fill in `.env` (never commit it — see `.env.example` for the placeholder
list). Required: `DROPBOX_ACCESS_TOKEN`, `QDRANT_URL`, `QDRANT_API_KEY`.
`EMBEDDING_PROVIDER`/`EMBEDDING_MODEL` default to a local, no-API-key
multilingual model (`intfloat/multilingual-e5-small`, good Hebrew+English
coverage) — override to `openai`/`cohere` + `EMBEDDING_API_KEY` if you'd
rather use a hosted embeddings API.

## CLI

```bash
knowledge-brain doctor              # validate deps/connections, no writes
knowledge-brain inventory           # enumerate Dropbox, no downloads
knowledge-brain sync --dry-run      # classify what would change, no writes
knowledge-brain sync --limit 25     # index a small controlled sample
knowledge-brain sync                # index everything new/changed
knowledge-brain sync --full         # force a full re-scan (ignore cursor)
knowledge-brain status              # checkpoint + Qdrant point counts
knowledge-brain search "query"      # end-to-end semantic search smoke test
```

Or via Docker:

```bash
docker compose run --rm knowledge-brain doctor
```

## Design notes

- **Read-only Dropbox.** Only `files_list_folder*`, `files_download`,
  `files_list_folder/continue|get_latest_cursor`, and
  `users_get_current_account` are called. Nothing is ever written or
  deleted in Dropbox.
- **Deterministic point IDs.** `uuid5(source, file_id, chunk_index)` —
  independent of content, so reruns overwrite the same points in place
  instead of duplicating them (`src/knowledge_brain/chunker.py`).
- **Idempotent resync.** Each file's Dropbox revision/content-hash is
  checkpointed in `state/checkpoint.sqlite3`; unchanged files are skipped
  on rerun. If a file shrinks, trailing stale chunks from the old, longer
  version are deleted.
- **Incremental sync.** A Dropbox delta cursor is persisted only after a
  run finishes *without* being cut short by `--limit`; subsequent `sync`
  calls then fetch only additions/changes/deletions. This means a `--limit`
  sample run never silently marks the untouched rest of the archive as
  "already synced" — rerunning `sync` (with or without `--limit`) picks up
  exactly where the last run left off, until one pass completes untruncated
  and the cursor advances. Deletions remove only the corresponding Qdrant
  points, never anything in Dropbox.
- **Collection creation is deferred** until the embedding client is
  built and its vector dimension is known (`qdrant_store.ensure_collection`).
- **Secrets** are wrapped in `pydantic.SecretStr` and redacted from all
  log output (`config.py`, `logging_setup.py`); CLI commands fail with a
  clean one-line message instead of a raw traceback on connectivity
  errors, so a proxy/network error never dumps request internals.

## Tests

```bash
pytest
```

Unit tests cover chunking, deterministic point IDs, checkpoint
persistence/resume, idempotent reruns (no duplicate points), incremental
deletion handling, dry-run safety, and traceback metadata completeness —
all against in-memory fakes, no network required.
