# Knowledge Brain — Dropbox → Qdrant

## Goal
Build a production-ready, resumable Dropbox indexer for a large archive.

Pipeline:
Dropbox → file discovery/download → text extraction → chunking → multilingual embeddings → Qdrant.

The index must preserve project/folder/file metadata and support incremental synchronization after the initial bulk import.

## Secrets
1. Open `.env`
2. Fill:
   - `DROPBOX_ACCESS_TOKEN`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
3. Do NOT commit `.env`, print secrets to logs, or paste secrets into chat.
4. `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, and if needed `EMBEDDING_API_KEY` must be configured before real indexing.

## Instructions for Codex

Work autonomously in this directory and finish the implementation. Do not ask for secrets; read them from `.env`. Never echo secret values.

### 1. Preflight
- Verify Docker/Desktop is available. If not, implement a normal Python venv path as fallback.
- Validate that required environment variables exist without printing their values.
- Test Dropbox authentication with a harmless account/read request.
- Test Qdrant connectivity.
- Detect the Qdrant cluster capacity. The current cluster may be a small/free cluster; do NOT attempt a full archive import if capacity is insufficient.

### 2. Implement the indexer
Use Python 3.12+.

Create a maintainable project with:
- `src/`
- configuration module
- Dropbox client
- parsers
- chunker
- embedding client
- Qdrant client
- sync/checkpoint state
- CLI
- tests
- Dockerfile
- docker-compose.yml
- requirements/pyproject
- `.gitignore`

Read-only Dropbox behavior. Never modify or delete Dropbox content.

Support at minimum:
- PDF
- DOCX
- XLSX
- PPTX
- TXT
- MD
- CSV
- JSON
- YAML/YML
- common source-code/config extensions

Unsupported/binary files must be recorded as skipped, not crash the run.

### 3. Metadata
Every indexed chunk must retain enough metadata to trace it back to the source:
- source = dropbox
- Dropbox file id/revision when available
- full path
- parent folder
- inferred project
- filename
- extension/content type
- modified timestamp
- content hash
- chunk number
- embedding model/version

Use stable deterministic point IDs so reruns are idempotent.

### 4. Embeddings
Use a multilingual embedding model appropriate for Hebrew and English.
Do NOT create the final Qdrant collection until the embedding model and vector dimensionality are known.
Batch embedding requests.
Implement retries/backoff and rate-limit handling.
Store embedding model/version in metadata so future re-embedding/migrations are possible.

### 5. Qdrant
Create/validate the collection programmatically.
Use cosine distance unless the chosen embedding model explicitly requires something else.
Batch upserts.
Create useful payload indexes for fields used as filters, such as project, path/source, content type, and timestamps where appropriate.
Do not assume Qdrant stores original files; only searchable chunks/vectors/metadata go there.

### 6. Initial bulk import
The archive may be large:
- enumerate before downloading everything
- produce a dry-run inventory with file counts/types and estimated workload
- use bounded parallelism
- stream/process files instead of keeping the archive in RAM
- checkpoint progress
- resume after interruption without starting over
- deduplicate using file revision/hash
- provide progress counters and error log
- do NOT overload a free/small Qdrant cluster

Add CLI commands similar to:
- `doctor` — validate dependencies/connections
- `inventory` — inspect Dropbox without indexing
- `sync --dry-run`
- `sync`
- `status`
- `search "query"` — end-to-end smoke test

Exact CLI naming may differ if there is a good reason.

### 7. Incremental sync
Use Dropbox cursor/change APIs where appropriate.
Persist the cursor/checkpoint locally.
On later runs process only additions/changes/deletions.
For deletions, remove only corresponding Qdrant points after safely identifying them.
Never delete anything from Dropbox.

### 8. Reliability/security
- `.env` must be gitignored.
- Add `.env.example` containing placeholders only.
- Never log access tokens/API keys.
- Add retries with exponential backoff.
- Handle corrupted files and individual parser failures without terminating the whole job.
- Write structured logs and a final summary.
- Add tests for deterministic IDs, chunking, metadata, checkpoint/resume, and deletion handling.

### 9. Acceptance test
Before declaring completion:
1. `doctor` passes.
2. `inventory` successfully reads Dropbox.
3. Index a SMALL controlled sample only.
4. Verify points exist in Qdrant.
5. Run at least one semantic search and show filename/path/chunk metadata in results.
6. Restart/resume and verify duplicate points are not created.
7. Run dry-run incremental sync.
8. Report:
   - files discovered
   - files processed/skipped/failed
   - chunks/vectors written
   - collection name/vector dimension
   - embedding model
   - elapsed time
   - estimated requirements for indexing the full archive
   - exact command for the full run

IMPORTANT: Do not launch the full Dropbox archive import automatically. Stop after the sample acceptance test and capacity estimate. The full run should require an explicit command from the user.

## First command
Start by inspecting this README and `.env`, then implement and run `doctor` and `inventory`. Continue through the sample acceptance test autonomously, fixing errors as needed.
