from __future__ import annotations

import json
import time
from contextlib import contextmanager

import typer
from rich.console import Console
from rich.table import Table

from knowledge_brain.checkpoint import open_checkpoint
from knowledge_brain.config import ConfigError, load_settings
from knowledge_brain.logging_setup import setup_logging
from knowledge_brain.pipeline import run_doctor, run_inventory, run_search, run_status, run_sync

app = typer.Typer(add_completion=False, help="Dropbox -> Qdrant knowledge indexer")
console = Console()


def _init():
    try:
        settings = load_settings()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1)

    secrets = [
        settings.dropbox_access_token.get_secret_value(),
        settings.qdrant_api_key.get_secret_value(),
    ]
    if settings.embedding_api_key:
        secrets.append(settings.embedding_api_key.get_secret_value())
    setup_logging(settings.log_dir, secrets)
    return settings


@contextmanager
def _fail_gracefully(action: str):
    """Turn connectivity/runtime errors into a clean one-line message +
    non-zero exit instead of a raw traceback, without ever printing the
    exception's __cause__ chain (which can embed request URLs/headers)."""
    try:
        yield
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]{action} failed:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from None


@app.command()
def doctor():
    """Validate dependencies and connections without indexing anything."""
    settings = _init()
    report = run_doctor(settings)

    table = Table(title="doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for check in report.checks:
        status = "[green]OK[/green]" if check.ok else (
            "[yellow]WARN[/yellow]" if check.severity == "optional" else "[red]FAIL[/red]"
        )
        table.add_row(check.name, status, check.detail)
    console.print(table)

    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def inventory():
    """Enumerate Dropbox without downloading/indexing anything."""
    settings = _init()
    console.print("[bold]Scanning Dropbox (metadata only, read-only)...[/bold]")
    started = time.time()
    with _fail_gracefully("inventory"):
        report = run_inventory(settings)
    elapsed = time.time() - started

    console.print(f"Files discovered: [bold]{report.total_files}[/bold]")
    console.print(f"Total size: {report.total_bytes / (1024**3):.3f} GB")
    console.print(f"Supported: {report.supported_count}  Unsupported/skipped: {report.unsupported_count}")
    console.print(f"Oversized (> max_file_size_mb): {report.oversized_count}")
    console.print(f"Elapsed: {elapsed:.1f}s")

    table = Table(title="by extension (top 20)")
    table.add_column("extension")
    table.add_column("count", justify="right")
    for ext, count in report.by_extension.most_common(20):
        table.add_row(ext, str(count))
    console.print(table)

    est_chunks = report.estimated_chunks(settings.chunk_size_chars)
    console.print(f"Estimated chunks for full archive: ~{est_chunks:,}")


@app.command()
def sync(
    dry_run: bool = typer.Option(False, "--dry-run", help="Classify/plan only, no downloads or writes"),
    limit: int = typer.Option(None, "--limit", help="Cap number of new/changed files processed (sample runs)"),
    full: bool = typer.Option(False, "--full", help="Force a full re-scan instead of incremental"),
):
    """Index new/changed Dropbox files into Qdrant (incremental if a cursor is stored)."""
    settings = _init()
    with _fail_gracefully("sync"), open_checkpoint(settings.checkpoint_db_path) as checkpoint:
        incremental = False if full else None
        summary = run_sync(settings, checkpoint, dry_run=dry_run, limit=limit, incremental=incremental)

    console.print_json(json.dumps(summary.as_dict()))


@app.command()
def status():
    """Show current index status."""
    settings = _init()
    with _fail_gracefully("status"), open_checkpoint(settings.checkpoint_db_path) as checkpoint:
        report = run_status(settings, checkpoint)
    table = Table(title="status")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("collection", report.collection)
    table.add_row("files indexed (checkpoint)", str(report.files_indexed))
    table.add_row("chunks indexed (checkpoint)", str(report.chunks_indexed))
    table.add_row("points in Qdrant", str(report.points_in_qdrant))
    table.add_row("incremental cursor stored", str(report.has_cursor))
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query"),
    limit: int = typer.Option(5, "--limit"),
):
    """End-to-end semantic search smoke test."""
    settings = _init()
    with _fail_gracefully("search"):
        results = run_search(settings, query, limit=limit)
    table = Table(title=f"search: {query!r}")
    table.add_column("score")
    table.add_column("filename")
    table.add_column("path")
    table.add_column("chunk")
    for r in results:
        payload = r.payload or {}
        table.add_row(
            f"{r.score:.4f}",
            str(payload.get("filename")),
            str(payload.get("path")),
            f"{payload.get('chunk_index')}/{payload.get('chunk_count')}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
