"""Read-only Dropbox client wrapper.

This module never calls any Dropbox write/delete endpoint. Only
``files_list_folder*``, ``files_download``, ``files_list_folder/continue``,
``files_list_folder/get_latest_cursor`` and ``users_get_current_account``
are used.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO

import dropbox
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import DeletedMetadata, FileMetadata, FolderMetadata
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from knowledge_brain.models import DropboxEntry

logger = logging.getLogger("knowledge_brain.dropbox")


class DropboxAuthTestFailed(RuntimeError):
    pass


@dataclass
class ChangeSet:
    upserts: list[DropboxEntry]
    deleted_paths: list[str]
    cursor: str
    has_more: bool


class DropboxClient:
    def __init__(self, access_token: str, root_path: str = ""):
        self._dbx = dropbox.Dropbox(access_token)
        self.root_path = root_path.rstrip("/")

    # -- Preflight -----------------------------------------------------
    def test_auth(self) -> str:
        """Harmless read-only auth check. Returns the account display name."""
        try:
            account = self._dbx.users_get_current_account()
        except AuthError as exc:
            raise DropboxAuthTestFailed(
                "Dropbox authentication failed (invalid/expired access token)."
            ) from exc
        return account.name.display_name

    # -- Discovery -------------------------------------------------------
    def iter_all_entries(self) -> Iterator[DropboxEntry]:
        """Yield every file (not folder) under root_path, recursively."""
        result = self._list_folder(self.root_path)
        yield from self._entries_from_result(result)
        while result.has_more:
            result = self._list_folder_continue(result.cursor)
            yield from self._entries_from_result(result)

    def get_latest_cursor(self) -> str:
        result = self._dbx.files_list_folder_get_latest_cursor(
            self.root_path or "", recursive=True, include_deleted=True
        )
        return result.cursor

    def get_changes(self, cursor: str) -> ChangeSet:
        """Fetch changes since ``cursor`` (additions/updates/deletions)."""
        result = self._list_folder_continue(cursor)
        upserts = list(self._entries_from_result(result))
        deleted = [
            e.path_display
            for e in result.entries
            if isinstance(e, DeletedMetadata)
        ]
        return ChangeSet(
            upserts=upserts,
            deleted_paths=deleted,
            cursor=result.cursor,
            has_more=result.has_more,
        )

    # -- Download ----------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(ApiError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def download_bytes(self, path_lower: str) -> bytes:
        """Read-only download. Returns file content as bytes."""
        _, response = self._dbx.files_download(path_lower)
        try:
            return response.content
        finally:
            response.close()

    # -- Internal helpers ----------------------------------------------
    @retry(
        retry=retry_if_exception_type(ApiError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _list_folder(self, path: str):
        return self._dbx.files_list_folder(
            path or "", recursive=True, include_deleted=False
        )

    @retry(
        retry=retry_if_exception_type(ApiError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _list_folder_continue(self, cursor: str):
        return self._dbx.files_list_folder_continue(cursor)

    @staticmethod
    def _entries_from_result(result) -> Iterator[DropboxEntry]:
        for entry in result.entries:
            if isinstance(entry, FileMetadata):
                yield DropboxEntry(
                    path_lower=entry.path_lower,
                    path_display=entry.path_display,
                    name=entry.name,
                    file_id=entry.id,
                    rev=entry.rev,
                    content_hash=entry.content_hash,
                    size=entry.size,
                    server_modified=entry.server_modified.replace(tzinfo=timezone.utc),
                )
            elif isinstance(entry, FolderMetadata):
                continue
