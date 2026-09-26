"""Extension -> parser dispatch.

Unsupported/binary extensions are not an error: :func:`extract_text` returns
``(None, "unsupported extension")`` so callers can record the file as
skipped and continue.
"""

from __future__ import annotations

from knowledge_brain.parsers.base import ParseError
from knowledge_brain.parsers.docx import parse_docx
from knowledge_brain.parsers.pdf import parse_pdf
from knowledge_brain.parsers.pptx import parse_pptx
from knowledge_brain.parsers.text import parse_plain_text
from knowledge_brain.parsers.xlsx import parse_xlsx

# Plain-text-like: decoded as-is, no structural parsing needed.
_PLAIN_TEXT_EXTENSIONS = {
    "txt", "md", "markdown", "csv", "tsv", "json", "yaml", "yml",
    # common source/config extensions
    "py", "js", "jsx", "ts", "tsx", "java", "c", "h", "cpp", "hpp", "cs",
    "go", "rb", "php", "sh", "bash", "zsh", "rs", "swift", "kt", "scala",
    "toml", "ini", "cfg", "conf", "xml", "html", "htm", "css", "scss",
    "sql", "graphql", "proto", "gradle", "dockerfile", "env",
    "log", "rst", "tex",
}

_STRUCTURED_PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "xlsx": parse_xlsx,
    "xlsm": parse_xlsx,
    "pptx": parse_pptx,
}

SUPPORTED_EXTENSIONS = _PLAIN_TEXT_EXTENSIONS | set(_STRUCTURED_PARSERS)


def content_type_for(extension: str) -> str:
    ext = extension.lower()
    if ext in _STRUCTURED_PARSERS:
        return ext
    if ext in _PLAIN_TEXT_EXTENSIONS:
        return "text"
    return "binary"


def extract_text(extension: str, content: bytes) -> tuple[str | None, str | None]:
    """Return ``(text, error)``. Exactly one of the two is not ``None``.

    ``error`` is ``"unsupported extension"`` for binary/unknown files (a
    normal skip, not a failure), or a human-readable message when a
    supported file failed to parse (recorded as failed, not fatal).
    """
    ext = extension.lower()
    parser = _STRUCTURED_PARSERS.get(ext)
    if parser is not None:
        try:
            return parser(content), None
        except ParseError as exc:
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001 - never let a bad file kill the run
            return None, f"unexpected parser error: {exc}"

    if ext in _PLAIN_TEXT_EXTENSIONS:
        try:
            return parse_plain_text(content), None
        except ParseError as exc:
            return None, str(exc)

    return None, "unsupported extension"
