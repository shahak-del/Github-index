from knowledge_brain.parsers.base import ParseError


def parse_plain_text(content: bytes) -> str:
    """Decode plain-text-like content (txt/md/csv/json/yaml/source code)."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    try:
        return content.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"failed to decode text content: {exc}") from exc
