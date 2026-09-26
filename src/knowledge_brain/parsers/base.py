"""Shared parser exception type."""


class ParseError(RuntimeError):
    """Raised by a parser when a file cannot be read/decoded.

    Callers must catch this (and any other exception) per-file so that one
    corrupted document never aborts the whole run.
    """
