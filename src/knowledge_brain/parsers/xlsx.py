from io import BytesIO

import openpyxl

from knowledge_brain.parsers.base import ParseError


def parse_xlsx(content: bytes) -> str:
    try:
        workbook = openpyxl.load_workbook(BytesIO(content), data_only=True, read_only=True)
        parts = []
        for sheet in workbook.worksheets:
            parts.append(f"# Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"failed to parse XLSX: {exc}") from exc
