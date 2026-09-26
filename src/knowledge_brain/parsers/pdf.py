from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from knowledge_brain.parsers.base import ParseError


def parse_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - one bad page shouldn't kill the file
                continue
        return "\n\n".join(pages)
    except PdfReadError as exc:
        raise ParseError(f"corrupted PDF: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"failed to parse PDF: {exc}") from exc
