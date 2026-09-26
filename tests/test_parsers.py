from knowledge_brain.parsers import content_type_for, extract_text


def test_unsupported_extension_is_skipped_not_errored():
    text, error = extract_text("exe", b"\x00\x01\x02binarydata")
    assert text is None
    assert error == "unsupported extension"


def test_plain_text_extension_decodes():
    text, error = extract_text("txt", "hello שלום".encode("utf-8"))
    assert error is None
    assert "hello" in text
    assert "שלום" in text  # Hebrew round-trips


def test_content_type_for():
    assert content_type_for("pdf") == "pdf"
    assert content_type_for("py") == "text"
    assert content_type_for("exe") == "binary"


def test_corrupted_pdf_is_recorded_as_failed_not_raised():
    text, error = extract_text("pdf", b"not a real pdf")
    assert text is None
    assert error is not None
