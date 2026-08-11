"""Tests for document loader."""

import pytest

from rag_agent.document.loader import SUPPORTED_EXTENSIONS, load_document
from rag_agent.utils.errors import DocumentLoadError, UnsupportedFileTypeError


class TestSupportedExtensions:
    def test_includes_common_formats(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".markdown" in SUPPORTED_EXTENSIONS


class TestLoadDocument:
    def test_loads_txt_file_and_returns_document(self, temp_dir):
        file_path = temp_dir / "test.txt"
        file_path.write_text("Hello world. This is a test document.", encoding="utf-8")

        doc = load_document(file_path)

        assert doc.filename == "test.txt"
        assert doc.file_type == "txt"
        assert str(file_path) in doc.file_path
        assert doc.id == ""  # Assigned by pipeline, not loader

    def test_loads_markdown_file(self, temp_dir):
        file_path = temp_dir / "notes.md"
        file_path.write_text("# Heading\n\nSome content here.", encoding="utf-8")

        doc = load_document(file_path)

        assert doc.filename == "notes.md"
        assert doc.file_type == "md"

    def test_raises_on_unsupported_extension(self, temp_dir):
        file_path = temp_dir / "image.png"
        file_path.write_text("not an image actually", encoding="utf-8")

        with pytest.raises(UnsupportedFileTypeError):
            load_document(file_path)

    def test_raises_on_nonexistent_file(self, temp_dir):
        file_path = temp_dir / "does_not_exist.txt"

        with pytest.raises(DocumentLoadError):
            load_document(file_path)

    def test_raises_with_extension_in_message(self, temp_dir):
        file_path = temp_dir / "data.csv"

        # CSV is not in SUPPORTED_EXTENSIONS
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            load_document(file_path)

        assert "csv" in str(exc_info.value)
        assert "data.csv" in str(exc_info.value)

    def test_unsupported_error_has_file_path_and_extension_attrs(self, temp_dir):
        file_path = temp_dir / "data.xyz"

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            load_document(file_path)

        assert exc_info.value.file_path == str(file_path)
        assert exc_info.value.extension == ".xyz"

    def test_size_bytes_is_populated(self, temp_dir):
        file_path = temp_dir / "sized.txt"
        file_path.write_text("a" * 500, encoding="utf-8")

        doc = load_document(file_path)

        assert doc.size_bytes == 500
