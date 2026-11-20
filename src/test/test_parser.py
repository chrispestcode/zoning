import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from docling.document_converter import DocumentConverter
from pipeline.parser import PdfParser, InvalidFileException
import os

class PdfParserTest(unittest.TestCase):
    @patch("pipeline.parser.get_config")
    @patch("pipeline.parser.DocumentDatabase")
    @patch("pipeline.parser.DocumentConverter")
    @patch.object(PdfParser, "_initialize_models")
    def setUp(self, mock_initialize_models, mock_converter, mock_document_database, mock_get_config):
        self.mock_document_database = mock_document_database
        self.mock_config = mock_get_config.return_value
        self.mock_converter = mock_converter.return_value
        self.pdf_parser = PdfParser()

    def test_is_valid_file_path_resolves(self):
        self.mock_config.get_config_value.return_value = "test_dir"
        expected_path = Path(os.getcwd(), "test_dir", "test.pdf")
        self.assertEqual(self.pdf_parser._is_file_path_valid("test.pdf"), expected_path)

    def test_is_valid_file_path(self):
        test_cases = [
            ("/var/www/uploads", "images/photo.pdf", Path("/var/www/uploads/images/photo.pdf")),
            ("/var/www/uploads", r"images\example.pdf", Path("/var/www/uploads/images/example.pdf")),
            ("/var/www/uploads", "/etc/passwd.pdf", Path("/var/www/uploads/etc/passwd.pdf")),
            ("/var/www/uploads", "../../etc/passwd.pdf", InvalidFileException),
            ("/var/www/uploads", "images/.env", InvalidFileException),
            ("/var/www/uploads", "file;rm -rf.pdf", InvalidFileException),
            ("/var/www/uploads", "images/photo.png", InvalidFileException),
            ("/var/www/uploads", "images/test.xml.pdf", InvalidFileException),
        ]

        for base_dir, file_path, expected_result in test_cases:
            with (self.subTest(base_dir=base_dir, file_path=file_path, expected_result=expected_result)):
                self.mock_config.get_config_value.return_value = base_dir
                if expected_result is InvalidFileException:
                    with self.assertRaises(expected_result):
                        self.pdf_parser._is_file_path_valid(file_path)
                else:
                    self.assertEqual(self.pdf_parser._is_file_path_valid(file_path), expected_result.resolve())

    @patch.object(PdfParser, "_is_file_path_valid")
    def test_get_document_data(self, mock_is_file_path_valid):
        mock_db = self.mock_document_database.return_value
        test_cases = [
            [
                "test.pdf",
                False,
                "test",
                "dummy data",
                {"filename": "test", "document_type": "PDF", "document": "dummy data"},
            ],
            [
                "test.pdf",
                True,
                "test",
                "dummy data",
                {
                    "filename": "test",
                    "document_type": "PDF",
                    "document": None,
                },
            ],
        ]

        for idx, (file_path, is_processed, expected_stem, expected_document_body, expected_result) in enumerate(test_cases):
            with self.subTest(file_path=file_path,
                              is_processed=is_processed,
                              expected_stem=expected_stem,
                              expected_document_body=expected_document_body,
                              expected_result=expected_result):
                mock_db.is_file_processed.reset_mock()
                mock_db.is_file_processed.return_value = is_processed
                mock_is_file_path_valid.reset_mock()
                mock_is_file_path_valid.return_value = file_path

                converted_doc = self.mock_converter.convert.return_value
                with tempfile.TemporaryDirectory() as tmp_dir:
                    self.mock_config.get_config_value.return_value = tmp_dir
                    new_file_path = Path(tmp_dir) / file_path
                    new_file_path.touch()
                    converted_doc.input.file = Path(new_file_path)
                    converted_doc.document = expected_document_body
                    if expected_result is InvalidFileException:
                        with self.assertRaises(expected_result):
                            self.pdf_parser.get_document_data(file_path=file_path)
                    else:
                        converted_doc.document = expected_document_body
                        result = self.pdf_parser.get_document_data(file_path=file_path)

                        self.assertEqual(
                            result,
                            expected_result
                        )

    @patch.object(PdfParser, "get_document_data")
    def test_process_pdf(self, mock_get_document_data):
        test_cases = [
            [
                "test.pdf",
                7,
                {
                    "filename": "test.pdf",
                    "document_type": "PDF",
                    "document": "dummy data",
                },
                7,
            ],
            ["test.pdf", 7, None, -1],
            ["test.pdf",
                -1,
                {
                    "filename": "test.pdf",
                    "document_type": "PDF",
                    "document": "dummy data",
                },
                 -1
             ],
            ]

        for filename, document_id, test_data, expected_result in test_cases:
            with self.subTest(filename=filename, test_data=test_data, document_id=document_id,
                              expected_result=expected_result):

                mock_db = self.mock_document_database.return_value
                mock_db.insert_document.return_value = document_id
                mock_get_document_data.return_value = test_data

                result = self.pdf_parser.process_pdf(filename)

                self.assertEqual(
                    result,
                    expected_result,
                )

if __name__ == '__main__':
    unittest.main()