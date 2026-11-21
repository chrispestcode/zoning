import sqlite3
import unittest
from unittest.mock import call, patch

from frozendict import frozendict

from models.contracts import DocumentModel, Status
from pipeline.chunker import Chunker


class ChunkerTest(unittest.TestCase):
    
    @patch("pipeline.chunker.clean_device_memory")
    @patch("pipeline.chunker.DocumentDatabase")
    @patch("pipeline.chunker.HuggingFaceBgeEmbeddings")
    @patch("pipeline.chunker.SemanticChunker")
    def setUp(self, mock_sem_chunker, mock_transformer, mock_db, mock_clean_device_memory):
        # Chunker holds the instances these classes return, so configure those
        self.mock_splitter = mock_sem_chunker.return_value
        self.mock_db = mock_db.return_value
        self.chunker = Chunker()
        
        _CHUNK_TEST_DATA = frozendict(
            {
                "happy": [
                    ["mary had a little lamb", "whose fleece was white as snow", "everywhere that mary went",
                          "here sheep were sure to go"],
                    ["the itsby bitsy spider", "went up the water spout", "down came the rain", "and washed the spider out",
                     "out came the sun"]
                ]
            }
        )
        _DOCUMENT_MODEL_TEST_DATA = frozendict(
            {
                "happy": {
                    "DocumentId": 1,
                    "FileName": "example.pdf",
                    "DocumentType": "PDF",
                    "PlainTextContent": "Sample document content...",
                    "CreatedAt": "2024-01-01 12:00:00.000",
                    "UpdatedAt": "2024-01-01 12:00:00.000",
                    "Status": "NOT_STARTED",
                },
                "happy_2": {
                    "DocumentId": 2,
                    "FileName": "example2.pdf",
                    "DocumentType": "PDF",
                    "PlainTextContent": "More document content...",
                    "CreatedAt": "2024-01-01 12:00:00.000",
                    "UpdatedAt": "2024-01-01 12:00:00.000",
                    "Status": "NOT_STARTED",
                },
            }
        )
        self.DOCUMENT_MODEL_TEST_DATA = _DOCUMENT_MODEL_TEST_DATA
        self.CHUNK_TEST_DATA = _CHUNK_TEST_DATA

    def test_get_documents(self):
        self.mock_db.get_documents.return_value = [DocumentModel(**self.DOCUMENT_MODEL_TEST_DATA.get("happy"))]

        self.assertEqual(self.chunker._get_documents(), [DocumentModel(**self.DOCUMENT_MODEL_TEST_DATA.get("happy"))])
        self.mock_db.get_documents.assert_called_once_with(Status.NOT_STARTED)

    @patch.object(Chunker, "_chunk_text")
    @patch.object(Chunker, "_get_documents")

    def test_chunk_documents_from_db(self, mock_get_documents, mock_chunk_text):
        # (case, documents, _chunk_text result, expected chunk count)
        test_cases = [
            [
                "returns chunk count",
                [DocumentModel(**self.DOCUMENT_MODEL_TEST_DATA.get("happy"))],
                ([13,15,18,20,21,22,25,1,12], [1,2]),
                9,
            ],
            [
                "no documents returns zero",
                [],
                ([13],[1]),
                0,
            ],
        ]
        for case, documents, chunk_and_doc_ids, expected_result in test_cases:
            with self.subTest(case=case):
                mock_chunk_text.reset_mock()
                mock_get_documents.return_value = documents
                mock_chunk_text.return_value = chunk_and_doc_ids
                self.assertEqual(self.chunker.chunk_documents_from_db(), expected_result)
                if not documents:
                    mock_chunk_text.assert_not_called()

    def test_chunk_text(self):
        happy = DocumentModel(**self.DOCUMENT_MODEL_TEST_DATA.get("happy"))
        happy_2 = DocumentModel(**self.DOCUMENT_MODEL_TEST_DATA.get("happy_2"))
        empty = DocumentModel(**{**self.DOCUMENT_MODEL_TEST_DATA.get("happy"), "PlainTextContent": ""})
        chunks, chunks_2 = self.CHUNK_TEST_DATA.get("happy")

        # (case, documents, chunks per split_text call, ids per insert_chunks call, expected result)
        test_cases = [
            ["single document", [happy], [chunks], [[10, 11, 12, 13]], ([10, 11, 12, 13], [1])],
            [
                "multiple documents",
                [happy, happy_2],
                [chunks, chunks_2],
                [[10, 11, 12, 13], [14, 15, 16, 17, 18]],
                ([10, 11, 12, 13, 14, 15, 16, 17, 18], [1, 2]),
            ],
            ["empty text is skipped", [empty], [], [], ([], [])],
            ["no documents", [], [], [], ([], [])],
        ]

        for case, documents, split_results, insert_results, expected_result in test_cases:
            with self.subTest(case=case):
                self.mock_splitter.reset_mock(return_value=True, side_effect=True)
                self.mock_db.reset_mock(return_value=True, side_effect=True)

                self.mock_splitter.split_text.side_effect = split_results
                self.mock_db.insert_chunks.side_effect = insert_results

                result = self.chunker._chunk_text(documents)

                self.assertEqual(result, expected_result)
                chunked = [document for document in documents if document.PlainTextContent]
                self.assertEqual(
                    self.mock_splitter.split_text.call_args_list,
                    [call(document.PlainTextContent) for document in chunked],
                )
                self.assertEqual(
                    self.mock_db.insert_chunks.call_args_list,
                    [
                        call([{"DocumentId": document.DocumentId, "ChunkText": chunk} for chunk in document_chunks],
                             Status.CHUNKING_COMPLETED)
                        for document, document_chunks in zip(chunked, split_results)
                    ],
                )
                self.assertEqual(
                    self.mock_db.update_document_status.call_args_list,
                    [call([document.DocumentId], Status.CHUNKING_COMPLETED) for document in chunked],
                )

    def test_chunk_text_writes_inside_one_transaction(self):
        happy = DocumentModel(**self.DOCUMENT_MODEL_TEST_DATA.get("happy"))
        self.mock_splitter.split_text.return_value = ["chunk"]
        self.mock_db.insert_chunks.return_value = [10]

        self.chunker._chunk_text([happy])

        self.assertEqual(
            [name for name, _, _ in self.mock_db.mock_calls],
            ["transaction", "transaction().__enter__", "insert_chunks",
             "update_document_status", "transaction().__exit__"],
        )

    def test_chunk_text_skips_document_when_transaction_fails(self):
        happy = DocumentModel(**self.DOCUMENT_MODEL_TEST_DATA.get("happy"))
        happy_2 = DocumentModel(**self.DOCUMENT_MODEL_TEST_DATA.get("happy_2"))
        chunks, chunks_2 = self.CHUNK_TEST_DATA.get("happy")
        self.mock_splitter.split_text.side_effect = [chunks, chunks_2]

        self.mock_db.insert_chunks.side_effect = [sqlite3.Error("disk I/O error"), [14, 15, 16, 17, 18]]

        result = self.chunker._chunk_text([happy, happy_2])

        self.assertEqual(result, ([14, 15, 16, 17, 18], [2]))
        self.mock_db.update_document_status.assert_called_once_with([2], Status.CHUNKING_COMPLETED)

        exit_args = self.mock_db.transaction.return_value.__exit__.call_args_list[0].args
        self.assertIs(exit_args[0], sqlite3.Error)

if __name__ == '__main__':
    unittest.main()