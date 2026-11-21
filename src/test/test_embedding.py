import unittest
from sqlite3 import Error as SQLiteError
from unittest.mock import call, patch

import numpy as np

from models.contracts import ChunkModel, Status
from pipeline.embedding import Embedder, InvalidInputException


def make_chunk(chunk_id, text):
    return ChunkModel(
        ChunkId=chunk_id, DocumentId=1, ChunkText=text, CreatedAt="2024-01-01 12:00:00.000",
        UpdatedAt="2024-01-01 12:00:00.000", Status="CHUNKING_COMPLETED",
    )


def fake_encode(text, **kwargs):
    return np.array([len(text), 0.5], dtype=np.float64)


class EmbedderTest(unittest.TestCase):
    @patch("pipeline.embedding.AutoModel")
    @patch("pipeline.embedding.SentenceTransformer")
    @patch("pipeline.embedding.get_config")
    @patch("pipeline.embedding.get_device", return_value="mps")
    @patch("pipeline.embedding.DocumentDatabase")
    def setUp(self, mock_document_database, mock_get_device, mock_get_config, mock_sentence_transformer,
              mock_auto_model):
        self.mock_db = mock_document_database.return_value
        self.mock_config = mock_get_config.return_value
        self.mock_config.get_config_value.return_value = "test-embedding-model"
        self.mock_sentence_transformer = mock_sentence_transformer
        self.mock_encoder = mock_sentence_transformer.return_value
        self.mock_auto_model = mock_auto_model
        self.embedder = Embedder()

        self.mock_clean_device_memory = self._start_patch("pipeline.embedding.clean_device_memory")
        self._start_patch("pipeline.embedding.tqdm", side_effect=lambda iterable, **kwargs: iterable)

    def _start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_init_loads_configured_model(self):
        self.mock_config.get_config_value.assert_called_once_with("models", "embedding_model")
        self.mock_sentence_transformer.assert_called_once_with("test-embedding-model", device="mps")
        self.mock_auto_model.from_pretrained.assert_called_once_with("test-embedding-model")

    def test_generate_text_embeddings(self):
        encoded = np.array([[0.1, 0.2]])

        # (case, text, expected result)
        test_cases = [
            ["encodes text", "first chunk", encoded, None],
            ["empty input raises error", None, [], InvalidInputException],
            ["raises error on text string validation", ["first chunk", "second chunk"], encoded, InvalidInputException]
        ]

        for case, text, expected_result, exception in test_cases:
            with self.subTest(case=case):
                # reset_mock resets the call count and side effects
                self.mock_encoder.reset_mock(return_value=True, side_effect=True)
                self.mock_clean_device_memory.reset_mock()
                self.mock_encoder.encode.return_value = encoded
                if exception:
                    with self.assertRaises(exception):
                        _ = self.embedder.generate_text_embeddings_bge(text)
                    self.mock_encoder.encode.assert_not_called()
                else:
                    result = self.embedder.generate_text_embeddings_bge(text)

                    np.testing.assert_array_equal(result, expected_result)
                    self.mock_encoder.encode.assert_called_once_with(text, convert_to_numpy=True)
                self.mock_clean_device_memory.assert_called_once()

    def test_generate_text_embeddings_bge_failure_still_cleans_memory(self):
        self.mock_encoder.encode.side_effect = RuntimeError("out of memory")

        with self.assertRaises(RuntimeError):
            self.embedder.generate_text_embeddings_bge("chunk")

        self.mock_clean_device_memory.assert_called_once()

    def test_process_incomplete_chunks(self):
        chunks = [make_chunk(1, "a"), make_chunk(2, "bb"), make_chunk(3, "ccc")]

        def as_bytes(*texts):
            return [fake_encode(text).astype(np.float32).tobytes() for text in texts]

        # (case, chunks, batch size, ids per insert call, expected insert calls, expected result)
        test_cases = [
            [
                "single batch",
                chunks,
                300,
                [[1, 2, 3]],
                [call([1, 2, 3], as_bytes("a", "bb", "ccc"))],
                [1, 2, 3],
            ],
            [
                "splits into batches",
                chunks,
                2,
                [[1, 2], [3]],
                [call([1, 2], as_bytes("a", "bb")), call([3], as_bytes("ccc"))],
                [1, 2, 3],
            ],
            [
                "result only includes ids the database stored",
                chunks,
                300,
                [[1, 3]],
                [call([1, 2, 3], as_bytes("a", "bb", "ccc"))],
                [1, 3],
            ],
            ["no chunks", [], 300, [], [], []],
        ]

        for case, chunk_rows, batch_size, insert_results, expected_insert_calls, expected_result in test_cases:
            with self.subTest(case=case):
                self.mock_db.reset_mock(return_value=True, side_effect=True)
                self.mock_encoder.reset_mock(return_value=True, side_effect=True)
                self.mock_db.get_chunks.return_value = chunk_rows
                self.mock_db.insert_embedding_by_chunk_id.side_effect = insert_results
                self.mock_encoder.encode.side_effect = fake_encode

                result = self.embedder.process_incomplete_chunks(batch_size=batch_size)

                self.assertEqual(result, expected_result)
                self.mock_db.get_chunks.assert_called_once_with(Status.CHUNKING_COMPLETED)
                # Embeddings are stored as float32 bytes, one per chunk, in chunk order
                self.assertEqual(self.mock_db.insert_embedding_by_chunk_id.call_args_list, expected_insert_calls)
                self.assertEqual(
                    self.mock_encoder.encode.call_args_list,
                    [call(row.ChunkText, convert_to_numpy=True) for row in chunk_rows],
                )

    def test_process_incomplete_chunks_database_error_propagates(self):
        self.mock_db.get_chunks.return_value = [make_chunk(1, "a")]
        self.mock_encoder.encode.side_effect = fake_encode
        self.mock_db.insert_embedding_by_chunk_id.side_effect = SQLiteError("database is locked")

        with self.assertRaises(SQLiteError):
            self.embedder.process_incomplete_chunks()


if __name__ == '__main__':
    unittest.main()
