from sqlite3 import Error as SQLiteError

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from document_database import DocumentDatabase
from models.contracts import Status
from utils.device import clean_device_memory, get_device
from utils.load_config import get_config

class InvalidInputException(Exception):
    def __init__(self):
        Exception.__init__(self, "Embedding requires valid string input")

class Embedder:
    def __init__(self):
        self.db = DocumentDatabase(initialize_tables=False)
        self.device = get_device()
        self.config = get_config()

        embedding_model = self.config.get_config_value('models','embedding_model')

        self.text_embedder = SentenceTransformer(embedding_model, device=self.device)
        self.model = AutoModel.from_pretrained(embedding_model)

    def _clean_cuda(self):
        """Clean CUDA memory"""
        clean_device_memory()
    
    
    def generate_text_embeddings_bge(self, text: str) -> list[np.ndarray]:
        """Generate embeddings for text chunks using BGE Large"""
        try:
            if not text or not isinstance(text, str):
                logger.debug("No chunks provided for embedding generation")
                
                raise InvalidInputException()
            embeddings = self.text_embedder.encode(text, convert_to_numpy=True)
            return embeddings
        except SQLiteError as e:
            logger.error(f"Error generating text embeddings: {e}")
            raise
        finally:
            self._clean_cuda()
    
    def process_incomplete_chunks(self, batch_size  = 300):
        chunk_ids = []
        chunks  = self.db.get_chunks(Status.CHUNKING_COMPLETED)
        for i in tqdm(range(0, len(chunks), batch_size), desc="Processing Chunks..."):
            cids = []
            embeddings = []
            try:
                batch = chunks[i:i + batch_size]
                for row in batch:
                    cids.append(row.ChunkId)
                    embeddings.append(self.generate_text_embeddings_bge(row.ChunkText))

                if len(embeddings) != len(cids):
                    logger.error(f" count mismatch for batch starting at {i}: {len(embeddings)} != {len(cids)}")
                    #TODO: send to ops
                    continue
                upserted_ids = self.db.insert_embedding_by_chunk_id(cids, [embd.astype(np.float32).tobytes() for embd in embeddings])
                chunk_ids.extend(upserted_ids)
            except SQLiteError as e:
                logger.error(f"Error generating text embeddings: {e}")
                raise
        logger.debug(f"Stored embeddings for {len(chunk_ids)} chunk IDs.")
        return chunk_ids
        

