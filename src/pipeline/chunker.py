from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from models.contracts import Status, DocumentModel
from document_database import DocumentDatabase
from utils.device import get_device, clean_device_memory
from loguru import logger
from sqlite3 import Error as SqliteError

IMAGE_RESOLUTION_SCALE = 2.0

class Chunker:
    def __init__(self):

        device = get_device()

        model_name = "BAAI/bge-large-en-v1.5"# Normalize embeddings for cosine similarity
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
  
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        self.chunker = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=70,
            min_chunk_size=100,
        )
        self.db = DocumentDatabase(initialize_tables=False)


    def chunk_documents_from_db(self):  
        documents = self._get_documents()
        if not documents:
            return 0
        chunk_ids, document_ids = self._chunk_text(documents)
        logger.info(f'{len(chunk_ids)} successfully chunked and stored in database')
        logger.debug(f'Succesfully chunked text for Documents: {document_ids}')
        return len(chunk_ids)

    def _get_documents(self):
        documents = self.db.get_documents(Status.NOT_STARTED)
        return documents
    
    def _chunk_text(self, data: list[DocumentModel]) -> tuple[list[int], list[int]]:
        """Chunk text and identify associated figure/table references."""
        chunk_ids, document_ids = [], []
        for row in data:
            text = row.PlainTextContent or ''
            if not text:
                continue
            chunks = self.chunker.split_text(text)

            chunk_data = [{'DocumentId': row.DocumentId, 'ChunkText': chunk} for chunk in chunks]
            try:
                with self.db.transaction():
                    inserted_ids = self.db.insert_chunks(chunk_data, Status.CHUNKING_COMPLETED)
                    self.db.update_document_status([row.DocumentId], Status.CHUNKING_COMPLETED)
            except SqliteError as e:
                logger.error(f'Failed to chunk DocumentId {row.DocumentId}; rolled back: {e}')
                #TODO: Send failures to OPS + dedupe them
                continue
            chunk_ids.extend(inserted_ids)
            logger.info(f'Chunked DocumentId {row.DocumentId} into {len(chunks)} chunks with ChunkIds: {inserted_ids}')
            document_ids.append(row.DocumentId)
        self.clean_cuda()
        return chunk_ids, document_ids

    def clean_cuda(self):
        """Clean CUDA memory"""
        clean_device_memory()
