import gc
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoTokenizer, AutoModel
import torch
import re
from typing import List, Dict, Any, Tuple

from contracts import Status, DocumentModel
from document_database import DocumentDatabase
from utils.device import get_device, clean_device_memory
from loguru import logger

IMAGE_RESOLUTION_SCALE = 2.0

class Chunker:
    def __init__(self):

        # Initialize BGE Large
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
            breakpoint_threshold_amount=0.95,
            min_chunk_size=100,
        )

    def chunk_documents_from_db(self):  
        documents = self._get_documents()
        chunk_ids, document_ids = self._chunk_text(documents)
        logger.info(f'{len(chunk_ids)} successfully chunked and stored in database')
        logger.debug(f'Succesfully chunked text for Documents: {document_ids}')
        return len(chunk_ids)

    def _get_documents(self):
        db = DocumentDatabase(initialize_tables=False)
        documents = db.get_documents(Status.NOT_STARTED)
        return documents
    
    def _chunk_text(self, data: List[DocumentModel]) -> Tuple[List[int], List[int]]:
        """Chunk text and identify associated figure/table references."""
        if not data:
            return [], []
        chunk_ids, document_ids = [], []
        for row in data:
            text = row.PlainTextContent or ''
            if not text:
                continue
            chunks = self.chunker.split_text(text)

            db = DocumentDatabase(initialize_tables=False)
            chunk_data = [{'DocumentId': row.DocumentId, 'ChunkText': chunk} for chunk in chunks]
            chunk_ids.extend(db.insert_chunks(chunk_data, Status.CHUNKING_COMPLETED))
            logger.info(f'Chunked DocumentId {row.DocumentId} into {len(chunks)} chunks with ChunkIds: {chunk_ids[-len(chunks):]}'  )
            document_ids.append(row.DocumentId)
            db.update_document_status([row.DocumentId], Status.CHUNKING_COMPLETED)
        self.clean_cuda()
        return chunk_ids, document_ids

    def clean_cuda(self):
        """Clean CUDA memory"""
        clean_device_memory()
