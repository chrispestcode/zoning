# Description: This script is used to interact with the Qdrant vector search engine.

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, HnswConfigDiff
from sentence_transformers import SentenceTransformer
import torch
from models.contracts import Status
from document_database import DocumentDatabase
from utils.device import get_device, clean_device_memory
from typing import List, Dict, Optional
from loguru import logger
import gc
from dataclasses import dataclass
import psutil
import time
from qdrant_client.http import models
import numpy as np
from tqdm import tqdm
from utils.load_config import get_qdrant_host


@dataclass
class ModelConfig:
    """Configuration for models"""
    bge_model: str = "BAAI/bge-large-en-v1.5"
    clip_model: str = "openai/clip-vit-base-patch32"
    batch_size: int = 32
    max_length: int = 77
    cache_size: int = 1000
    memory_threshold: float = 0.9

class QdrantManager:
    def __init__(self, qdrant_url: str = get_qdrant_host(), timeout: int = 3600, initialize: bool = False):
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            timeout=timeout,
            prefer_grpc=False,  # Use HTTP instead of gRPC
            https=False
        )
        self.db = DocumentDatabase()
        self._setup_monitoring()
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600
        self.device = get_device()
        # Always initialize models as they're needed for search
        self._initialize_models()
        
        if initialize:
            self._initialize_collections()
            self.upsert_data()

    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            if hasattr(self, 'qdrant_client'):
                self.qdrant_client.close()
        except Exception as e:
            print(f"Error closing Qdrant client: {e}")

    def _setup_monitoring(self):
        """Setup system monitoring"""
        self.memory_threshold = ModelConfig.memory_threshold
        self._monitor_memory()

    def _monitor_memory(self):
        """Monitor system memory usage"""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold * 100:
                print(f"High memory usage detected: {memory.percent}%")
                self._clean_cuda()
                gc.collect()
        except Exception as e:
            print(f"Error monitoring memory: {e}")

    def _initialize_models(self):
        """Initialize models with memory optimizations"""
        try:
            # Initialize BGE Large model for text chunks
            self.bge_model = SentenceTransformer(
                ModelConfig.bge_model,
                device=self.device
            )
            self.clip_processor = None
            self.clip_model = None
            print("Models initialized successfully")
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise

    def _initialize_collections(self):
        """Initialize Qdrant collections with optimized settings"""
        try:

            collections = {
                "text": ("chunks", 1024),
                "images": ("images", 512),
                "tables": ("tables", 512),
            }
            for collection_name, (vector_name, size) in collections.items():
                if self.qdrant_client.collection_exists(collection_name=collection_name):
                    continue
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={vector_name: models.VectorParams(size=size, distance=models.Distance.COSINE)},
                    hnsw_config=HnswConfigDiff(m=32, ef_construct=150)
                )
            logger.info("Collections initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
            raise
        finally:
            self._clean_cuda()

    def _clean_cuda(self):
        """Clean CUDA memory"""
        clean_device_memory()

    def _batch_generator(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def upsert_data(self):
        """Upsert data from SQLite to Qdrant collections with on-the-fly embedding generation and batching"""
        try:
            chunk_rows = self.db.get_chunks(Status.EMBEDDING_COMPLETED)
            batch_size = ModelConfig.batch_size
            document_ids = set()
            for batch in tqdm(self._batch_generator(chunk_rows, batch_size), desc="Upserting data..."):
                points = []
                for row in batch:
                    if row.Embedding is None or not isinstance(row.Embedding, bytes):
                        continue
                    point_id = row.ChunkId
                    points.append(PointStruct(
                        id=point_id,
                        vector={'chunks': np.frombuffer(row.Embedding, dtype=np.float32)},
                        payload={
                            "type": "chunk",
                            "chunk_id": row.ChunkId,
                            "document_id": row.DocumentId,
                        }
                    ))
                if not points:
                    continue
                self.qdrant_client.upsert(collection_name='text', points=points, wait=True)
                self.db.update_chunk_status([p.payload["chunk_id"] for p in points], Status.PROCESSING_COMPLETED)
                document_ids.update(p.payload["document_id"] for p in points)
                self._clean_cuda()

            completed_ids = self.db.complete_documents(sorted(document_ids))
            logger.debug(f"Updated document status to PROCESSING_COMPLETED for {len(completed_ids)} documents.")

            logger.info("Data upserted successfully to Qdrant collections")
            return True
        except Exception as e:
            logger.error(f"Error upserting data to Qdrant: {e}")
            return False
        finally:
            self._clean_cuda()

    def encode_query_bge(self, query: str):
        """Encode query for text retrieval (BGE)"""
        return self.bge_model.encode(query).tolist()

    def encode_query_clip(self, query: str):
        """Encode query for image/table retrieval (CLIP)"""
        clip_inputs = self.clip_processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        with torch.no_grad():
            clip_emb = self.clip_model.get_text_features(**clip_inputs)
        return clip_emb[0].cpu().numpy().tolist()

    def _check_collection_status(self, collection_name: str) -> bool:
        """Check if a collection exists and has points"""
        try:
            collection_info = self.qdrant_client.get_collection(collection_name=collection_name)
            return collection_info.points_count > 0
        except Exception:
            return False

    def query_qdrant(self, query: str, top_k: int = 10) -> Dict[str, List]:
        """Hybrid search with BGE for text, CLIP for images/tables, then CLIP reranking"""
        try:

            collections_status = {
                "text": self._check_collection_status("text"),
                "images": self._check_collection_status("images"),
                "tables": self._check_collection_status("tables")
            }
            
            if not any(collections_status.values()):
                print("No collections found with data. Please run execute_qdrant() first.")
                return []

            chunk_results = []
            if collections_status["text"]:
                chunk_results = self.qdrant_client.query_points(
                    collection_name="text",
                    query=self.encode_query_bge(query),
                    using="chunks",
                    limit=top_k,
                    with_payload=True,
                    with_vectors=True
                ).points
           
            return chunk_results
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
        finally:
            # Clean up resources
            self._clean_cuda()

    def get_results(self, qdrant_results):
        chunk_ids = [
            points.payload.get('chunk_id') + i
            for points in qdrant_results
            for i in range(-3, 3)
        ]
        return self.db.get_chunks_by_ids(chunk_ids)

def get_qdrant() -> Optional[QdrantManager]:
    """Main function to execute Qdrant operations"""
    try:
        qdrant_manager = QdrantManager()
        print("Qdrant operations completed successfully")
        if not qdrant_manager._check_collection_status("text"):
            raise Exception("No data found in for qdrant collections. Please instantiate QdrantManager() with initialize=True first.")
        return qdrant_manager
    except Exception as e:
        logger.error(f"Error in execute_qdrant: {e}")
        raise

if __name__ == "__main__":
    qdrant = get_qdrant()
    results = qdrant.query_qdrant("Information about taxes?")
    print(results)