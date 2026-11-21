# Description: This file contains the Embeddings class which is responsible for generating embeddings for text and images.
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import PIL.ImageOps
import torch
from models.contracts import Status
from document_database import DocumentDatabase
from utils.device import get_device, clean_device_memory
from typing import List
import numpy as np
from loguru import logger
from tqdm import tqdm




class Embedder:
    def __init__(self):
        self.db = DocumentDatabase(initialize_tables=False)
        self.device = get_device()

        # Load BGE Large
        bge_model_name = "BAAI/bge-large-en-v1.5"
        self.text_embedder = SentenceTransformer(bge_model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(bge_model_name)
        self.model = AutoModel.from_pretrained(bge_model_name)

    '''
    def get_image_embedding(self, image):
        try:
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            return image_features.cpu().detach().numpy()  # Convert to numpy array
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return None
        finally:
            self._clean_cuda()
    '''
    def _clean_cuda(self):
        """Clean CUDA memory"""
        clean_device_memory()

    def generate_text_embeddings(self, chunks):
        embeddings = []
        for chunk in chunks:
        # Converts text chunks into numerical tokens
            inputs = self.tokenizer(chunk, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)

            # Generate text embeddings
            with torch.no_grad():
                embd = self.model.get_text_features(**inputs)
        
            embeddings.append(embd.cpu().numpy())
        
   
        return embeddings
    
    
    def generate_text_embeddings_bge(self, text: List[str]) -> List[np.ndarray]:
        """Generate embeddings for text chunks using BGE Large"""
        try:
            if not text:
                print("No chunks provided for embedding generation")
                return []
            
            # Generate embeddings
            embeddings = self.text_embedder.encode(text, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error generating text embeddings: {e}")
            return []
        finally:
            self._clean_cuda()

    def generate_image_embeddings_clip(self, images: List[PIL.Image.Image]) -> List[np.ndarray]:
        """Generate embeddings for images using CLIP"""
        try:
            if not images:
                print("No images provided for embedding generation")
                return []
            
            embeddings = []
            for image in images:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                embeddings.append(image_features.cpu().numpy())
            
            return embeddings
        except Exception as e:
            print(f"Error generating image embeddings: {e}")
            return []
        finally:
            self._clean_cuda()
    
    def process_incomplete_chunks(self):
        chunk_ids = []
        BATCH_SIZE  = 300
        chunks  = self.db.get_chunks(Status.CHUNKING_COMPLETED)
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Processing Chunks..."):
            batch = chunks[i:i + BATCH_SIZE]
            cids = [row.ChunkId for row in batch]
            embeddings = self.generate_text_embeddings_bge([row.ChunkText for row in batch])
            if len(embeddings) != len(cids):
                logger.error(f"Embedding count mismatch for batch starting at {i}: {len(embeddings)} != {len(cids)}")
                continue
            chunk_ids.extend(self.db.insert_embedding_by_chunk_id(cids, [embd.astype(np.float32).tobytes() for embd in embeddings]))
        logger.debug(f"Stored embeddings for {len(chunk_ids)} chunk IDs.")
        

