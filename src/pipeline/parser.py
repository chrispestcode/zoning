from typing import Optional

import torch
import gc
import os
import re
import time
from tqdm import tqdm

from pathlib import Path
from utils.load_config import get_config
from loguru import logger
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from document_database import DocumentDatabase


class InvalidFileException(Exception):
    def __init__(self, file_path):
        Exception.__init__(self, f"Invalid file path: {file_path}")

class PdfParser:

    IMAGE_RESOLUTION_SCALE = 2.0
    output = Path("/home/tchrisp/logs/results.log")

    # Initialize PDFParser.device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        self._initialize_models()
        self.db = DocumentDatabase()
        self.config = get_config()
        pipeline_options = PdfPipelineOptions(do_ocr=False)
        pipeline_options.images_scale = PdfParser.IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )


    def _clean_cuda(self):
        """Clean CUDA memory"""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _initialize_models(self):
        """Initialize models with memory optimizations"""
        try:
            
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
        finally:
            self._clean_cuda()

    def _is_file_path_valid(self, file_path: str) -> Path:
        base_dir = self.config.get_config_value("paths", "data")

        file_path = file_path.replace("\\", "/").lstrip("/")
        if (Path(file_path).suffix.lower() != ".pdf"
                or not Path(file_path).suffix):
            raise InvalidFileException(file_path)
        file_stem = Path(file_path).stem

        has_only_valid_chars = bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", file_stem))
        if not has_only_valid_chars:
            raise InvalidFileException(file_path)

        if Path(file_stem).name != file_stem:
            raise InvalidFileException(file_path)

        base = Path(base_dir).resolve()
        target = (base / file_path).resolve()

        if target.is_relative_to(base) and target != base:
            return target
        raise InvalidFileException(file_path)

    def get_document_data(self, file_path: str) -> dict | None:
        self._clean_cuda()
        DOCUMENT_TYPE = "PDF"
        """Process text with memory optimization."""
        file_path = self._is_file_path_valid(file_path)
        filename = Path(file_path).stem
        if self.db.is_file_processed(filename):
            logger.debug(f"File '{filename}' is already processed.")
            return {
                'filename': filename,
                'document_type': DOCUMENT_TYPE,
                'document': None }

        start_time = time.time()
        converted_document = self.doc_converter.convert(file_path)
        filename = converted_document.input.file.stem

        document_data = {
            'filename': filename,
            'document_type': DOCUMENT_TYPE,
            'document': converted_document.document
        }

        logger.debug(f"Processed file '{filename}' in {time.time() - start_time:.2f} seconds")
        return document_data

    def process_pdf(self, file_path: str, opts: dict[str, bool] | None = None ) -> int | None:
        """Process a document, ensuring document_id exists before storing tables/images."""
        start_time = time.time()
        filename_for_logging = Path(file_path).stem
        document_data = self.get_document_data(file_path)
        if not document_data:
            logger.debug(f'Document could not be processed: {filename_for_logging}')
            return -1
        
        document_id = self.db.insert_document(document_data)
        log_ctx = logger.bind(filename=filename_for_logging, document_id=document_id) #noqa: Needed for sub-element processing
        if document_id == -1:
            logger.error(f"Failed to store document text or get document_id for {filename_for_logging}. Skipping image and table storage.")
            return -1
        log_ctx.info(f"Successfully stored text and obtained document_id {document_id}.")
        log_ctx.info(f"Document processing completed: {time.time() - start_time:.2f} seconds")
        return document_id

def main():
    processor = PdfParser()
    corpus_dir = "/home/tchrisp/projects/modern_search/corpus/"
    i = 1
    for file in tqdm(os.listdir(corpus_dir), desc=f"Processing file {i} of {len(os.listdir(corpus_dir))}..."):
        processor.process_pdf( corpus_dir + file)
        i+=1

if __name__ == "__main__":
    pass
    #main()

