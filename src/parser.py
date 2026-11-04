import torch
import gc
import os
import time
from tqdm import tqdm

from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from document_database import DocumentDatabase
from utils.docling_utils import process_tables, process_images, process_forms

class PDFParser:

    IMAGE_RESOLUTION_SCALE = 2.0
    output = Path("/home/tchrisp/logs/results.log")

    # Initialize PDFParser.device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        self._initialize_models()
        self.db = DocumentDatabase()


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


#TODO: worker logic integrated here
    def get_document_data(self, file_path: str) -> Optional[Any]:
        self._clean_cuda()
        """Process text with memory optimization."""
        try:
            # Check if file is already processed
            doc_filename = Path(file_path).stem
            if self.db.is_file_processed(doc_filename):
                print(f"File '{doc_filename}' is already processed.")
                return None

            # Configure pipeline options
            pipeline_options = PdfPipelineOptions(do_ocr=False)
            pipeline_options.images_scale = PDFParser.IMAGE_RESOLUTION_SCALE
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True

            ''' USE  document_data['document'].groups[1].label
                OUTPUT : <GroupLabel.FORM_AREA: 'form_area'> 
                to get FORMS
            '''

            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            start_time = time.time()
            converted_document = doc_converter.convert(file_path)
            doc_filename = converted_document.input.file.stem

            document_data = {
                'filename': doc_filename,
                'document_type': "PDF",
                'document': converted_document.document
            }

            logger.debug(f"Processed file '{doc_filename}' in {time.time() - start_time:.2f} seconds")
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return None
        finally:
            self._clean_cuda()
    
    
                    
    def process_pdf(self, file_path: str, opts: dict[str, bool] | None = None ) -> Dict[str, Any]:
        """Process a document, ensuring document_id exists before storing tables/images."""
        start_time = time.time() # Track overall time
        filename_for_logging = Path(file_path).stem # Get filename early for logging
        document_data = self.get_document_data(file_path)
        if not document_data:
            logger.debug(f'Document could not be processed: {filename_for_logging}')
            return {}
        
        document_id = self.db.insert_document(document_data)
        log_ctx = logger.bind(filename=filename_for_logging, document_id=document_id)
        if document_id == -1:
            logger.error(f"Failed to store document text or get document_id for {filename_for_logging}. Skipping image and table storage.")
            return {} 
        
        document_data['document_id'] = document_id
        document = document_data['document']
        log_ctx.info(f"Successfully stored text and obtained document_id {document_id}.")
        
        chunk_ids, table_ids, image_ids, form_ids = [], [], [], []

        # if opts.get("tables", False):
            # tables = process_tables(document, self.device)
            # if not tables:
            #     logger.info(f"No valid tables processed or found.")
            # else:
            #     log_ctx.debug(f'Extracted {len(tables)} tables.')
            #     table_ids =  self.db.insert_table_data(document_id, tables)
            #     log_ctx.debug(f'Inserted {len(table_ids)} tables @ {[tbl.get('page_no', None) for tbl in tables]}')
            #     log_ctx.info(f'Successfully processed {len(table_ids)} tables.')
            #
        # if opts.get("images", False):
            # images = process_images(document, self.device)
            # if not images:
            #     logger.info(f"No valid images processed or found.")
            # else:
            #     log_ctx.debug(f'Extracted {len(images)} images.')
            #     image_ids =  self.db.insert_images(document_id, images)
            #     log_ctx.debug(f'Inserted {len(image_ids)} images @ {[img.get('page_no', None) for img in images]}.')
            #     log_ctx.info(f'Successfully processed {len(image_ids)} images.')
        # if opts.get("forms", False):
            # forms = process_forms(document)
            # if not forms:
            #     logger.info(f"No valid forms processed or found.")
            # else:
            #     log_ctx.debug(f'Extracted {len(forms)} forms.')
            #     form_ids =  self.db.insert_form_data(document_id, forms)
            #     log_ctx.debug(f'Inserted {len(form_ids)} forms @ {[frm.get('page_no', None) for frm in forms]}.')
            #     log_ctx.info(f'Successfully processed {len(form_ids)} forms.')

        document_data['metadata'] = {
            'file_name': '',
            'file_size_in_kb': '',
            'author': '',
            'creation_date': '',
            'modification_date': '',
            'format': '',
            'keywords': '',
            'image_count': len(image_ids),
            'table_count': len(table_ids),
            'form_count': len(form_ids),
            'processing_time_seconds': time.time() - start_time
        }

        log_ctx.info(f"Document processing completed: {time.time() - start_time:.2f} seconds")
        
        self._clean_cuda()

def main():
    processor = PDFParser()
    corpus_dir = "/home/tchrisp/projects/modern_search/corpus/"
    i = 1
    for file in tqdm(os.listdir(corpus_dir), desc=f"Processing file {i} of {len(os.listdir(corpus_dir))}..."):
        processor.process_pdf( corpus_dir + file)
        i+=1

if __name__ == "__main__":
    pass
    #main()

