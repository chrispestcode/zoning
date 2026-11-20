# Description: This is the main file that will be used to run the entire project.
import cProfile
import os
import pstats

from document_database import DocumentDatabase
from pipeline.parser import PdfParser
from pipeline.chunker import Chunker
from pipeline.embedding import Embedder
from utils.load_config import get_config, get_config_path
from qdrant import QdrantManager
from aws.aws_session import AwsSession
from tempfile import TemporaryDirectory as TempDirectory
from utils.utils import _initialize_logger

logger = _initialize_logger()
# Instantiate clients

cloud_storage_enabled = False

def main():
    config = get_config()
    document_files = []
    corpus_directory = get_config_path(config, 'paths', 'data')
    if not os.path.exists(corpus_directory):
        return


    def parse_files(document_files):
        parser = PdfParser()
        for file in document_files:
            parser.process_pdf(file)
    
    def chunk_files():
        chunker = Chunker()
        chunker.chunk_documents_from_db()

    def generate_content():
        db = DocumentDatabase()
        tables = db.get_chunk_table()
        #content_generator = ContentGenerator(chunk for chunk in [row['Chunk'] for row in tables])
        #content_generator.get_derivatives_chunks()
    def generate_embeddings():
        embedder = Embedder()
        embedder.process_incomplete_chunks()
    
    def upsert_qdrant():
        qdrant_manager = QdrantManager(initialize=True)
        
    if cloud_storage_enabled:
        session = AwsSession(
            role_arn="arn:aws:iam::158179669543:role/s3-read-only",
            region_name="us-east-1")
        client = session.get_s3_client()
        bucket = client.list_buckets()[1]
        with TempDirectory() as temp_dir:
            document_files = []
            for i, aws_object in enumerate(client.list_objects(bucket_name=bucket)['Contents']):
                object_aws = (client.get_object(bucket_name=bucket, object_name=aws_object['Key']))
                with open(os.path.join(temp_dir, aws_object['Key']), 'wb') as f:
                    f.write(object_aws)
                document_files.append(os.path.join(temp_dir, aws_object['Key']))
                if i > 3:
                    break
    else:
        for root, dirs, files in os.walk(corpus_directory):
            for name in files:
                if name.lower().endswith('.pdf'):
                    document_files.append(os.path.join(root, name))
    #parse_files(document_files)
    logger.info("PARSE COMPLETE")
    #chunk_files()
    logger.info("CHUNKING COMPLETE")
    # generate_content()
    logger.info("CONTENT GENERATION COMPLETE")
    #generate_embeddings()
    logger.info("EMBEDDING GENERATION COMPLETE")
    #upsert_qdrant()
    logger.info("QDRANT UPSERT COMPLETE")
    logger.eval("This work?")


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        main()

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
    results.dump_stats("./logs/profile.prof")