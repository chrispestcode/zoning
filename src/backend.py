from src.content_generator import ContentGenerator
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant import QdrantManager
from document_database import DocumentDatabase

# Initialize FastAPI app
app = FastAPI()
db = DocumentDatabase(initialize_tables=False)
qdrant_manager = QdrantManager(initialize=False)  # Don't initialize collections on startup
content_generator = ContentGenerator()
# Request model for API
class QueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search_documents(request: QueryRequest):
    query_text = request.query  # Get the query text from the request

    # Print message in the terminal (for debugging)
    print(f"🔍 Received search query: {query_text}")

    # Get results from Qdrant
    qdrant_results = qdrant_manager.query_qdrant(query_text)
    response = {}
    if isinstance(qdrant_results, dict) and qdrant_results:
        hits = [res for res in qdrant_results[:20] if res is not None]
        document_ids = list({res.DocumentId for res in hits})
        chunks = [res.ChunkText for res in hits]
        response["text_response"] = await content_generator.get_chunk_response(query_text, chunks)

        query_documents = db.get_document_by_id(document_ids)
        query_forms = db.get_forms_by_document_id(document_ids)

        response["query_documents"] = {
            doc_id: doc.model_dump() if doc is not None else None
            for doc_id, doc in query_documents.items()
        }

        response["query_forms"] =  [form.model_dump() if form is not None else None
            for form in query_forms]
    return {"response": response}



# Run server using: uvicorn src.backend:app --reload
