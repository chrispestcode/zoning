"""
Pydantic contracts for design-by-contract validation.

This module defines Pydantic models for:
- Database entity models (Document, Chunk, Image, TableData, Form, Metadata, Derivative, QAPair)
- Query response models for database operations
- Qdrant query response models
- Shared base models for common fields (timestamps, status)

Usage:
    # Validate database query results
    from contracts import validate_chunk_dict, ChunkModel
    
    chunk_data = db.get_chunks_by_ids([1])[0]
    validated_chunk = validate_chunk_dict(chunk_data)  # Returns ChunkModel or None
    
    # Validate Qdrant query results
    from contracts import validate_query_response, QdrantQueryResponse
    
    results = qdrant_manager.query_qdrant("query text")
    validated_results = validate_query_response(results)  # Returns QdrantQueryResponse or QdrantEmptyResponse

Note on BLOB fields:
    Fields like Embedding, Base64Encoding, Dataframe, and FormText are stored as BLOB in SQLite.
    In JSON serialization, these may be represented as base64-encoded strings or remain as bytes.
    The models use Union[bytes, str] to accommodate both representations.

Note on return type discrepancies:
    Some database methods have type hints that don't match their actual return types:
    - get_document_by_id() type hint says Optional[Dict[str, Any]] but actually returns Dict[int, Dict[str, Any]]
    - query_qdrant() type hint says Dict[str, List] but actually returns List[Optional[Dict[str, Any]]]
    The contracts reflect the actual return types.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import ConfigDict, Field, BaseModel

class Status(str, Enum):
    """Status enumeration matching DocumentDatabase.Status"""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    CHUNKING_COMPLETED = "CHUNKING_COMPLETED"
    EMBEDDING_COMPLETED = "EMBEDDING_COMPLETED"
    PROCESSING_COMPLETED = "PROCESSING_COMPLETED"


# Alias used in mixins to prevent the field name 'Status' from shadowing
# the Status enum in Pydantic v2's get_type_hints() evaluation.
_StatusEnum = Status


class TimestampMixin(BaseModel):
    """Base model for entities with CreatedAt and UpdatedAt timestamps"""
    CreatedAt: Optional[str] = Field(None, description="Creation timestamp")
    UpdatedAt: Optional[str] = Field(None, description="Last update timestamp")

    model_config = ConfigDict(
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    )


class StatusMixin(BaseModel):
    """Base model for entities with Status field"""
    Status: Optional[_StatusEnum] = Field(None, description="Processing status")


# ============================================================================
# Database Entity Models
# ============================================================================

class DocumentModel(TimestampMixin, StatusMixin):
    """Model for Document table records"""
    DocumentId: int = Field(..., description="Primary key")
    FileName: str = Field(..., description="Document filename")
    DocumentType: str = Field(..., description="Type of document")
    PlainTextContent: Optional[str] = Field(None, description="Plain text content of the document")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "DocumentId": 1,
                "FileName": "example.pdf",
                "DocumentType": "PDF",
                "PlainTextContent": "Sample document content...",
                "CreatedAt": "2024-01-01 12:00:00.000",
                "UpdatedAt": "2024-01-01 12:00:00.000",
                "Status": "PROCESSING_COMPLETED"
            }
        }
    )


class ChunkModel(TimestampMixin, StatusMixin):
    """Model for Chunk table records"""
    ChunkId: int = Field(..., description="Primary key")
    DocumentId: int = Field(..., description="Foreign key to Document")
    ChunkText: str = Field(..., description="Text content of the chunk")
    FigureRefs: Optional[str] = Field(None, description="References to figures in the chunk")
    Embedding: Optional[Union[bytes, str]] = Field(
        None,
        description="Embedding vector as BLOB. May be bytes or base64-encoded string in JSON."
    )

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "ChunkId": 1,
                "DocumentId": 1,
                "ChunkText": "Sample chunk text...",
                "FigureRefs": None,
                "Embedding": None,
                "CreatedAt": "2024-01-01 12:00:00.000",
                "UpdatedAt": "2024-01-01 12:00:00.000",
                "Status": "EMBEDDING_COMPLETED"
            }
        }
    )


class ImageModel(TimestampMixin, StatusMixin):
    """Model for Image table records"""
    ImageId: int = Field(..., description="Primary key")
    ImageRef: int = Field(..., description="Image reference number")
    Base64Encoding: Union[bytes, str] = Field(
        ...,
        description="Base64 encoded image. May be bytes or base64-encoded string in JSON."
    )
    DocumentId: int = Field(..., description="Foreign key to Document")
    MimeType: str = Field(..., description="MIME type of the image")
    Height: int = Field(..., description="Image height in pixels")
    Width: int = Field(..., description="Image width in pixels")
    Dpi: int = Field(..., description="Image DPI")
    PageNo: Optional[int] = Field(None, description="Page number where image appears")
    AltText: Optional[str] = Field(None, description="Alternative text for the image")
    BoundingBox: Optional[str] = Field(None, description="Bounding box coordinates")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "ImageId": 1,
                "ImageRef": 1,
                "Base64Encoding": b"base64_encoded_data",
                "DocumentId": 1,
                "MimeType": "image/png",
                "Height": 800,
                "Width": 600,
                "Dpi": 300,
                "PageNo": 1,
                "AltText": "Sample image",
                "BoundingBox": None,
                "CreatedAt": "2024-01-01 12:00:00.000",
                "UpdatedAt": "2024-01-01 12:00:00.000",
                "Status": "EMBEDDING_COMPLETED"
            }
        }
    )

class TableDataModel(TimestampMixin, StatusMixin):
    """Model for TableData table records"""
    TableId: int = Field(..., description="Primary key")
    TableRef: Optional[int] = Field(None, description="Table reference number")
    Base64Encoding: Union[bytes, str] = Field(
        ..., 
        description="Base64 encoded table image. May be bytes or base64-encoded string in JSON."
    )
    Dataframe: Union[bytes, str] = Field(
        ..., 
        description="Serialized dataframe data. May be bytes or base64-encoded string in JSON."
    )
    DocumentId: int = Field(..., description="Foreign key to Document")
    NumRows: int = Field(..., description="Number of rows in the table")
    NumColumns: int = Field(..., description="Number of columns in the table")
    BoundingBox: Optional[str] = Field(None, description="Bounding box coordinates")
    PageNo: Optional[int] = Field(None, description="Page number where table appears")
    AltText: Optional[str] = Field(None, description="Alternative text for the table")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "TableId": 1,
                "TableRef": 1,
                "Base64Encoding": b"base64_encoded_data",
                "Dataframe": b"serialized_dataframe",
                "DocumentId": 1,
                "NumRows": 10,
                "NumColumns": 5,
                "BoundingBox": None,
                "PageNo": 1,
                "AltText": "Sample table",
                "CreatedAt": "2024-01-01 12:00:00.000",
                "UpdatedAt": "2024-01-01 12:00:00.000",
                "Status": "EMBEDDING_COMPLETED"
            }
        }
    )


class FormModel(TimestampMixin, StatusMixin):
    """Model for Form table records"""
    FormId: int = Field(..., description="Primary key")
    Base64Encoding: Union[bytes, str] = Field(
        ..., 
        description="Base64 encoded form image. May be bytes or base64-encoded string in JSON."
    )
    FormText: Union[bytes, str] = Field(
        ..., 
        description="Form text content. May be bytes or base64-encoded string in JSON."
    )
    FormTitle: Union[bytes, str] = Field(
        ...,
        description="Form title text. May be bytes or base64-encoded string in JSON."
    )
    DocumentId: int = Field(..., description="Foreign key to Document")
    PageNo: Optional[int] = Field(None, description="Page number where form appears")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "FormId": 1,
                "Base64Encoding": b"base64_encoded_data",
                "FormText": b"form_text_content",
                "FormTitle": b"form_title",
                "DocumentId": 1,
                "PageNo": 1,
                "CreatedAt": "2024-01-01 12:00:00.000",
                "UpdatedAt": "2024-01-01 12:00:00.000",
                "Status": "EMBEDDING_COMPLETED"
            }
        }
    )


class MetadataModel(TimestampMixin):
    """Model for Metadata table records"""
    MetadataId: int = Field(..., description="Primary key")
    DocumentId: int = Field(..., description="Foreign key to Document")
    Key: str = Field(..., description="Metadata key")
    Value: str = Field(..., description="Metadata value")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "MetadataId": 1,
                "DocumentId": 1,
                "Key": "author",
                "Value": "John Doe",
                "CreatedAt": "2024-01-01 12:00:00.000",
                "UpdatedAt": "2024-01-01 12:00:00.000"
            }
        }
    )


class DerivativeCreate(BaseModel):
    """Input model for insert_derivatives() - DerivativeId and timestamps are assigned by the database"""
    DocumentId: int = Field(..., description="Foreign key to Document")
    ChunkId: int = Field(..., description="Foreign key to Chunk")
    DerivativeText: str = Field(..., min_length=1, description="Derivative text (compressed on write)")

    model_config = ConfigDict(
        str_strip_whitespace = True,
        json_schema_extra = {
            "example": {
                "DocumentId": 1,
                "ChunkId": 1,
                "DerivativeText": "Summary of the zoning ordinance..."
            }
        }
    )


class DerivativeModel(TimestampMixin):
    """Model for Derivative table records (text is decompressed on read)"""
    DerivativeId: int = Field(..., description="Primary key")
    DocumentId: int = Field(..., description="Foreign key to Document")
    ChunkId: int = Field(..., description="Foreign key to Chunk")
    DerivativeText: str = Field(..., description="Decompressed derivative text")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "DerivativeId": 1,
                "DocumentId": 1,
                "ChunkId": 1,
                "DerivativeText": "Summary of the zoning ordinance...",
                "CreatedAt": "2024-01-01 12:00:00.000",
                "UpdatedAt": "2024-01-01 12:00:00.000"
            }
        }
    )


class QAPairCreate(BaseModel):
    """Input model for insert_qa_pairs() - QAPairId and timestamps are assigned by the database"""
    Question: str = Field(..., min_length=1, description="Question text")
    Answer: str = Field(..., min_length=1, description="Answer text")
    DocumentIds: List[int] = Field(..., min_length=1, description="Source documents (written to QAPairDocument)")

    model_config = ConfigDict(
        str_strip_whitespace = True,
        json_schema_extra = {
            "example": {
                "Question": "What is the minimum lot size in R-1?",
                "Answer": "7,500 square feet.",
                "DocumentIds": [1, 4]
            }
        }
    )


class QAPairModel(TimestampMixin):
    """Model for QAPair table records joined with their source documents"""
    QAPairId: int = Field(..., description="Primary key")
    Question: str = Field(..., description="Question text")
    Answer: str = Field(..., description="Answer text")
    DocumentIds: List[int] = Field(default_factory=list, description="Source documents (via QAPairDocument)")
    DerivativeIds: List[int] = Field(default_factory=list, description="Derivative ids")
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "QAPairId": 1,
                "Question": "What is the minimum lot size in R-1?",
                "Answer": "7,500 square feet.",
                "DocumentIds": [1, 4],
                "DerivativeIds": [1],
                "CreatedAt": "2024-01-01 12:00:00.000",
                "UpdatedAt": "2024-01-01 12:00:00.000"
            }
        }
    )


# ============================================================================
# Response Models for Database Queries
# ============================================================================

class DocumentListResponse(BaseModel):
    """Response model for get_documents() - list of documents"""
    documents: List[DocumentModel] = Field(default_factory=list, description="List of documents")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "DocumentId": 1,
                        "FileName": "example.pdf",
                        "DocumentType": "PDF",
                        "PlainTextContent": "Sample content...",
                        "CreatedAt": "2024-01-01 12:00:00.000",
                        "UpdatedAt": "2024-01-01 12:00:00.000",
                        "Status": "PROCESSING_COMPLETED"
                    }
                ]
            }
        }
    )

class DocumentByIdResponse(BaseModel):
    """Response model for get_document_by_id() - dictionary mapping document_id to document.
    
    Note: The actual return type is Dict[int, Dict[str, Any]] which can be empty or contain None values.
    This model represents the validated structure.
    """
    documents: Dict[int, Optional[DocumentModel]] = Field(
        default_factory=dict, 
        description="Dictionary of documents by ID. Keys are document IDs, values are document models or None."
    )

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "documents": {
                    1: {
                        "DocumentId": 1,
                        "FileName": "example.pdf",
                        "DocumentType": "PDF",
                        "PlainTextContent": "Sample content...",
                        "CreatedAt": "2024-01-01 12:00:00.000",
                        "UpdatedAt": "2024-01-01 12:00:00.000",
                        "Status": "PROCESSING_COMPLETED"
                    }
                }
            }
        }
    )

class ChunkListResponse(BaseModel):
    """Response model for get_chunks() - list of chunks"""
    chunks: List[ChunkModel] = Field(default_factory=list, description="List of chunks")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "chunks": [
                    {
                        "ChunkId": 1,
                        "DocumentId": 1,
                        "ChunkText": "Sample chunk text...",
                        "FigureRefs": None,
                        "Embedding": None,
                        "CreatedAt": "2024-01-01 12:00:00.000",
                        "UpdatedAt": "2024-01-01 12:00:00.000",
                        "Status": "EMBEDDING_COMPLETED"
                    }
                ]
            }
        }
    )


class ImageListResponse(BaseModel):
    """Response model for get_images() - list of images"""
    images: List[ImageModel] = Field(default_factory=list, description="List of images")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "images": [
                    {
                        "ImageId": 1,
                        "ImageRef": 1,
                        "Base64Encoding": b"base64_data",
                        "DocumentId": 1,
                        "MimeType": "image/png",
                        "Height": 800,
                        "Width": 600,
                        "Dpi": 300,
                        "PageNo": 1,
                        "AltText": "Sample image",
                        "BoundingBox": None,
                        "CreatedAt": "2024-01-01 12:00:00.000",
                        "UpdatedAt": "2024-01-01 12:00:00.000",
                        "Status": "EMBEDDING_COMPLETED"
                    }
                ]
            }
        }
    )

class TableListResponse(BaseModel):
    """Response model for get_tables() - list of tables"""
    tables: List[TableDataModel] = Field(default_factory=list, description="List of tables")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "tables": [
                    {
                        "TableId": 1,
                        "TableRef": 1,
                        "Base64Encoding": b"base64_data",
                        "Dataframe": b"dataframe_data",
                        "DocumentId": 1,
                        "NumRows": 10,
                        "NumColumns": 5,
                        "BoundingBox": None,
                        "PageNo": 1,
                        "AltText": "Sample table",
                        "CreatedAt": "2024-01-01 12:00:00.000",
                        "UpdatedAt": "2024-01-01 12:00:00.000",
                        "Status": "EMBEDDING_COMPLETED"
                    }
                ]
            }
        }
    )


class FormListResponse(BaseModel):
    """Response model for get_forms() - list of forms"""
    forms: List[FormModel] = Field(default_factory=list, description="List of forms")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "forms": [
                    {
                        "FormId": 1,
                        "Base64Encoding": b"base64_data",
                        "FormText": b"form_text",
                        "DocumentId": 1,
                        "PageNo": 1,
                        "CreatedAt": "2024-01-01 12:00:00.000",
                        "UpdatedAt": "2024-01-01 12:00:00.000",
                        "Status": "EMBEDDING_COMPLETED"
                    }
                ]
            }
        }
    )

class MetadataListResponse(BaseModel):
    """Response model for get_metadata() and get_metadata_by_document_id() - list of metadata"""
    metadata: List[MetadataModel] = Field(default_factory=list, description="List of metadata entries")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "metadata": [
                    {
                        "MetadataId": 1,
                        "DocumentId": 1,
                        "Key": "author",
                        "Value": "John Doe",
                        "CreatedAt": "2024-01-01 12:00:00.000",
                        "UpdatedAt": "2024-01-01 12:00:00.000"
                    }
                ]
            }
        }
    )


class DerivativeListResponse(BaseModel):
    """Response model for get_derivatives_by_document_id() - list of derivatives"""
    derivatives: List[DerivativeModel] = Field(default_factory=list, description="List of derivatives")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "derivatives": [
                    {
                        "DerivativeId": 1,
                        "DocumentId": 1,
                        "ChunkId": 1,
                        "DerivativeText": "Summary of the zoning ordinance...",
                        "CreatedAt": "2024-01-01 12:00:00.000",
                        "UpdatedAt": "2024-01-01 12:00:00.000"
                    }
                ]
            }
        }
    )

# ============================================================================
# Qdrant Query Response Models
# ============================================================================

class QdrantQueryResponse(BaseModel):
    """Response model for query_qdrant() - returns a list of chunks"""
    chunks: List[Optional[ChunkModel]] = Field(
        default_factory=list,
        description="List of chunk results from vector search. May contain None values if chunk lookup fails."
    )

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "chunks": [
                    {
                        "ChunkId": 1,
                        "DocumentId": 1,
                        "ChunkText": "Relevant chunk text matching the query...",
                        "FigureRefs": None,
                        "Embedding": None,
                        "CreatedAt": "2024-01-01 12:00:00.000",
                        "UpdatedAt": "2024-01-01 12:00:00.000",
                        "Status": "EMBEDDING_COMPLETED"
                    }
                ]
            }
        }
    )

class QdrantEmptyResponse(BaseModel):
    """Response model for query_qdrant() when no collections are found"""
    results: List[Any] = Field(default_factory=list, description="Empty results list")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "results": []
            }
        }
    )

# ============================================================================
# Union Types for Flexible Responses
# ============================================================================

QdrantQueryResult = Union[QdrantQueryResponse, QdrantEmptyResponse]
"""Union type for query_qdrant() return values"""


# ============================================================================
# Helper Functions for Validation
# ============================================================================

def validate_document_dict(data: Dict[str, Any]) -> DocumentModel:
    """Validate and convert a dictionary to DocumentModel"""
    return DocumentModel(**data)


def validate_chunk_dict(data: Optional[Dict[str, Any]]) -> Optional[ChunkModel]:
    """Validate and convert a dictionary to ChunkModel"""
    if data is None:
        return None
    return ChunkModel(**data)


def validate_chunk_list(data: List[Dict[str, Any]]) -> List[ChunkModel]:
    """Validate and convert a list of dictionaries to ChunkModel list"""
    return [ChunkModel(**item) for item in data]


def validate_query_response(data: Union[List[Optional[Dict[str, Any]]], Dict[str, List]]) -> Union[QdrantQueryResponse, QdrantEmptyResponse]:
    """Validate and convert query_qdrant() response to appropriate response model.
    
    Args:
        data: Either a list of chunk dictionaries (from get_results) or 
              a dict with "results" key containing empty list (error case)
    
    Returns:
        QdrantQueryResponse if data is a list, QdrantEmptyResponse if data is a dict
    """
    if isinstance(data, dict) and "results" in data:
        return QdrantEmptyResponse(**data)
    elif isinstance(data, list):
        chunks = [validate_chunk_dict(item) for item in data]
        return QdrantQueryResponse(chunks=chunks)
    else:
        # Fallback: treat as empty response
        return QdrantEmptyResponse(results=[])
