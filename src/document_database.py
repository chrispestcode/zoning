import sqlite3
import json
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from pydantic import ValidationError

from models.contracts import (
    Status,
    DocumentModel,
    ChunkModel,
    ImageModel,
    TableDataModel,
    FormModel,
    MetadataModel,
    DerivativeCreate,
    DerivativeModel,
    QAPairCreate,
    QAPairModel,
)
from utils.load_config import get_config


class InvalidPathException(Exception):
    pass

class DocumentDatabase:
    """SQLite-backed document store.

    Manages persistence for documents, chunks, images, tables, forms, and
    metadata.  Processing state is tracked via :class:`contracts.Status`.

    Example:
        db = DocumentDatabase()
        doc_id = db.insert_document({"filename": "report", "document_type": "PDF", "document": doc})
        chunks = db.get_chunks()  # List[ChunkModel]
    """

    def __init__(self, db_name= None, initialize_tables = True):
        if not db_name:
            config = get_config()
            path = config.get_config_path
            if not Path(path).exists():
                raise InvalidPathException("Invalid path provided for database")
            self.db_name = path.as_posix()

        else:
            self.db_name = db_name

        self._transaction_conn: Optional[sqlite3.Connection] = None
        if initialize_tables:
            self.create_tables()


    @contextmanager
    def get_connection(self):
        """
        Yield a connection. Outside a transaction it commits on success, rolls back on
        error and closes. Inside transaction() it yields the shared connection and leaves
        commit/rollback to the transaction.
        """
        if self._transaction_conn is not None:
            yield self._transaction_conn
            return
        conn = sqlite3.connect(self.db_name)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # WARNING: "Not thread-safe: one DocumentDatabase instance per thread."
    @contextmanager
    def transaction(self):
        """
        Make several DAO calls atomic: they all commit together, or all roll back if
        any raises. Nested transaction() blocks join the outer one.

        Example:
            with db.transaction():
                db.insert_chunks(chunks, Status.CHUNKING_COMPLETED)
                db.update_document_status([document_id], Status.CHUNKING_COMPLETED)
        """
        if self._transaction_conn is not None:
            yield
            return
        with self.get_connection() as conn:
            self._transaction_conn = conn
            try:
                yield
            finally:
                self._transaction_conn = None

    def _in_transaction(self) -> bool:
        return self._transaction_conn is not None

    def create_tables(self):
        """Create tables with optimized indexes"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Document (
                        DocumentId INTEGER PRIMARY KEY,
                        FileName TEXT NOT NULL,
                        DocumentType TEXT NOT NULL,
                        PlainTextContent TEXT,
                        BlobStorageURI TEXT,
                        CreatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        UpdatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        Status TEXT CHECK (Status IN ('NOT_STARTED', 'IN_PROGRESS', 'CHUNKING_COMPLETED', 'EMBEDDING_COMPLETED', 'PROCESSING_COMPLETED'))
                    )
                ''')


                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Chunk (
                        ChunkId INTEGER PRIMARY KEY,
                        DocumentId INTEGER NOT NULL,
                        ChunkText TEXT NOT NULL,
                        FigureRefs TEXT,
                        Embedding BLOB,
                        CreatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        UpdatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        Status TEXT CHECK (Status IN ('NOT_STARTED', 'IN_PROGRESS', 'CHUNKING_COMPLETED', 'EMBEDDING_COMPLETED', 'PROCESSING_COMPLETED')),
                        FOREIGN KEY (DocumentID) REFERENCES Document(DocumentID)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Image (
                        ImageId INTEGER PRIMARY KEY AUTOINCREMENT,
                        ImageRef INTEGER NOT NULL,
                        Base64Encoding BLOB NOT NULL,
                        DocumentId INTEGER NOT NULL,
                        MimeType TEXT NOT NULL,
                        Height INTEGER NOT NULL,
                        Width INTEGER NOT NULL,
                        Dpi INTEGER NOT NULL,
                        PageNo INTEGER,
                        AltText TEXT,
                        BoundingBox TEXT,
                        CreatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        UpdatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        Status TEXT CHECK (Status IN ('NOT_STARTED', 'IN_PROGRESS', 'CHUNKING_COMPLETED', 'EMBEDDING_COMPLETED', 'PROCESSING_COMPLETED')),
                        FOREIGN KEY (DocumentID) REFERENCES Document(DocumentID)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS TableData (
                        TableId INTEGER PRIMARY KEY AUTOINCREMENT,
                        TableRef INTEGER,
                        Base64Encoding BLOB NOT NULL,
                        Dataframe BLOB NOT NULL,
                        DocumentId INTEGER NOT NULL,
                        NumRows INTEGER NOT NULL,
                        NumColumns INTEGER NOT NULL,
                        BoundingBox TEXT,
                        PageNo Integer,
                        AltText TEXT,
                        CreatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        UpdatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        Status TEXT CHECK (Status IN ('NOT_STARTED', 'IN_PROGRESS', 'CHUNKING_COMPLETED', 'EMBEDDING_COMPLETED', 'PROCESSING_COMPLETED')),
                        FOREIGN KEY (DocumentID) REFERENCES Document(DocumentID)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Form (
                        FormId INTEGER PRIMARY KEY AUTOINCREMENT,
                        Base64Encoding BLOB NOT NULL,
                        FormText BLOB NOT NULL,
                        FormTitle TEXT,
                        DocumentId INTEGER NOT NULL,
                        PageNo Integer,
                        CreatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        UpdatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        Status TEXT CHECK (Status IN ('NOT_STARTED', 'IN_PROGRESS', 'CHUNKING_COMPLETED', 'EMBEDDING_COMPLETED', 'PROCESSING_COMPLETED')),
                        FOREIGN KEY (DocumentID) REFERENCES Document(DocumentID)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Metadata (
                        MetadataId INTEGER PRIMARY KEY,
                        DocumentId INTEGER NOT NULL,
                        Key TEXT NOT NULL,
                        Value TEXT NOT NULL,
                        CreatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        UpdatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        FOREIGN KEY (DocumentID) REFERENCES Document(DocumentID)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Derivative (
                        DerivativeId INTEGER PRIMARY KEY AUTOINCREMENT,
                        DocumentId INTEGER NOT NULL,
                        ChunkId INTEGER NOT NULL,
                        CompressedText BLOB NOT NULL,
                        CreatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        UpdatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        FOREIGN KEY (DocumentId) REFERENCES Document(DocumentId),
                        FOREIGN KEY (ChunkId) REFERENCES Chunk(ChunkId)
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_derivative_document ON Derivative(DocumentId)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_derivative_chunk ON Derivative(ChunkId)')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS QAPair (
                        QAPairId INTEGER PRIMARY KEY AUTOINCREMENT,
                        Question TEXT NOT NULL,
                        Answer TEXT NOT NULL,
                        CreatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime')),
                        UpdatedAt TIMESTAMP DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime'))
                    )
                ''')

                # Junction table: a QA pair may be sourced from multiple documents
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS QAPairDocument (
                        QAPairId INTEGER NOT NULL,
                        DocumentId INTEGER NOT NULL,
                        PRIMARY KEY (QAPairId, DocumentId),
                        FOREIGN KEY (QAPairId) REFERENCES QAPair(QAPairId) ON DELETE CASCADE,
                        FOREIGN KEY (DocumentId) REFERENCES Document(DocumentId)
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_qapairdocument_document ON QAPairDocument(DocumentId)')

                conn.commit()
                logger.info("Tables and indexes created successfully")

        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def is_file_processed(self, filename):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM Document WHERE FileName = ?
            ''', (filename,))
            return cursor.fetchone() is not None

    def insert_document(self, document: Dict[str, Any]) -> int:
        """
        Insert document
        """
        if not document:
            return -1
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO Document (FileName, DocumentType, PlainTextContent, Status)
                    VALUES (?, ?, ?, ?)
                ''', (
                    document['filename'],
                    document['document_type'],
                    document['document'].export_to_markdown(),
                    Status.NOT_STARTED.value
                ))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.Error as row_e:
                logger.error(f"Failed to insert document {document}: {row_e}")
        return -1

    def insert_chunks(self, chunks: List[Dict[str, Any]], status: Status = Status.NOT_STARTED) -> List[int]:
        """
        Insert chunks one by one, logger errors for failed rows.
        Inside transaction() a failed row raises instead, so the whole transaction rolls back.
        """
        if not chunks:
            return []
        inserted_ids = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for chunk in chunks:
                    try:
                        cursor.execute('''
                            INSERT INTO Chunk (DocumentId, ChunkText, FigureRefs, Embedding ,Status)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            chunk['DocumentId'],
                            chunk['ChunkText'],
                            chunk.get('FigureRefs', None),
                            chunk.get('Embedding', None),
                            status.value
                        ))
                        inserted_ids.append(cursor.lastrowid)
                    except sqlite3.Error as row_e:
                        logger.error(f"Failed to insert chunk {chunk}: {row_e}")
                        if self._in_transaction():
                            raise
        except sqlite3.Error as e:
            logger.error(f"Error inserting chunks: {e}")
            if self._in_transaction():
                raise
        return inserted_ids

    def insert_images(self, document_id: int, images: List[Dict[str, Any]]) -> List[int]:
        """
        Insert images one by one, logger errors for failed rows.
        """
        if not images:
            return []
        inserted_ids = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for img in images:
                    try:
                        cursor.execute('''
                            INSERT INTO Image (ImageRef, Base64Encoding, DocumentId, MimeType, Height, Width, Dpi, PageNo, AltText, Status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            img['ref'],
                            img['base64'],
                            document_id,
                            img['mime_type'],
                            img['height'],
                            img['width'],
                            img['dpi'],
                            img.get('page_no', None),
                            img.get('alt_text', None),
                            Status.NOT_STARTED.value
                        ))

                        conn.commit()
                        inserted_ids.append(cursor.lastrowid)
                    except sqlite3.Error as row_e:
                        logger.error(f"Failed to insert image {img}: {row_e}")
        except sqlite3.Error as e:
            logger.error(f"Error inserting images: {e}")
        return inserted_ids

    def insert_table_data(self, document_id: int, tables: List[Dict[str, Any]]) -> List[int]:
        """
        Insert table data one by one, logger errors for failed rows.
        """
        if not tables:
            return []
        inserted_ids = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for tbl in tables:
                    try:
                        cursor.execute('''
                            INSERT INTO TableData (TableRef, Base64Encoding, Dataframe, DocumentId, NumRows, NumColumns, BoundingBox, PageNo, AltText, Status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            tbl.get('ref', None),
                            tbl['base64'],
                            tbl['dataframe'],
                            document_id,
                            tbl['num_rows'],
                            tbl['num_cols'],
                            json.dumps(tbl.get('bbox', {})),
                            tbl.get('page_no', None),
                            tbl.get('alt_text', None),
                            Status.NOT_STARTED.value
                        ))

                        conn.commit()
                        inserted_ids.append(cursor.lastrowid)
                    except sqlite3.Error as row_e:
                        logger.error(f"Failed to insert table data {tbl}: {row_e}")
        except sqlite3.Error as e:
            logger.error(f"Error inserting table data: {e}")
        return inserted_ids
    
    def insert_form_data(self, document_id: int, forms: List[Dict[str, Any]]) -> List[int]:
        """
        Insert form data one by one, logger errors for failed rows.
        """
        if not forms:
            return []
        inserted_ids = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for frm in forms:
                    try:
                        cursor.execute('''
                            INSERT INTO Form (Base64Encoding, FormText, FormTitle, DocumentId, PageNo, Status)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            frm['page_base64'],
                            frm['form_text_full'],
                            frm.get('form_title', None),
                            document_id,
                            frm.get('page_no', None),
                            Status.NOT_STARTED.value
                        ))

                        conn.commit()
                        inserted_ids.append(cursor.lastrowid)
                    except sqlite3.Error as row_e:
                        logger.error(f"Failed to insert form data {frm}: {row_e}")
        except sqlite3.Error as e:
            logger.error(f"Error inserting form data: {e}")
        return inserted_ids

    def insert_metadata(self, metadata: List[Dict[str, Any]]) -> List[int]:
        """
        Insert metadata one by one, logger errors for failed rows.
        """
        if not metadata:
            return []
        inserted_ids = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for meta in metadata:
                    try:
                        cursor.execute('''
                            INSERT INTO Metadata (DocumentId, Key, Value)
                            VALUES (?, ?, ?)
                        ''', (
                            meta['DocumentId'],
                            meta['Key'],
                            meta['Value']
                        ))
                        conn.commit()
                        inserted_ids.append(cursor.lastrowid)
                    except sqlite3.Error as row_e:
                        logger.error(f"Failed to insert metadata {meta}: {row_e}")
        except sqlite3.Error as e:
            logger.error(f"Error inserting metadata: {e}")
        return inserted_ids

    def insert_derivatives(self, derivatives: List[DerivativeCreate]) -> List[int]:
        """
        Insert derivatives one by one (text is zlib-compressed), logger errors for failed rows.
        """
        if not derivatives:
            return []
        inserted_ids = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for der in derivatives:
                    try:
                        cursor.execute('''
                            INSERT INTO Derivative (DocumentId, ChunkId, CompressedText)
                            VALUES (?, ?, ?)
                        ''', (
                            der.DocumentId,
                            der.ChunkId,
                            zlib.compress(der.DerivativeText.encode('utf-8')),
                        ))
                        conn.commit()
                        inserted_ids.append(cursor.lastrowid)
                    except sqlite3.Error as row_e:
                        logger.error(f"Failed to insert derivative for document {der.DocumentId}, chunk {der.ChunkId}: {row_e}")
        except sqlite3.Error as e:
            logger.error(f"Error inserting derivatives: {e}")
        return inserted_ids

    def insert_qa_pairs(self, qa_pairs: List[QAPairCreate]) -> List[int]:
        """
        Insert QA pairs one by one, linking each to its source DocumentIds.
        """
        if not qa_pairs:
            return []
        inserted_ids = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for qa in qa_pairs:
                    try:
                        cursor.execute('''
                            INSERT INTO QAPair (Question, Answer)
                            VALUES (?, ?)
                        ''', (qa.Question, qa.Answer))
                        qa_pair_id = cursor.lastrowid
                        cursor.executemany('''
                            INSERT OR IGNORE INTO QAPairDocument (QAPairId, DocumentId)
                            VALUES (?, ?)
                        ''', [(qa_pair_id, doc_id) for doc_id in qa.DocumentIds])
                        conn.commit()
                        inserted_ids.append(qa_pair_id)
                    except sqlite3.Error as row_e:
                        conn.rollback()
                        logger.error(f"Failed to insert QA pair {qa}: {row_e}")
        except sqlite3.Error as e:
            logger.error(f"Error inserting QA pairs: {e}")
        return inserted_ids

    def get_derivatives_by_document_id(self, document_id: int) -> List[DerivativeModel]:
        """
        Retrieve all derivatives for a given document ID, decompressing the text.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DerivativeId, DocumentId, ChunkId, CompressedText, CreatedAt, UpdatedAt
                    FROM Derivative
                    WHERE DocumentId = ?
                ''', (document_id,))
                return [
                    DerivativeModel(
                        DerivativeId=row[0],
                        DocumentId=row[1],
                        ChunkId=row[2],
                        DerivativeText=zlib.decompress(row[3]).decode('utf-8'),
                        CreatedAt=row[4],
                        UpdatedAt=row[5],
                    )
                    for row in cursor.fetchall()
                ]
        except (sqlite3.Error, zlib.error) as e:
            logger.error(f"Error retrieving derivatives by document id: {e}")
            return []

    def get_derivatives(self, derivative_ids: List[int]) -> List[DerivativeModel]:
        """
        Retrieve derivatives by DerivativeId, decompressing the text. Missing ids are skipped.
        """
        if not derivative_ids:
            return []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(derivative_ids))
                cursor.execute(f'''
                    SELECT DerivativeId, DocumentId, ChunkId, CompressedText, CreatedAt, UpdatedAt
                    FROM Derivative
                    WHERE DerivativeId IN ({placeholders})
                ''', list(derivative_ids))
                return [
                    DerivativeModel(
                        DerivativeId=row[0],
                        DocumentId=row[1],
                        ChunkId=row[2],
                        DerivativeText=zlib.decompress(row[3]).decode('utf-8'),
                        CreatedAt=row[4],
                        UpdatedAt=row[5],
                    )
                    for row in cursor.fetchall()
                ]
        except (sqlite3.Error, zlib.error) as e:
            logger.error(f"Error retrieving derivatives by id: {e}")
            return []

    def get_random_derivatives(self, limit: int) -> List[DerivativeModel]:
        """
        Retrieve up to `limit` randomly selected derivatives, decompressing the text.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DerivativeId, DocumentId, ChunkId, CompressedText, CreatedAt, UpdatedAt
                    FROM Derivative
                    ORDER BY RANDOM()
                    LIMIT ?
                ''', (limit,))
                return [
                    DerivativeModel(
                        DerivativeId=row[0],
                        DocumentId=row[1],
                        ChunkId=row[2],
                        DerivativeText=zlib.decompress(row[3]).decode('utf-8'),
                        CreatedAt=row[4],
                        UpdatedAt=row[5],
                    )
                    for row in cursor.fetchall()
                ]
        except (sqlite3.Error, zlib.error) as e:
            logger.error(f"Error retrieving random derivatives: {e}")
            return []

    def get_qa_pairs(self, document_id: Optional[int] = None) -> List[QAPairModel]:
        """
        Retrieve QA pairs with their source DocumentIds. If document_id is given,
        only QA pairs linked to that document are returned (with all of their sources).
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT q.QAPairId, q.Question, q.Answer, q.CreatedAt, q.UpdatedAt,
                           GROUP_CONCAT(qd.DocumentId) AS DocumentIds
                    FROM QAPair q
                    LEFT JOIN QAPairDocument qd ON qd.QAPairId = q.QAPairId
                    WHERE ? IS NULL OR q.QAPairId IN (
                        SELECT QAPairId FROM QAPairDocument WHERE DocumentId = ?
                    )
                    GROUP BY q.QAPairId
                ''', (document_id, document_id))
                return [
                    QAPairModel(
                        QAPairId=row[0],
                        Question=row[1],
                        Answer=row[2],
                        CreatedAt=row[3],
                        UpdatedAt=row[4],
                        DocumentIds=[int(d) for d in row[5].split(',')] if row[5] else [],
                    )
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving QA pairs: {e}")
            return []

    def get_documents(self, status: Status = Status.NOT_STARTED) -> List[DocumentModel]:
        """
        Retrieve all documents with the given status.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM Document WHERE Status = ?",
                    (status.value,),
                )
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [DocumentModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def get_images_not_completed(self) -> List[ImageModel]:
        """
        Retrieve all images whose Status is not EMBEDDING_COMPLETED.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM Image WHERE Status != ?",
                    (Status.EMBEDDING_COMPLETED.value,),
                )
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [ImageModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving non-completed images: {e}")
            return []

    def get_tables_not_completed(self) -> List[TableDataModel]:
        """
        Retrieve all tables whose Status is not EMBEDDING_COMPLETED.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM TableData WHERE Status != ?",
                    (Status.EMBEDDING_COMPLETED.value,),
                )
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [TableDataModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving non-completed tables: {e}")
            return []

    def get_forms_not_completed(self) -> List[FormModel]:
        """
        Retrieve all forms whose Status is not EMBEDDING_COMPLETED.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM Form WHERE Status != ?",
                    (Status.EMBEDDING_COMPLETED.value,),
                )
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [FormModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving non-completed forms: {e}")
            return []

    def get_document_by_id(self, document_ids: List[int]) -> Dict[int, Optional[DocumentModel]]:
        """
        Retrieve documents by their IDs, keyed by document ID.
        """
        document_data: Dict[int, Optional[DocumentModel]] = {}
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for document_id in document_ids:
                    cursor.execute('SELECT * FROM Document WHERE DocumentId = ?', (document_id,))
                    row = cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        document_data[document_id] = DocumentModel(**dict(zip(columns, row)))
                    else:
                        document_data[document_id] = None
                return document_data
        except sqlite3.Error as e:
            logger.error(f"Error retrieving document by id: {e}")
        return document_data

    def get_chunks(self, status: Status = Status.NOT_STARTED) -> List[ChunkModel]:
        """
        Retrieve all chunks with the given status.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM Chunk WHERE STATUS = ?', (status.value,))
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [ChunkModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

    def get_chunks_by_document_id(self, document_id: int, status: Status = Status.NOT_STARTED) -> List[ChunkModel]:
        """
        Retrieve all chunks with the given status.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM Chunk WHERE STATUS=? AND DocumentId = ?', (status.value,document_id,))
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [ChunkModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []


    def get_unique_chunk_ids_from_derivatives(self, document_id: int) -> List[int]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT CHUNK_ID FROM Derivatives WHERE DocumentId=?',(document_id,))
                rows = cursor.fetchall()
                return [row[1] for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving unique chunks: {e}")
            return []

    def is_document_generated(self, document_id: int) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM Derivative WHERE DocumentId = ? LIMIT 1', (document_id,))
                rows = cursor.fetchall()
                if rows:
                    return True
        except sqlite3.Error as e:
            logger.error(f"Error retrieving documents: {e}")
            return False
        return False

    def get_images(self, status: Status = Status.NOT_STARTED) -> List[ImageModel]:
        """
        Retrieve all images with the given status.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM Image WHERE STATUS = ?', (status.value,))
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [ImageModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving images: {e}")
            return []

    def get_forms(self, status: Status = Status.NOT_STARTED) -> List[FormModel]:
        """
        Retrieve all forms with the given status.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM Form WHERE STATUS = ?', (status.value,))
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [FormModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving forms: {e}")
            return []
        
    def get_forms_by_document_id(self, document_ids: List[int]) -> List[FormModel]:
        """
        Retrieve all forms associated with a specific document ID.
        """
        try:
            with self.get_connection() as conn:
                forms = []
                for document_id in document_ids:
                    cursor = conn.cursor()
                    cursor.execute('SELECT * FROM Form WHERE DocumentId = ?', (document_id,))
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    forms.extend(([FormModel(**dict(zip(columns, row))) for row in rows]) if rows else [])
                return forms
        except sqlite3.Error as e:
            logger.error(f"Error retrieving forms: {e}")
            return []

    def get_tables(self, status: Status = Status.NOT_STARTED) -> List[TableDataModel]:
        """
        Retrieve all tables with the given status.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM TableData WHERE STATUS = ?', (status.value,))
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [TableDataModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving tables: {e}")
            return []

    def get_chunks_by_ids(self, chunk_ids: List[int], batch_size: int = 500) -> List[ChunkModel]:
        """
        Retrieve chunks by ID, querying in batches of ``batch_size`` to stay under
        SQLite's bound-parameter limit. Results follow the order of ``chunk_ids``
        (duplicates included); IDs with no matching chunk are skipped.

        Failures are contained: a failed batch or an invalid row is skipped, its
        IDs are written to the review queue (see _record_failed_ids), and the
        remaining batches still load.
        """
        unique_ids = list(dict.fromkeys(chunk_ids))
        found: Dict[int, ChunkModel] = {}
        # Per-batch errors are handled inside; this outer except only sees connection failures
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for start in range(0, len(unique_ids), batch_size):
                    batch = unique_ids[start:start + batch_size]
                    placeholders = ','.join('?' * len(batch))
                    try:
                        cursor.execute(f'SELECT * FROM Chunk WHERE ChunkId IN ({placeholders})', batch)
                        rows = cursor.fetchall()
                    except sqlite3.Error as e:
                        self._record_failed_ids("Chunk", batch, e)
                        continue
                    columns = [description[0] for description in cursor.description]
                    for row in rows:
                        record = dict(zip(columns, row))
                        try:
                            chunk = ChunkModel(**record)
                        except ValidationError as e:
                            self._record_failed_ids("Chunk", [record.get("ChunkId")], e)
                            continue
                        found[chunk.ChunkId] = chunk
        except sqlite3.Error as e:
            self._record_failed_ids("Chunk", unique_ids, e)
            return []
        return [found[cid] for cid in chunk_ids if cid in found]

    def _record_failed_ids(self, table: str, ids: List[int], error: Exception) -> None:
        """
        Log a load failure and append it to the review queue (paths.review_queue)
        as one JSON line, so a person can inspect and re-run those IDs later.
        Written to a file rather than the database, since the database may be
        the thing that is failing.
        """
        logger.error(f"Failed to load {len(ids)} {table} row(s) {ids}: {type(error).__name__}: {error}")
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "table": table,
            "ids": ids,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            review_path = get_config().get_config_path(get_config(), "paths", "review_queue")
            review_path.parent.mkdir(parents=True, exist_ok=True)
            with open(review_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except (OSError, KeyError, TypeError) as e:
            logger.error(f"Could not write to review queue; failure only logged above: {e}")


    def get_image_by_id(self, image_id: int) -> Optional[ImageModel]:
        """
        Retrieve a single image by its ID.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM Image WHERE ImageId = ?', (image_id,))
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return ImageModel(**dict(zip(columns, row)))
        except sqlite3.Error as e:
            logger.error(f"Error retrieving image by id: {e}")
        return None

    def get_table_by_id(self, table_id: int) -> Optional[TableDataModel]:
        """
        Retrieve a single table by its ID.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM TableData WHERE TableId = ?', (table_id,))
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return TableDataModel(**dict(zip(columns, row)))
        except sqlite3.Error as e:
            logger.error(f"Error retrieving table by id: {e}")
        return None

    def get_form_by_id(self, form_id: int) -> Optional[FormModel]:
        """
        Retrieve a single form by its ID.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM Form WHERE FormId = ?', (form_id,))
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return FormModel(**dict(zip(columns, row)))
        except sqlite3.Error as e:
            logger.error(f"Error retrieving form by id: {e}")
        return None

    def get_metadata(self) -> List[MetadataModel]:
        """
        Retrieve all metadata as a list of MetadataModel instances.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM Metadata')
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [MetadataModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving metadata: {e}")
            return []

    def get_metadata_by_document_id(self, document_id: int) -> List[MetadataModel]:
        """
        Retrieve all metadata for a given document ID.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM Metadata WHERE DocumentId = ?', (document_id,))
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [MetadataModel(**dict(zip(columns, row))) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving metadata by document id: {e}")
            return []
    
    def insert_embedding_by_chunk_id(self, chunk_ids: List[int], embeddings: List[Any]) -> List[int]:
        """
        Insert embedding by chunk Id
        """
        cids = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for cid, embedding in zip(chunk_ids, embeddings):
                    cursor.execute('Update Chunk '
                                    'SET Embedding = ?,'
                                    'Status = ? '
                                    'WHERE ChunkId = ?', (embedding, Status.EMBEDDING_COMPLETED.value, cid))
                    conn.commit()
                    cids.append(cid)
        except sqlite3.Error as e:
            logger.error(f"Error inserting embeddings by chunk id: {e}")
            return []
        return cids

    # TODO: Add additional update_chunk_status, update_form_status, update_table_data
    def update_document_status(self, document_ids: List[int], status: Status = Status.PROCESSING_COMPLETED) -> List[int]:
        """
        Updates the status of the specified document
        Inside transaction() an error raises instead, so the whole transaction rolls back.
        """
        try:
            with self.get_connection() as conn:
                updated_ids = []
                cursor = conn.cursor()
                for document_id in document_ids:
                    cursor.execute(
                        "UPDATE Document "
                        "SET Status = ?, UpdatedAt = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime') "
                        "WHERE DocumentId = ?",
                        (status.value, document_id),
                    )
                    updated_ids.append(document_id)
                return updated_ids
        except sqlite3.Error as e:
            logger.error(f"Error updating documents: {e}")
            if self._in_transaction():
                raise
            return []

    def complete_documents(self, document_ids: List[int]) -> List[int]:
        """
        Marks documents PROCESSING_COMPLETED once all of their chunks are PROCESSING_COMPLETED
        """
        completed_ids: List[int] = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for document_id in document_ids:
                    cursor.execute(
                        "UPDATE Document "
                        "SET Status = ?, UpdatedAt = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime') "
                        "WHERE DocumentId = ? AND NOT EXISTS "
                        "(SELECT 1 FROM Chunk WHERE DocumentId = ? AND Status != ?)",
                        (Status.PROCESSING_COMPLETED.value, document_id, document_id, Status.PROCESSING_COMPLETED.value),
                    )
                    if cursor.rowcount:
                        completed_ids.append(document_id)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error completing documents: {e}")
            return []
        return completed_ids

    def update_chunk_status(self, chunk_ids: List[int], status: Status = Status.CHUNKING_COMPLETED) -> List[int]:
        """
        Updates the status of the specified chunk
        """
        processed_ids: List[int] = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for id in chunk_ids:
                    cursor.execute(
                        "UPDATE Chunk "
                        "SET Status = ?, UpdatedAt = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime') "
                        "WHERE ChunkId = ?",
                        (status.value, id),
                    )
                    processed_ids.append(id)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
        return processed_ids

    def update_form_status(self, form_ids: List[int], status: Status = Status.EMBEDDING_COMPLETED) -> List[int]:
        """
        Updates the status of the specified forms (by FormId)
        """
        processed_ids: List[int] = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for id in form_ids:
                    cursor.execute(
                        "UPDATE Form "
                        "SET Status = ?, UpdatedAt = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime') "
                        "WHERE FormId = ?",
                        (status.value, id),
                    )
                    processed_ids.append(id)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating form status: {e}")
            return []
        return processed_ids

    def update_image_status(self, image_ids: List[int], status: Status = Status.EMBEDDING_COMPLETED) -> List[int]:
        """
        Updates the status of the specified images (by ImageId)
        """
        processed_ids: List[int] = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for id in image_ids:
                    cursor.execute(
                        "UPDATE Image "
                        "SET Status = ?, UpdatedAt = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime') "
                        "WHERE ImageId = ?",
                        (status.value, id),
                    )
                    processed_ids.append(id)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating image status: {e}")
            return []
        return processed_ids

    def update_table_status(self, table_ids: List[int], status: Status = Status.EMBEDDING_COMPLETED) -> List[int]:
        """
        Updates the status of the specified tables (by TableId in TableData)
        """
        processed_ids: List[int] = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for id in table_ids:
                    cursor.execute(
                        "UPDATE TableData "
                        "SET Status = ?, UpdatedAt = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'localtime') "
                        "WHERE TableId = ?",
                        (status.value, id),
                    )
                    processed_ids.append(id)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating table status: {e}")
            return []
        return processed_ids