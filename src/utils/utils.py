import gzip

def compress_text(text: str) -> bytes:
    """Compresses text string into gzip bytes for database storage."""
    return gzip.compress(text.encode('utf-8'))

def decompress_text(compressed_bytes: bytes) -> str:
    """Decompresses gzip bytes from the database back into a string."""
    return gzip.decompress(compressed_bytes).decode('utf-8')