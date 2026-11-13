import gzip
import sys
from loguru import logger as _loguru_logger

def compress_text(text: str) -> bytes:
    """Compresses text string into gzip bytes for database storage."""
    return gzip.compress(text.encode('utf-8'))

def decompress_text(compressed_bytes: bytes) -> str:
    """Decompresses gzip bytes from the database back into a string."""
    return gzip.decompress(compressed_bytes).decode('utf-8')

def dynamic_formatter(record):
    if record["level"] in ["EVAL"]:
        return (
            "<blue>{time:YYYY-MM-DD HH:mm:ss.SSS}</blue> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>\n"
        )
    else:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>\n"
        )

def _initialize_logger():
    _loguru_logger.remove()

    try:
        _loguru_logger.level("EVAL", no=22, color="<bold><yellow>")
    except TypeError:
        pass  # Already registered if reloaded

    _loguru_logger.add(sys.stderr, format=dynamic_formatter, level="INFO")

    _loguru_logger.eval = lambda message, *args, **kwargs: _loguru_logger.log(
        "EVAL", message, *args, **kwargs
    )

    return _loguru_logger

logger = _initialize_logger()
