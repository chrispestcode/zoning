import gc
import torch


def get_device() -> str:
    """Pick the best available torch device: CUDA, then Apple Silicon (MPS), then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_device_memory():
    """Free cached accelerator memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
