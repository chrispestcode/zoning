import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='cuda_memory.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_cuda_memory(prefix, device=torch.device("cuda:0")):
    if not torch.cuda.is_available():
        logging.warning(f'{prefix}: CUDA is not available on this machine.')
        return

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    allocated_mb = allocated / (1024 ** 2)
    reserved_mb = reserved / (1024 ** 2)
    max_allocated_mb = max_allocated / (1024 ** 2)
    max_reserved_mb = max_reserved / (1024 ** 2)

    logging.info(f"{prefix}: [CUDA Memory Usage on {device}]")
    logging.info(f"  Allocated: {allocated_mb:.2f} MB - Memory currently used by tensors.")
    logging.info(f"  Reserved : {reserved_mb:.2f} MB - Memory reserved by PyTorch's caching allocator (may include unused memory).")
    logging.info(f"  Max Allocated: {max_allocated_mb:.2f} MB - Peak allocated memory since start or last reset.")
    logging.info(f"  Max Reserved : {max_reserved_mb:.2f} MB - Peak reserved memory since start or last reset.")
    logging.info("-" * 60)
