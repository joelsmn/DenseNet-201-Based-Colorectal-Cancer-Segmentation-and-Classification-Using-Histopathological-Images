"""
utils/gpu_memory.py — GPU memory cleanup helpers.
"""

import gc
import torch


def configure_cudnn_safe_mode():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def free_gpu_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_gpu_memory(tag: str = ""):
    if not torch.cuda.is_available():
        print(f"  [GPU MEM{(' ' + tag) if tag else ''}] CUDA not available")
        return
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"  [GPU MEM{(' ' + tag) if tag else ''}] "
          f"allocated={alloc:.0f}MB  reserved={reserved:.0f}MB  "
          f"peak={peak:.0f}MB")