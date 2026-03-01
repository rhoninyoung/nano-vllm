import os
import torch
from functools import lru_cache


@lru_cache(1)
def detect_device_type() -> str:
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            return "rocm"
        return "cuda"
    return "cpu"


@lru_cache(1)
def get_device() -> torch.device:
    dt = detect_device_type()
    if dt in ("cuda", "rocm"):
        return torch.device("cuda")
    return torch.device("cpu")


def get_torch_device_str() -> str:
    dt = detect_device_type()
    return "cuda" if dt in ("cuda", "rocm") else "cpu"


def get_dist_backend() -> str:
    dt = detect_device_type()
    if dt in ("cuda", "rocm"):
        return "nccl"
    return "gloo"


def is_gpu() -> bool:
    return detect_device_type() in ("cuda", "rocm")


def is_rocm() -> bool:
    return detect_device_type() == "rocm"


def is_cpu() -> bool:
    return detect_device_type() == "cpu"


def supports_cuda_graphs() -> bool:
    dt = detect_device_type()
    if dt == "cpu":
        return False
    if dt == "rocm":
        return False
    return True


def device_synchronize():
    if is_gpu():
        torch.cuda.synchronize()


def empty_cache():
    if is_gpu():
        torch.cuda.empty_cache()


def reset_peak_memory_stats():
    if is_gpu():
        torch.cuda.reset_peak_memory_stats()


def set_device(rank: int):
    if is_gpu():
        torch.cuda.set_device(rank)


def get_memory_info() -> tuple[int, int]:
    """Returns (free_bytes, total_bytes)."""
    if is_gpu():
        return torch.cuda.mem_get_info()
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                mem[parts[0].rstrip(":")] = int(parts[1]) * 1024
    total = mem.get("MemTotal", 0)
    available = mem.get("MemAvailable", mem.get("MemFree", 0))
    return available, total


def get_memory_stats() -> dict:
    if is_gpu():
        return torch.cuda.memory_stats()
    return {"allocated_bytes.all.peak": 0, "allocated_bytes.all.current": 0}


def to_device(data, dtype, device=None):
    """Create tensor and move to target device efficiently."""
    if device is None:
        device = get_device()
    if device.type == "cpu":
        return torch.tensor(data, dtype=dtype)
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
