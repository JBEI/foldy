import datetime
import gc
import logging
import os
import socket
import threading
import time

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_OK = True
except Exception as e:
    print(e)
    _GPU_OK = False


def _fmt(val):
    """Format numbers nicely, use -1 as sentinel for 'not available'."""
    return f"{val:.1f}" if isinstance(val, float) else str(val)


def _sample():
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
    rss = psutil.Process().memory_info().rss / 1_048_576  # MB
    swap = psutil.swap_memory().used / 1_048_576  # MB
    cpu = psutil.cpu_percent(interval=None)

    gpu_mem = gpu_util = tcore = tmem = -1
    if _GPU_OK:
        mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
        tcore = pynvml.nvmlDeviceGetTemperature(_GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
        tmem = pynvml.nvmlDeviceGetTemperature(
            _GPU_HANDLE, pynvml.NVML_TEMPERATURE_MEMORY  # type: ignore[reportAttributeAccessIssue] # pynvml typing incomplete
        )
        gpu_mem = mem.used / 1_048_576  # type: ignore[reportOperatorIssue] # pynvml mem.used type unclear                     # MB
        gpu_util = util.gpu

    memory_field_dict = {
        "timestamp": ts,
        "host": socket.gethostname(),
        "rss_mb": _fmt(rss),
        "swap_mb": _fmt(swap),
        "cpu_pct": _fmt(cpu),
        "gpu_mem_mb": _fmt(gpu_mem),
        "gpu_util_pct": _fmt(gpu_util),
        "gpu_temp_core_c": _fmt(tcore),
        "gpu_temp_hbm_c": _fmt(tmem),
    }
    return memory_field_dict


def log_memory_usage():
    memory_notes = "Memory usage: "
    try:
        import torch

        memory_notes += torch.cuda.memory_summary() + " "
    except Exception as e:
        memory_notes += f"Error getting torch memory summary: {e} "

    try:
        memory_fields = _sample()
        memory_notes += "; ".join([f"{k}: {v}" for k, v in memory_fields.items()])
    except Exception as e:
        memory_notes += f"Error getting memory fields: {e}"
    logging.info(memory_notes)


def clean_up_torch_memory():
    # Force garbage collection
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            torch.cuda.ipc_collect()  # Critical for multi-process environments
    except Exception as e:
        logging.error(f"Error cleaning up torch memory: {e}")
