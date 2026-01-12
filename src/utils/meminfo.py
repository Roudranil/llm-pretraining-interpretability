from __future__ import annotations

import gc
import inspect
import os
import sys
import time
import traceback
from typing import List, Optional, Union

import loguru
import psutil
import torch


def clean_mem(logger: Optional[loguru.Logger] = None) -> None:
    process = psutil.Process(os.getpid())

    # Measure RAM before cleanup
    ram_before = process.memory_info().rss / (1024**2)  # in MB

    # Measure GPU before cleanup
    if torch.cuda.is_available():
        gpu_alloc_before = torch.cuda.memory_allocated() / (1024**2)  # in MB
        gpu_reserved_before = torch.cuda.memory_reserved() / (1024**2)  # in MB
    else:
        gpu_alloc_before = gpu_reserved_before = 0

    # clean all traceback
    if hasattr(sys, "last_traceback"):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, "last_traceback")
    if hasattr(sys, "last_type"):
        delattr(sys, "last_type")
    if hasattr(sys, "last_value"):
        delattr(sys, "last_value")

    # clean all ipython history
    if "get_ipython" in globals():
        try:
            from IPython import get_ipython  # type: ignore

            ip = get_ipython()
            user_ns = ip.user_ns  # type: ignore
            ip.displayhook.flush()  # type: ignore
            pc = ip.displayhook.prompt_count + 1  # type: ignore
            for n in range(1, pc):
                user_ns.pop("_i" + repr(n), None)
            user_ns.update(dict(_i="", _ii="", _iii=""))
            hm = ip.history_manager  # type: ignore
            hm.input_hist_parsed[:] = [""] * pc  # type: ignore
            hm.input_hist_raw[:] = [""] * pc  # type: ignore
            hm._i = hm._ii = hm._iii = hm._i00 = ""  # type: ignore
        except Exception as e:
            message = "ipython mem could not be cleared"
            if logger:
                logger.warning(message)
            else:
                print(message)

    # do a garbage collection and flush cuda cache
    gc.collect()
    torch.cuda.empty_cache()

    # Give system a small moment to settle (helps RAM measurement be more accurate)
    time.sleep(0.1)

    # Measure RAM after cleanup
    ram_after = process.memory_info().rss / (1024**2)  # in MB

    # Measure GPU after cleanup
    if torch.cuda.is_available():
        gpu_alloc_after = torch.cuda.memory_allocated() / (1024**2)  # in MB
        gpu_reserved_after = torch.cuda.memory_reserved() / (1024**2)  # in MB
    else:
        gpu_alloc_after = gpu_reserved_after = 0

    # Report freed memory
    ram_freed_message = f"RAM freed: {ram_before - ram_after:.2f} MB ({ram_before:.2f} -> {ram_after:.2f})"
    if logger:
        logger.info(ram_freed_message)
    else:
        print(ram_freed_message)
    if torch.cuda.is_available():
        gpu_alloc_msg = f"GPU allocated freed: {gpu_alloc_before - gpu_alloc_after:.2f} MB ({gpu_alloc_before:.2f} -> {gpu_alloc_after:.2f})"
        gpu_reserv_msg = f"GPU reserved freed: {gpu_reserved_before - gpu_reserved_after:.2f} MB ({gpu_reserved_before:.2f} -> {gpu_reserved_after:.2f})"
        if logger:
            logger.info(gpu_alloc_msg)
            logger.info(gpu_reserv_msg)
        else:
            print(gpu_alloc_msg)
            print(gpu_reserv_msg)
    else:
        if logger:
            logger.warning("No GPU detected.")
        else:
            print("No GPU detected.")


def get_ram_usage():
    process = psutil.Process()
    return process.memory_info().rss  # bytes


def free_vars(
    vars_to_delete: List[Union[str, object]],
    namespace: Optional[dict] = None,
    try_gpu: bool = True,
    logger: Optional[loguru.Logger] = None,  # type: ignore
):
    """
    Deletes variables by name or reference, frees RAM and GPU (PyTorch) memory,
    logs actions via logger if provided.

    Args:
      vars_to_delete: list of variable names (str) or object refs
      namespace: dict to remove names from (defaults to caller's globals())
      try_gpu: clear GPU memory for torch objects
      logger: logging object or None (use print)
    Returns:
      (freed_ram_bytes, freed_gpu_bytes)
    """
    # Setup logger if not provided
    if logger is None:

        def logger(msg):
            print(msg)

    else:
        logger = logger.info  # type: ignore

    # Automatic namespace resolution
    if namespace is None:
        # Get frame of the caller, locals then globals
        frame = inspect.currentframe().f_back  # type: ignore
        namespace = frame.f_globals  # type: ignore

    before_ram = get_ram_usage()
    try:
        import torch
    except ImportError:
        torch = None

    freed_gpu_bytes = 0
    torch_objs = []
    deleted = []

    for var in vars_to_delete:
        if isinstance(var, str):
            obj = namespace.get(var, None)
            if obj is not None:
                deleted.append(var)
                if torch and try_gpu:
                    torch_objs.append(obj)
                del namespace[var]
                logger(f"Deleted variable '{var}'")
            else:
                logger(f"Variable '{var}' not found in namespace")
        else:
            # Try to remove all names referencing the object
            names = [n for n, v in namespace.items() if v is var]
            for n in names:
                del namespace[n]
                deleted.append(n)
                logger(f"Deleted variable '{n}' (by reference)")
            if not names:
                logger(
                    f"Could not find a variable name for object {var!r}, may not be deleted"
                )
            if torch and try_gpu:
                torch_objs.append(var)

    if torch and try_gpu and torch_objs and torch.cuda.is_available():
        before_gpu = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_gpu = torch.cuda.memory_allocated()
        freed_gpu_bytes = after_gpu - before_gpu
        logger(f"GPU memory freed: {freed_gpu_bytes / (1024**2):.2f} MB")
    # Always run gc
    gc.collect()
    after_ram = get_ram_usage()
    freed_ram_bytes = after_ram - before_ram
    logger(f"RAM memory freed: {freed_ram_bytes / (1024**2):.2f} MB")
    clean_mem()
    # return freed_ram_bytes, freed_gpu_bytes
    # return freed_ram_bytes, freed_gpu_bytes
