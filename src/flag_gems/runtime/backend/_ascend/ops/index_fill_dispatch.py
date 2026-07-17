"""Lightweight dispatch for the NPU index_fill C++ fast path."""

import importlib
import os
from pathlib import Path

import torch

_CPP_INDEX_FILL_FUNCTIONAL = None
_CPP_INDEX_FILL_INPLACE = None
_CPP_INDEX_FILL_LOOKED_UP = False
_CPP_INDEX_FILL_ENABLED = os.environ.get(
    "FLAG_GEMS_NPU_INDEX_FILL_CPP_LAUNCHER", "1"
).lower() not in ("", "0", "false", "off", "none", "disable", "disabled")
_CPP_INDEX_FILL_MAX_INDEX_LEN = 16
_CPP_INDEX_FILL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _get_cpp_index_fill_ops():
    global _CPP_INDEX_FILL_FUNCTIONAL
    global _CPP_INDEX_FILL_INPLACE, _CPP_INDEX_FILL_LOOKED_UP
    if _CPP_INDEX_FILL_LOOKED_UP:
        return _CPP_INDEX_FILL_FUNCTIONAL, _CPP_INDEX_FILL_INPLACE
    _CPP_INDEX_FILL_LOOKED_UP = True
    try:
        launcher_module = importlib.import_module("flag_gems._index_fill_npu_launcher")
        kernel_path = (
            Path(__file__).resolve().parent.parent
            / "kernels"
            / "index_fill_dim0_npu.py"
        )
        if not kernel_path.is_file():
            return None, None
        launcher_module.configure_kernel_path(str(kernel_path))
    except (ImportError, AttributeError):
        return None, None
    _CPP_INDEX_FILL_FUNCTIONAL = getattr(
        torch.ops.flag_gems, "_index_fill_dim0_npu", None
    )
    _CPP_INDEX_FILL_INPLACE = getattr(
        torch.ops.flag_gems, "_index_fill_dim0_npu_", None
    )
    return _CPP_INDEX_FILL_FUNCTIONAL, _CPP_INDEX_FILL_INPLACE


def can_use_cpp_index_fill_dim0(inp, dim, index, value):
    return (
        _CPP_INDEX_FILL_ENABLED
        and inp.ndim == 2
        and inp.is_contiguous()
        and dim == 0
        and inp.dtype in _CPP_INDEX_FILL_DTYPES
        and type(value) in (bool, int, float)
        and 0 < index.numel() <= _CPP_INDEX_FILL_MAX_INDEX_LEN
    )


def try_cpp_index_fill(inp, dim, index, value, *, inplace, fallback, launcher=None):
    if not can_use_cpp_index_fill_dim0(inp, dim, index, value):
        return fallback()
    if launcher is None:
        functional, inplace_op = _get_cpp_index_fill_ops()
        launcher = inplace_op if inplace else functional
    return fallback() if launcher is None else launcher(inp, index, value)
