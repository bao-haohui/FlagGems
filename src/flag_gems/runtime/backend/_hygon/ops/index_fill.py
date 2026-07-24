import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_fill import _prepare_index, _prepare_tensor_value
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.jit(
    do_not_specialize=[
        "value",
        "index_len",
        "dim_size",
        "inner_size",
    ]
)
def index_fill_contiguous_kernel(
    out,
    index,
    value,
    index_len,
    dim_size,
    inner_size,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid_outer_index = tl.program_id(axis=0)
    pid_inner = tl.program_id(axis=1)

    outer_index = pid_outer_index // index_len
    index_offset = pid_outer_index - outer_index * index_len
    raw_index = tl.load(index + index_offset).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    tl.device_assert(valid_index, "index out of bounds")

    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)

    inner_offsets = pid_inner * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    mask = inner_offsets < inner_size
    out_offsets = (
        outer_index.to(tl.int64) * dim_size + normalized_index
    ) * inner_size + inner_offsets

    if VALUE_IS_TENSOR:
        fill_value = tl.load(value)
    else:
        fill_value = value
    tl.store(out + out_offsets, fill_value, mask=mask & valid_index)


def _native_clone(inp):
    return torch.ops.aten.clone.default.redispatch(_FALLBACK_KEYSET, inp)


def _native_copy_(out, inp):
    return torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, out, inp, False)


def _native_index_fill_scalar(inp, dim, index, value):
    return torch.ops.aten.index_fill.int_Scalar.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, value
    )


def _native_index_fill_scalar_(inp, dim, index, value):
    return torch.ops.aten.index_fill_.int_Scalar.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, value
    )


def _native_index_fill_tensor(inp, dim, index, value):
    return torch.ops.aten.index_fill.int_Tensor.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, value
    )


def _native_index_fill_tensor_(inp, dim, index, value):
    return torch.ops.aten.index_fill_.int_Tensor.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, value
    )


def _native_index_fill_scalar_out(inp, dim, index, value, out):
    return torch.ops.aten.index_fill.int_Scalar_out.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, value, out=out
    )


def _native_index_fill_tensor_out(inp, dim, index, value, out):
    return torch.ops.aten.index_fill.int_Tensor_out.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, value, out=out
    )


def _block_inner(inner_size):
    return min(1024, triton.next_power_of_2(max(1, inner_size)))


def _index_fill_contiguous_(out, dim, index, value, value_is_tensor):
    if out.numel() == 0 or index.numel() == 0:
        return out

    dim_size = out.size(dim)
    inner_size = 1
    for size in out.shape[dim + 1 :]:
        inner_size *= size
    outer_size = out.numel() // (dim_size * inner_size)
    block_inner = _block_inner(inner_size)
    grid = (
        outer_size * index.numel(),
        triton.cdiv(inner_size, block_inner),
    )

    with torch_device_fn.device(out.device):
        index_fill_contiguous_kernel[grid](
            out,
            index,
            value,
            index.numel(),
            dim_size,
            inner_size,
            VALUE_IS_TENSOR=value_is_tensor,
            BLOCK_INNER=block_inner,
        )
    return out


def _prepare_hcu_index(inp, dim, index):
    dim, index = _prepare_index(inp, dim, index)
    return dim, index.contiguous()


def index_fill_scalar(inp, dim, index, value):
    logger.debug("GEMS_HYGON INDEX_FILL SCALAR")
    dim, index = _prepare_hcu_index(inp, dim, index)
    if not inp.is_contiguous():
        return _native_index_fill_scalar(inp, dim, index, value)
    out = _native_clone(inp)
    return _index_fill_contiguous_(out, dim, index, value, False)


def index_fill_scalar_(inp, dim, index, value):
    logger.debug("GEMS_HYGON INDEX_FILL_ SCALAR")
    dim, index = _prepare_hcu_index(inp, dim, index)
    if not inp.is_contiguous():
        return _native_index_fill_scalar_(inp, dim, index, value)
    return _index_fill_contiguous_(inp, dim, index, value, False)


def index_fill_tensor(inp, dim, index, value):
    logger.debug("GEMS_HYGON INDEX_FILL TENSOR")
    dim, index = _prepare_hcu_index(inp, dim, index)
    value_is_tensor, value = _prepare_tensor_value(inp, value)
    if not inp.is_contiguous():
        if value_is_tensor:
            return _native_index_fill_tensor(inp, dim, index, value)
        return _native_index_fill_scalar(inp, dim, index, value)
    out = _native_clone(inp)
    return _index_fill_contiguous_(out, dim, index, value, value_is_tensor)


def index_fill_tensor_(inp, dim, index, value):
    logger.debug("GEMS_HYGON INDEX_FILL_ TENSOR")
    dim, index = _prepare_hcu_index(inp, dim, index)
    value_is_tensor, value = _prepare_tensor_value(inp, value)
    if not inp.is_contiguous():
        if value_is_tensor:
            return _native_index_fill_tensor_(inp, dim, index, value)
        return _native_index_fill_scalar_(inp, dim, index, value)
    return _index_fill_contiguous_(inp, dim, index, value, value_is_tensor)


def index_fill_scalar_out(inp, dim, index, value, *, out):
    logger.debug("GEMS_HYGON INDEX_FILL SCALAR_OUT")
    dim, index = _prepare_hcu_index(inp, dim, index)
    if not inp.is_contiguous() or not out.is_contiguous():
        return _native_index_fill_scalar_out(inp, dim, index, value, out)
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    _native_copy_(out, inp)
    return _index_fill_contiguous_(out, dim, index, value, False)


def index_fill_tensor_out(inp, dim, index, value, *, out):
    logger.debug("GEMS_HYGON INDEX_FILL TENSOR_OUT")
    dim, index = _prepare_hcu_index(inp, dim, index)
    value_is_tensor, value = _prepare_tensor_value(inp, value)
    if not inp.is_contiguous() or not out.is_contiguous():
        if value_is_tensor:
            return _native_index_fill_tensor_out(inp, dim, index, value, out)
        return _native_index_fill_scalar_out(inp, dim, index, value, out)
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    _native_copy_(out, inp)
    return _index_fill_contiguous_(out, dim, index, value, value_is_tensor)
