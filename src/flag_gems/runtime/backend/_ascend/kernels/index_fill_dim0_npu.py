import triton
import triton.language as tl


@triton.jit(debug=True)
def index_fill_dim0_row_kernel(
    out,
    index,
    value,
    row_count,
    row_width,
    index_len,
    HAS_NEGATIVE: tl.constexpr,
    USE_INT32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    column_id = tl.program_id(1)
    columns = column_id * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_id < index_len

    if USE_INT32:
        row_count_value = row_count.to(tl.int32)
        row_width_value = row_width.to(tl.int32)
        row = tl.load(index + row_id, mask=row_mask, other=0).to(tl.int32)
        if HAS_NEGATIVE:
            row = tl.where(row < 0, row + row_count_value, row)
        offsets = row * row_width_value + columns
    else:
        row_count_value = row_count.to(tl.int64)
        row_width_value = row_width.to(tl.int64)
        row = tl.load(index + row_id, mask=row_mask, other=0).to(tl.int64)
        if HAS_NEGATIVE:
            row = tl.where(row < 0, row + row_count_value, row)
        offsets = row * row_width_value + columns.to(tl.int64)

    mask = row_mask & (columns < row_width)
    tl.store(out + offsets, value, mask=mask)
