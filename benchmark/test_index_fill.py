import math

import pytest
import torch

from . import base, consts

INDEX_RATIOS = ("1/16", "1/2", "full")
INDEX_FILL_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
MIN_SELECTED_NUMEL = 16 * 1024
DIM0_BENCHMARK_CASES = {
    (4096, 256): ((8, "small"),),
    (4096, 4096): ((512, "general"),),
    (8192, 4096): ((4096, "fallback"),),
}


class IndexFillBenchmark(base.GenericBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS + ["gbps"]
    DEFAULT_SHAPES = [
        (65536,),
        (4096, 256),
        (4096, 4096),
    ]
    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        self.shapes = list(self.DEFAULT_SHAPES)
        if (
            base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            and not base.Config.query
        ):
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))

    def set_more_shapes(self):
        return [
            (8192, 4096),
            (200, 40999, 3),
        ]

    def _clone_inplace_args(self, args):
        if not self.is_inplace:
            return args
        return (args[0].clone(), *args[1:])

    def get_latency(self, op, *args, **kwargs):
        if base.Config.mode == consts.BenchMode.OPERATOR:
            # Keep lazy module loading out of the five-call adaptive estimate.
            for _ in range(5):
                op(*self._clone_inplace_args(args), **kwargs)
            base.torch_device_fn.synchronize()
        return super().get_latency(op, *self._clone_inplace_args(args), **kwargs)


def _generate_input(shape, dtype, device):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randint(-10, 10, shape, dtype=dtype, device=device)


def _dims_for_shape(shape):
    if len(shape) == 1:
        return (0,)
    if len(shape) == 2 and tuple(shape) in DIM0_BENCHMARK_CASES:
        return (1, 0)
    return (1,)


def _index_len(dim_size, ratio):
    if ratio == "1/16":
        return max(1, dim_size // 16)
    if ratio == "1/2":
        return max(1, dim_size // 2)
    if ratio == "full":
        return dim_size
    raise ValueError(f"Unknown index ratio: {ratio}")


def _make_index(dim_size, index_len, device):
    return torch.randperm(dim_size, device=device)[:index_len]


def _scalar_value(dtype):
    if dtype == torch.bool:
        return True
    if dtype.is_floating_point:
        return 3.14159
    return 3


def _base_inputs(shape, dtype, device):
    for dim in _dims_for_shape(shape):
        dim_size = shape[dim]
        dim0_cases = DIM0_BENCHMARK_CASES.get(tuple(shape), ()) if dim == 0 else ()
        if dim0_cases:
            index_lens = [index_len for index_len, _ in dim0_cases]
        else:
            index_lens = [_index_len(dim_size, ratio) for ratio in INDEX_RATIOS]

        for index_len in dict.fromkeys(index_lens):
            selected_numel = math.prod(shape) // dim_size * index_len
            if not dim0_cases and selected_numel < MIN_SELECTED_NUMEL:
                continue
            inp = _generate_input(shape, dtype, device)
            index = _make_index(dim_size, index_len, device)
            yield inp, dim, index


def _dim0_case_variant(shape, dim, index_len):
    if dim != 0:
        return None
    for case_index_len, variant in DIM0_BENCHMARK_CASES.get(tuple(shape), ()):
        if index_len == case_index_len:
            return variant
    return None


def _selected_numel(inp, dim, index):
    return math.prod(inp.shape) // inp.size(dim) * index.numel()


def _inplace_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    bytes_per_elem = inp.element_size()
    io_amount = index.numel() * index.element_size()
    io_amount += _selected_numel(inp, dim, index) * bytes_per_elem
    return io_amount * 1e-9 / (latency * 1e-3)


def _out_of_place_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    bytes_per_elem = inp.element_size()
    io_amount = inp.numel() * bytes_per_elem * 2
    io_amount += index.numel() * index.element_size()
    io_amount += _selected_numel(inp, dim, index) * bytes_per_elem
    return io_amount * 1e-9 / (latency * 1e-3)


def index_fill_input_fn(shape, dtype, device):
    for inp, dim, index in _base_inputs(shape, dtype, device):
        yield inp, dim, index, _scalar_value(dtype)


def _skip_unrepresentative_ascend_torch_baseline():
    if base.vendor_name == "ascend" and base.device == "npu":
        pytest.skip(
            "torch_npu index_fill extracts every NPU index element on the host; "
            "use test_index_fill_npu_reference.py for the direct ACLNN comparison"
        )


@pytest.mark.index_fill
def test_index_fill():
    _skip_unrepresentative_ascend_torch_baseline()
    bench = IndexFillBenchmark(
        op_name="index_fill",
        input_fn=index_fill_input_fn,
        torch_op=torch.index_fill,
        dtypes=INDEX_FILL_DTYPES,
        get_gbps=_out_of_place_gbps,
    )
    bench.run()


@pytest.mark.index_fill_
def test_index_fill_():
    _skip_unrepresentative_ascend_torch_baseline()
    bench = IndexFillBenchmark(
        op_name="index_fill_",
        input_fn=index_fill_input_fn,
        torch_op=torch.Tensor.index_fill_,
        dtypes=INDEX_FILL_DTYPES,
        is_inplace=True,
        get_gbps=_inplace_gbps,
    )
    bench.run()
