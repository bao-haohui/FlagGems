import importlib
import os
import subprocess
import sys

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE


INDEX_FILL_SHAPES = (
    [(2, 32)] if QUICK_MODE else [(1, 2), (4, 8), (2, 3, 5)]
)
DIM_LIST = [1] if QUICK_MODE else [0, -1]
INDEX_CASES = ["normal", "negative", "scalar"]
INDEX_FILL_DTYPES = utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
INDEX_FILL_OPS = [
    "index_fill_scalar",
    "index_fill_scalar_",
    "index_fill_scalar_out",
    "index_fill_tensor",
    "index_fill_tensor_",
    "index_fill_tensor_out",
]
INDEX_FILL_OOB_PATHS = (
    ("cpp", "python_contiguous", "python_strided")
    if flag_gems.device == "cuda"
    else ("python_contiguous", "python_strided")
)


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-10, 10, shape, dtype=dtype, device=flag_gems.device)


def _scalar_value(dtype):
    if dtype == torch.bool:
        return True
    if dtype.is_floating_point:
        return -3.5
    return -3


def _make_index(dim_size, case):
    if case == "normal":
        values = [0, dim_size - 1] if dim_size > 1 else [0]
        return torch.tensor(values, dtype=torch.long, device=flag_gems.device)
    if case == "negative":
        return torch.tensor([-1], dtype=torch.long, device=flag_gems.device)
    if case == "scalar":
        return torch.tensor(0, dtype=torch.long, device=flag_gems.device)
    raise ValueError(f"Unknown index case: {case}")


def _to_ref_value(value):
    if isinstance(value, torch.Tensor):
        return utils.to_reference(value, False)
    return value


@pytest.mark.index_fill
@pytest.mark.parametrize("shape", INDEX_FILL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
@pytest.mark.parametrize("index_case", INDEX_CASES)
def test_index_fill_scalar(shape, dim, dtype, index_case):
    inp = _make_input(shape, dtype)
    dim = dim % inp.ndim
    index = _make_index(inp.size(dim), index_case)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(dim, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(dim, index, value)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out is not inp


@pytest.mark.index_fill_
@pytest.mark.parametrize("shape", INDEX_FILL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
@pytest.mark.parametrize("index_case", INDEX_CASES)
def test_index_fill_scalar_(shape, dim, dtype, index_case):
    inp = _make_input(shape, dtype)
    dim = dim % inp.ndim
    index = _make_index(inp.size(dim), index_case)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_inp.index_fill_(dim, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill_(dim, index, value)

    assert res_out is inp
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
@pytest.mark.parametrize("value_device", ["device", "cpu"])
def test_index_fill_tensor_value(dtype, value_device):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([1, -1], dtype=torch.long, device=flag_gems.device)
    value = torch.tensor(
        _scalar_value(dtype),
        dtype=dtype,
        device=flag_gems.device if value_device == "device" else "cpu",
    )

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_value = _to_ref_value(value)
    ref_out = ref_inp.index_fill(1, ref_index, ref_value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(1, index, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.index_fill_
def test_index_fill_duplicate_index():
    inp = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    index = torch.tensor([1, 1, -1], dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_inp.index_fill_(1, ref_index, -7.0)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        inp.index_fill_(1, index, -7.0)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill_
def test_index_fill_empty_index_noop():
    inp = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    index = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_inp.index_fill_(1, ref_index, -7.0)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        inp.index_fill_(1, index, -7.0)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill_
def test_index_fill_noncontiguous_view():
    base = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    ref_base = utils.to_reference(base.clone(), False)
    res_base = base.clone()
    ref_view = ref_base.t()
    res_view = res_base.t()
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    ref_index = utils.to_reference(index, False)

    ref_view.index_fill_(1, ref_index, -8.0)
    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res = res_view.index_fill_(1, index, -8.0)

    assert res is res_view
    utils.gems_assert_equal(res_base, ref_base)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.parametrize("value_is_tensor", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_fill_noncontiguous_index(value_is_tensor, inplace):
    inp = _make_input((3, 16), torch.float32)
    index_storage = torch.arange(16, dtype=torch.long, device=flag_gems.device)
    index = index_storage[::2]
    value = (
        torch.tensor(-7.0, dtype=inp.dtype, device=flag_gems.device)
        if value_is_tensor
        else -7.0
    )

    assert not index.is_contiguous()
    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_value = _to_ref_value(value)

    if inplace:
        ref_inp.index_fill_(1, ref_index, ref_value)
        with flag_gems.use_gems(include=INDEX_FILL_OPS):
            result = inp.index_fill_(1, index, value)
        assert result is inp
        utils.gems_assert_equal(inp, ref_inp)
    else:
        expected = ref_inp.index_fill(1, ref_index, ref_value)
        with flag_gems.use_gems(include=INDEX_FILL_OPS):
            result = inp.index_fill(1, index, value)
        assert result is not inp
        utils.gems_assert_equal(result, expected)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device != "cuda", reason="C++ launcher modes are CUDA-only"
)
@pytest.mark.parametrize("launcher_enabled", [False, True])
def test_index_fill_noncontiguous_index_cuda_dispatch_paths(launcher_enabled):
    if launcher_enabled:
        from flag_gems.config import c_operators

        if c_operators is None or not hasattr(
            c_operators, "IndexFillAtenRegistration"
        ):
            pytest.skip("FlagGems was built without the CUDA C++ launcher")

    child_code = f"""
import importlib
import torch
import flag_gems

ops = {INDEX_FILL_OPS!r}
index_fill_module = importlib.import_module("flag_gems.ops.index_fill")
original_impl = index_fill_module._index_fill_impl
triton_calls = 0

def counted_impl(*args, **kwargs):
    global triton_calls
    triton_calls += 1
    return original_impl(*args, **kwargs)

index_fill_module._index_fill_impl = counted_impl
probe_inp = torch.arange(48, dtype=torch.float32, device=flag_gems.device).reshape(3, 16)
probe_index = torch.arange(16, dtype=torch.long, device=flag_gems.device)[::2]
plan = index_fill_module._select_cpp_index_fill_plan(
    probe_inp,
    1,
    probe_index,
    -7.0,
    mode=index_fill_module.INDEX_FILL_FUNCTIONAL,
    prepared=False,
)
expected_implementation = (
    index_fill_module.INDEX_FILL_CUDA_CPP
    if {launcher_enabled!r}
    else index_fill_module.INDEX_FILL_TRITON
)
assert plan == index_fill_module.IndexFillPlan(
    backend="cuda",
    implementation=expected_implementation,
    mode=index_fill_module.INDEX_FILL_FUNCTIONAL,
    validation="device",
)

for value_is_tensor in (False, True):
    for inplace in (False, True):
        inp = torch.arange(48, dtype=torch.float32, device=flag_gems.device).reshape(3, 16)
        index_storage = torch.arange(16, dtype=torch.long, device=flag_gems.device)
        index = index_storage[::2]
        value = torch.tensor(-7.0, device=flag_gems.device) if value_is_tensor else -7.0
        expected = inp.index_fill(1, index, value)

        with flag_gems.use_gems(include=ops):
            if inplace:
                actual = inp.clone()
                result = actual.index_fill_(1, index, value)
                assert result is actual
            else:
                actual = inp.index_fill(1, index, value)
        torch.testing.assert_close(actual, expected)

expected_triton_calls = {2 if launcher_enabled else 4}
assert triton_calls == expected_triton_calls, (triton_calls, expected_triton_calls)
"""
    env = os.environ.copy()
    env["FLAG_GEMS_INDEX_FILL_CPP_LAUNCHER"] = "1" if launcher_enabled else "0"
    result = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_fill_contiguous_inner3_fast_path(dtype):
    inp = _make_input((4, 17, 3), dtype)
    index = torch.tensor([0, 1, 8, -1], dtype=torch.long, device=flag_gems.device)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(1, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(1, index, value)
        inplace = inp.clone()
        res_inplace = inplace.index_fill_(1, index, value)

    assert res_out is not inp
    assert res_inplace is inplace
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inplace, ref_out)


@pytest.mark.index_fill
@pytest.mark.parametrize("value_is_tensor", [False, True])
def test_index_fill_large_contiguous_membership_functional(value_is_tensor):
    inp = _make_input((1024, 1024), torch.float16)
    index = torch.arange(16, dtype=torch.long, device=flag_gems.device)
    value = (
        torch.tensor(-3.5, dtype=inp.dtype, device=flag_gems.device)
        if value_is_tensor
        else -3.5
    )

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_value = _to_ref_value(value)
    ref_out = ref_inp.index_fill(1, ref_index, ref_value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(1, index, value)

    assert actual is not inp
    utils.gems_assert_equal(actual, ref_out)


@pytest.mark.index_fill
@pytest.mark.index_fill_
def test_index_fill_large_contiguous_membership_duplicate_index():
    inp = _make_input((1024, 1024), torch.float16)
    base_index = torch.arange(127, dtype=torch.long, device=flag_gems.device)
    index = torch.cat(
        (
            base_index,
            base_index,
            torch.tensor([-1], dtype=torch.long, device=flag_gems.device),
        )
    )
    value = -3.5

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(1, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(1, index, value)
        inplace = inp.clone()
        inplace.index_fill_(1, index, value)

    utils.gems_assert_equal(actual, ref_out)
    utils.gems_assert_equal(inplace, ref_out)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device != "npu", reason="Ascend C fast path is NPU-only"
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_index_fill_ascendc_fast_path(monkeypatch, dtype):
    index_fill_module = importlib.import_module(
        flag_gems.index_fill_scalar.__module__
    )
    functional = index_fill_module._get_ascendc_index_fill_scalar()
    inplace_function = index_fill_module._get_ascendc_index_fill_scalar_inplace()
    if functional is None or inplace_function is None:
        pytest.skip("FlagGems was built without the Ascend C extension")

    calls = {"functional": 0, "inplace": 0}

    def counted_functional(*args):
        calls["functional"] += 1
        return functional(*args)

    def counted_inplace(*args):
        calls["inplace"] += 1
        return inplace_function(*args)

    monkeypatch.setattr(
        index_fill_module, "_ASCENDC_INDEX_FILL_SCALAR", counted_functional
    )
    monkeypatch.setattr(
        index_fill_module,
        "_ASCENDC_INDEX_FILL_SCALAR_INPLACE",
        counted_inplace,
    )

    def unexpected_triton(*args, **kwargs):
        pytest.fail("AscendC-supported index_fill fell back to Triton")

    monkeypatch.setattr(
        index_fill_module, "_index_fill_functional", unexpected_triton
    )
    monkeypatch.setattr(index_fill_module, "_index_fill_impl", unexpected_triton)

    inp = _make_input((4096, 4096), dtype)
    index = torch.cat(
        (
            torch.arange(253, dtype=torch.long, device=flag_gems.device),
            torch.tensor([1, -1, -1], dtype=torch.long, device=flag_gems.device),
        )
    )
    value = -3.5
    functional_plan = index_fill_module._select_ascendc_index_fill_plan(
        inp,
        1,
        index,
        value,
        mode=index_fill_module.INDEX_FILL_FUNCTIONAL,
    )
    inplace_plan = index_fill_module._select_ascendc_index_fill_plan(
        inp,
        1,
        index,
        value,
        mode=index_fill_module.INDEX_FILL_INPLACE,
    )
    assert functional_plan == index_fill_module.IndexFillPlan(
        backend="npu",
        implementation=index_fill_module.INDEX_FILL_ASCENDC,
        mode=index_fill_module.INDEX_FILL_FUNCTIONAL,
        validation="device",
    )
    assert inplace_plan == index_fill_module.IndexFillPlan(
        backend="npu",
        implementation=index_fill_module.INDEX_FILL_ASCENDC,
        mode=index_fill_module.INDEX_FILL_INPLACE,
        validation="device",
    )
    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    expected = ref_inp.index_fill(1, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(1, index, value)
        inplace = inp.clone()
        result = inplace.index_fill_(1, index, value)

    assert calls == {"functional": 1, "inplace": 1}
    assert result is inplace
    utils.gems_assert_equal(actual, expected)
    utils.gems_assert_equal(inplace, expected)

    invalid_index = index.clone()
    invalid_index[0] = inp.size(1)
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        IndexError, match="index out of range in self"
    ):
        inp.index_fill(1, invalid_index, value)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device != "npu", reason="Ascend C fast path is NPU-only"
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("index_case", ["tail", "mixed", "dense"])
@pytest.mark.parametrize("shape", [(4, 257), (8, 513), (17, 4095)])
def test_index_fill_ascendc_bf16_tail(
    monkeypatch, shape, index_case, inplace
):
    index_fill_module = importlib.import_module(
        flag_gems.index_fill_scalar.__module__
    )
    ascendc_function = (
        index_fill_module._get_ascendc_index_fill_scalar_inplace()
        if inplace
        else index_fill_module._get_ascendc_index_fill_scalar()
    )
    if ascendc_function is None:
        pytest.skip("FlagGems was built without the Ascend C extension")

    calls = 0

    def counted_ascendc(*args):
        nonlocal calls
        calls += 1
        return ascendc_function(*args)

    ascendc_name = (
        "_ASCENDC_INDEX_FILL_SCALAR_INPLACE"
        if inplace
        else "_ASCENDC_INDEX_FILL_SCALAR"
    )
    monkeypatch.setattr(index_fill_module, ascendc_name, counted_ascendc)

    inp = _make_input(shape, torch.bfloat16)
    if index_case == "tail":
        index = torch.full(
            (8,), -1, dtype=torch.long, device=flag_gems.device
        )
    elif index_case == "mixed":
        index = torch.tensor(
            [0, 1, 255, -1, 0, 1, 255, -1],
            dtype=torch.long,
            device=flag_gems.device,
        )
    else:
        index_len = min(((shape[1] + 7) // 8) * 8, 4096)
        index = torch.arange(
            index_len, dtype=torch.long, device=flag_gems.device
        ) % shape[1]
        index[-1] = -1

    value = -3.5
    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)

    if inplace:
        ref_inp.index_fill_(1, ref_index, value)
        with flag_gems.use_gems(include=INDEX_FILL_OPS):
            result = inp.index_fill_(1, index, value)
        assert result is inp
        actual = inp
        expected = ref_inp
    else:
        expected = ref_inp.index_fill(1, ref_index, value)
        with flag_gems.use_gems(include=INDEX_FILL_OPS):
            actual = inp.index_fill(1, index, value)

    assert calls == 1
    utils.gems_assert_equal(actual, expected)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device != "npu", reason="Ascend C fast path is NPU-only"
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "tensor_value",
        "noncontiguous_input",
        "unsupported_dtype",
        "unsupported_shape",
        "unsupported_dim",
    ],
)
def test_index_fill_ascendc_fallback_paths(monkeypatch, case, inplace):
    index_fill_module = importlib.import_module(
        flag_gems.index_fill_scalar.__module__
    )
    ascendc_calls = 0

    def unexpected_ascendc(*args, **kwargs):
        nonlocal ascendc_calls
        ascendc_calls += 1
        pytest.fail(f"{case} unexpectedly selected the AscendC fast path")

    monkeypatch.setattr(
        index_fill_module, "_ASCENDC_INDEX_FILL_LOOKED_UP", True
    )
    monkeypatch.setattr(
        index_fill_module, "_ASCENDC_INDEX_FILL_SCALAR", unexpected_ascendc
    )
    monkeypatch.setattr(
        index_fill_module,
        "_ASCENDC_INDEX_FILL_SCALAR_INPLACE",
        unexpected_ascendc,
    )

    if case == "noncontiguous_input":
        inp = _make_input((16, 3), torch.float16).t()
    elif case == "unsupported_dtype":
        inp = _make_input((3, 16), torch.int32)
    elif case == "unsupported_shape":
        inp = _make_input((2, 4104), torch.float16)
    elif case == "unsupported_dim":
        inp = _make_input((2, 3, 16), torch.float16)
    else:
        inp = _make_input((3, 16), torch.float16)

    dim = 2 if case == "unsupported_dim" else 1
    index = torch.arange(8, dtype=torch.long, device=flag_gems.device)
    value = (
        torch.tensor(-3.5, dtype=inp.dtype, device=flag_gems.device)
        if case == "tensor_value"
        else _scalar_value(inp.dtype)
    )
    mode = (
        index_fill_module.INDEX_FILL_INPLACE
        if inplace
        else index_fill_module.INDEX_FILL_FUNCTIONAL
    )
    plan = index_fill_module._select_ascendc_index_fill_plan(
        inp, dim, index, value, mode=mode
    )
    assert plan == index_fill_module.IndexFillPlan(
        backend="npu",
        implementation=index_fill_module.INDEX_FILL_TRITON,
        mode=mode,
        validation="device",
    )
    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_value = _to_ref_value(value)

    generic_path = "_index_fill_impl" if inplace else "_index_fill_functional"
    original_generic = getattr(index_fill_module, generic_path)
    generic_calls = 0

    def counted_generic(*args, **kwargs):
        nonlocal generic_calls
        generic_calls += 1
        return original_generic(*args, **kwargs)

    monkeypatch.setattr(index_fill_module, generic_path, counted_generic)

    if inplace:
        ref_inp.index_fill_(dim, ref_index, ref_value)
        with flag_gems.use_gems(include=INDEX_FILL_OPS):
            result = inp.index_fill_(dim, index, value)
        assert result is inp
        actual = inp
        expected = ref_inp
    else:
        expected = ref_inp.index_fill(dim, ref_index, ref_value)
        with flag_gems.use_gems(include=INDEX_FILL_OPS):
            actual = inp.index_fill(dim, index, value)

    assert ascendc_calls == 0
    assert generic_calls == 1
    utils.gems_assert_equal(actual, expected)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device != "npu", reason="Ascend C fast path is NPU-only"
)
@pytest.mark.parametrize(
    ("shape", "dim", "index_len", "dtype"),
    [
        ((3, 17), 1, 8, torch.float16),
        ((4, 64), 1, 4096, torch.float16),
        ((4, 257), 1, 8, torch.bfloat16),
        ((1, 256), 1, 8, torch.float32),
        ((4096, 256), 1, 16, torch.float16),
        ((1024, 1024), 1, 128, torch.float16),
        ((1024, 4096), 1, 2048, torch.float16),
        ((17, 31), 0, 8, torch.float16),
        ((257, 4099), 0, 256, torch.bfloat16),
        ((4, 8193), 0, 4096, torch.float32),
    ],
)
def test_index_fill_ascendc_generalized_shape_index(
    monkeypatch, shape, dim, index_len, dtype
):
    index_fill_module = importlib.import_module(
        flag_gems.index_fill_scalar.__module__
    )
    functional = index_fill_module._get_ascendc_index_fill_scalar()
    inplace_function = index_fill_module._get_ascendc_index_fill_scalar_inplace()
    if functional is None or inplace_function is None:
        pytest.skip("FlagGems was built without the Ascend C extension")

    calls = {"functional": 0, "inplace": 0}

    def counted_functional(*args):
        calls["functional"] += 1
        return functional(*args)

    def counted_inplace(*args):
        calls["inplace"] += 1
        return inplace_function(*args)

    monkeypatch.setattr(
        index_fill_module, "_ASCENDC_INDEX_FILL_SCALAR", counted_functional
    )
    monkeypatch.setattr(
        index_fill_module,
        "_ASCENDC_INDEX_FILL_SCALAR_INPLACE",
        counted_inplace,
    )

    inp = _make_input(shape, dtype)
    index = torch.cat(
        (
            torch.arange(
                index_len - 3, dtype=torch.long, device=flag_gems.device
            )
            % shape[dim],
            torch.tensor([1, -1, -1], dtype=torch.long, device=flag_gems.device),
        )
    )
    value = -3.5
    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    expected = ref_inp.index_fill(dim, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(dim, index, value)
        inplace = inp.clone()
        result = inplace.index_fill_(dim, index, value)

    assert calls == {"functional": 1, "inplace": 1}
    assert result is inplace
    utils.gems_assert_equal(actual, expected)
    utils.gems_assert_equal(inplace, expected)


@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device != "npu", reason="Ascend C fast path is NPU-only"
)
@pytest.mark.parametrize(
    ("shape", "index_len", "dtype"),
    [
        ((17, 4099), 8, torch.float16),
        ((17, 4099), 8, torch.bfloat16),
        ((17, 4099), 8, torch.float32),
        ((2, 1), 8, torch.float16),
        ((2, 1), 8, torch.bfloat16),
        ((2, 1), 8, torch.float32),
        ((257, 4099), 256, torch.bfloat16),
    ],
)
def test_index_fill_ascendc_dim0_inplace_small(shape, index_len, dtype):
    inp = _make_input(shape, dtype)
    index = torch.arange(
        index_len, dtype=torch.long, device=flag_gems.device
    ) % shape[0]
    if index_len == 8:
        index[-4:] -= shape[0]
    value = -3.5
    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    expected = ref_inp.index_fill(0, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.clone()
        result = actual.index_fill_(0, index, value)

    assert result is actual
    utils.gems_assert_equal(actual, expected)


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu", reason="Ascend C fast path is NPU-only"
)
@pytest.mark.parametrize(
    ("shape", "index_len", "dtype"),
    [
        ((2, 1), 8, torch.float16),
        ((2, 1), 8, torch.bfloat16),
        ((2, 1), 8, torch.float32),
        ((17, 4099), 8, torch.float16),
        ((17, 4099), 8, torch.bfloat16),
        ((17, 4099), 8, torch.float32),
        ((17, 4099), 16, torch.float16),
        ((257, 4099), 256, torch.float16),
        ((257, 4099), 256, torch.bfloat16),
        ((257, 4099), 256, torch.float32),
    ],
)
def test_index_fill_ascendc_dim0_functional_small(shape, index_len, dtype):
    inp = _make_input(shape, dtype)
    original = inp.clone()
    index = torch.arange(
        index_len, dtype=torch.long, device=flag_gems.device
    ) % shape[0]
    if index_len <= shape[0]:
        index[index_len // 2 :] -= shape[0]
    value = -3.5
    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    expected = ref_inp.index_fill(0, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(0, index, value)

    assert actual.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(inp, original)
    utils.gems_assert_equal(actual, expected)


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu", reason="Ascend C fast path is NPU-only"
)
@pytest.mark.parametrize(
    ("shape", "dim", "index_len", "expected"),
    [
        ((1, 1), 0, 8, True),
        ((1, 1), 1, 8, True),
        ((1, 4097), 0, 8, True),
        ((4097, 1), 0, 8, False),
        ((1, 4097), 1, 8, False),
        ((4, 4), 1, 7, False),
        ((4, 4), 1, 10, False),
        ((4, 4), 1, 4104, False),
    ],
)
def test_index_fill_ascendc_dispatch_boundary(
    monkeypatch, shape, dim, index_len, expected
):
    index_fill_module = importlib.import_module(
        flag_gems.index_fill_scalar.__module__
    )
    monkeypatch.setattr(index_fill_module, "_is_ascend_910b", lambda: True)
    inp = torch.empty(shape, dtype=torch.float16, device=flag_gems.device)
    index = torch.zeros(index_len, dtype=torch.long, device=flag_gems.device)

    assert (
        index_fill_module._should_use_ascendc_index_fill(inp, dim, index)
        is expected
    )


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu", reason="Ascend C fast path is NPU-only"
)
def test_index_fill_ascendc_capabilities_match_python_plan(monkeypatch):
    from flag_gems.config import c_operators

    if (
        c_operators is None
        or not hasattr(c_operators, "index_fill_ascendc_capabilities")
        or not hasattr(c_operators, "index_fill_ascendc_debug_path")
    ):
        pytest.skip("FlagGems was built without the Ascend C extension")

    index_fill_module = importlib.import_module(
        flag_gems.index_fill_scalar.__module__
    )
    monkeypatch.setattr(index_fill_module, "_is_ascend_910b", lambda: True)
    capabilities = c_operators.index_fill_ascendc_capabilities()

    assert capabilities == {
        "device_type": "npu",
        "supported_dtypes": ["float16", "bfloat16", "float32"],
        "supported_dims": [0, 1],
        "max_dim_size": 4096,
        "min_index_len": 8,
        "max_index_len": 4096,
        "index_alignment": 8,
        "index_dtype": "int64",
        "requires_contiguous": True,
    }

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    index = torch.arange(
        capabilities["min_index_len"],
        dtype=torch.long,
        device=flag_gems.device,
    )
    for dtype_name in capabilities["supported_dtypes"]:
        dtype = dtype_map[dtype_name]
        for dim in capabilities["supported_dims"]:
            shape = (4096, 1) if dim == 0 else (1, 4096)
            inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
            functional_plan = index_fill_module._select_ascendc_index_fill_plan(
                inp,
                dim,
                index,
                -3.5,
                mode=index_fill_module.INDEX_FILL_FUNCTIONAL,
            )
            inplace_plan = index_fill_module._select_ascendc_index_fill_plan(
                inp,
                dim,
                index,
                -3.5,
                mode=index_fill_module.INDEX_FILL_INPLACE,
            )
            assert functional_plan.implementation == index_fill_module.INDEX_FILL_ASCENDC
            assert inplace_plan.implementation == index_fill_module.INDEX_FILL_ASCENDC

            expected_functional_path = (
                "dim0_functional_direct" if dim == 0 else "general"
            )
            expected_inplace_path = "dim0_inplace_small" if dim == 0 else "general"
            assert (
                c_operators.index_fill_ascendc_debug_path(inp, dim, index, False)
                == expected_functional_path
            )
            assert (
                c_operators.index_fill_ascendc_debug_path(inp, dim, index, True)
                == expected_inplace_path
            )

    unsupported_index = torch.zeros(
        capabilities["min_index_len"] - 1,
        dtype=torch.long,
        device=flag_gems.device,
    )
    unsupported_input = torch.empty(
        (1, 4096), dtype=torch.float16, device=flag_gems.device
    )
    fallback_plan = index_fill_module._select_ascendc_index_fill_plan(
        unsupported_input,
        1,
        unsupported_index,
        -3.5,
        mode=index_fill_module.INDEX_FILL_FUNCTIONAL,
    )
    assert fallback_plan.implementation == index_fill_module.INDEX_FILL_TRITON
    with pytest.raises(RuntimeError, match="unsupported.*index"):
        c_operators.index_fill_ascendc_debug_path(
            unsupported_input, 1, unsupported_index, False
        )


@pytest.mark.index_fill_out
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_scalar_out(dtype):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    value = _scalar_value(dtype)
    out = torch.empty_like(inp)
    ref_out = torch.empty_like(utils.to_reference(inp, False))

    ref = torch.ops.aten.index_fill.int_Scalar_out(
        utils.to_reference(inp, False),
        1,
        utils.to_reference(index, False),
        value,
        out=ref_out,
    )
    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res = torch.ops.aten.index_fill.int_Scalar_out(inp, 1, index, value, out=out)

    assert res is out
    utils.gems_assert_equal(res, ref)


@pytest.mark.index_fill_out
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_tensor_out(dtype):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    value = torch.tensor(_scalar_value(dtype), dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_out = torch.empty_like(utils.to_reference(inp, False))

    ref = torch.ops.aten.index_fill.int_Tensor_out(
        utils.to_reference(inp, False),
        1,
        utils.to_reference(index, False),
        utils.to_reference(value, False),
        out=ref_out,
    )
    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res = torch.ops.aten.index_fill.int_Tensor_out(inp, 1, index, value, out=out)

    assert res is out
    utils.gems_assert_equal(res, ref)


@pytest.mark.index_fill_
def test_index_fill_invalid_index_dtype():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([1], dtype=torch.int32, device=flag_gems.device)
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        IndexError, match="Expected dtype int64"
    ):
        inp.index_fill_(1, index, -1.0)


@pytest.mark.index_fill_
def test_index_fill_invalid_index_ndim():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([[1]], dtype=torch.long, device=flag_gems.device)
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        IndexError, match="Index is supposed to be a vector"
    ):
        inp.index_fill_(1, index, -1.0)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device not in ("cuda", "npu"),
    reason="out-of-range behavior is backend-specific",
)
@pytest.mark.parametrize("op_name", ("index_fill", "index_fill_"))
@pytest.mark.parametrize(
    "execution_path", INDEX_FILL_OOB_PATHS
)
def test_index_fill_out_of_range_index_device_assert(op_name, execution_path):
    if execution_path == "python_strided":
        input_setup = (
            "inp = torch.zeros((3, 4), device=flag_gems.device).t()\n"
            "dim = 1\n"
            "index_value = 3"
        )
    else:
        input_setup = (
            "inp = torch.zeros((3, 4), device=flag_gems.device)\n"
            "dim = 1\n"
            "index_value = 4"
        )
    operation = (
        "inp = inp.index_fill(dim, index, 1.0)"
        if op_name == "index_fill"
        else "inp.index_fill_(dim, index, 1.0)"
    )

    child_code = f"""
import torch
import flag_gems
from flag_gems.runtime import torch_device_fn

{input_setup}
index = torch.tensor([index_value], dtype=torch.long, device=flag_gems.device)
ops = {INDEX_FILL_OPS!r}
try:
    with flag_gems.use_gems(include=ops):
        {operation}
        torch_device_fn.synchronize()
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
    raise SystemExit(0)
raise SystemExit(1)
"""
    env = os.environ.copy()
    env.pop("FLAG_GEMS_INDEX_FILL_BOUNDS_CHECK", None)
    if execution_path != "cpp":
        env["FLAG_GEMS_INDEX_FILL_CPP_LAUNCHER"] = "0"

    result = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert (
        "device-side assert" in output
        or "index out of bounds" in output
        or "index out of range" in output
    )


@pytest.mark.index_fill_
def test_index_fill_out_of_range_index(monkeypatch):
    monkeypatch.setenv("FLAG_GEMS_INDEX_FILL_BOUNDS_CHECK", "sync")
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([4], dtype=torch.long, device=flag_gems.device)
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        IndexError, match="index out of range"
    ):
        inp.index_fill_(1, index, -1.0)


@pytest.mark.index_fill_
def test_index_fill_invalid_tensor_value_ndim():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([1], dtype=torch.long, device=flag_gems.device)
    value = torch.tensor([1.0], device=flag_gems.device)
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        RuntimeError, match="0-dimensional value tensor"
    ):
        inp.index_fill_(1, index, value)


@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device == "cpu", reason="device mismatch requires device backend"
)
def test_index_fill_cpu_index_rejected():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([1], dtype=torch.long, device="cpu")
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        RuntimeError, match="same device"
    ):
        inp.index_fill_(1, index, -1.0)
