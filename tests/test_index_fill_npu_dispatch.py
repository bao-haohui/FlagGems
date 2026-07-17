import importlib.util
from pathlib import Path

import pytest
import torch


_DISPATCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/flag_gems/runtime/backend/_ascend/ops/index_fill_dispatch.py"
)
_SPEC = importlib.util.spec_from_file_location("_index_fill_dispatch_under_test", _DISPATCH_PATH)
dispatch = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(dispatch)


def _counter_callable(calls, name, result):
    def call(*_args, **_kwargs):
        calls[name] += 1
        return result

    return call


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
def test_cpp_index_fill_dispatch_uses_injected_launcher(dtype, inplace):
    calls = {"cpp": 0, "fallback": 0}
    inp = torch.empty((4, 8), dtype=dtype)
    index = torch.arange(8, dtype=torch.long)
    result = dispatch.try_cpp_index_fill(
        inp,
        0,
        index,
        3.14159,
        inplace=inplace,
        launcher=_counter_callable(calls, "cpp", "cpp"),
        fallback=_counter_callable(calls, "fallback", "fallback"),
    )
    assert result == "cpp"
    assert calls == {"cpp": 1, "fallback": 0}


@pytest.mark.parametrize(
    "inp,dim,index,value",
    [
        (torch.empty((4, 8)), 1, torch.arange(8, dtype=torch.long), 1.0),
        (torch.empty((4, 16))[:, ::2], 0, torch.arange(8, dtype=torch.long), 1.0),
        (torch.empty((4, 8)), 0, torch.arange(8, dtype=torch.long), torch.tensor(1.0)),
        (torch.empty((4, 17)), 0, torch.arange(17, dtype=torch.long), 1.0),
    ],
)
def test_cpp_index_fill_dispatch_falls_back_for_unsupported_inputs(inp, dim, index, value):
    calls = {"cpp": 0, "fallback": 0}
    result = dispatch.try_cpp_index_fill(
        inp,
        dim,
        index,
        value,
        inplace=False,
        launcher=_counter_callable(calls, "cpp", "cpp"),
        fallback=_counter_callable(calls, "fallback", "fallback"),
    )
    assert result == "fallback"
    assert calls == {"cpp": 0, "fallback": 1}


def test_installed_npu_launcher_kernel_resource():
    kernel_path = (
        _DISPATCH_PATH.parent.parent
        / "kernels"
        / "index_fill_dim0_npu.py"
    )
    assert kernel_path.is_file()


def test_cpp_index_fill_dispatch_falls_back_when_launcher_is_unavailable(monkeypatch):
    calls = {"fallback": 0}
    monkeypatch.setattr(dispatch, "_get_cpp_index_fill_ops", lambda: (None, None))
    result = dispatch.try_cpp_index_fill(
        torch.empty((4, 8), dtype=torch.float16),
        0,
        torch.arange(8, dtype=torch.long),
        1.0,
        inplace=False,
        fallback=_counter_callable(calls, "fallback", "fallback"),
    )
    assert result == "fallback"
    assert calls == {"fallback": 1}
