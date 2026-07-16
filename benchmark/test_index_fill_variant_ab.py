"""A/B guardrails for specialized AscendC index_fill paths.

This benchmark is opt-in because it runs several synchronized measurements for
every case. It compares the selected specialized launcher path against the
same launcher forced to ``general``, then records the end-to-end FlagGems to
ACLNN ratio separately. No production dispatch condition is changed.
"""

import os
import statistics
import time

import flag_gems
import pytest
import torch

from . import base, consts
from .ascend_index_fill_reference import index_fill as aclnn_index_fill
from .ascend_index_fill_reference import index_fill_ as aclnn_index_fill_


_ENABLE_ENV = "FLAGGEMS_INDEX_FILL_VARIANT_AB"
_ROUNDS_ENV = "FLAGGEMS_INDEX_FILL_VARIANT_AB_ROUNDS"
_SAMPLES_ENV = "FLAGGEMS_INDEX_FILL_VARIANT_AB_SAMPLES"
_DEFAULT_ROUNDS = 5
_DEFAULT_SAMPLES = 7
_GENERAL_MIN_RATIO = 0.85
_SPECIALIZED_MIN_GAIN = 1.10


_CASES = (
    # path, shape, index length, duplicate indices
    ("dim0_functional_direct", (64, 4096), 8, False),
    ("dim0_functional_direct", (512, 1024), 16, False),
    ("dim0_functional_direct", (4096, 256), 16, False),
    ("dim0_functional_membership", (64, 4096), 24, False),
    ("dim0_functional_membership", (512, 1024), 128, False),
    ("dim0_functional_membership", (4096, 256), 256, False),
    ("dim0_inplace_small", (512, 4096), 8, False),
    ("dim0_inplace_small", (512, 1024), 128, False),
    ("dim0_inplace_small", (4096, 256), 256, False),
    ("dim0_inplace_deduplicate", (64, 4096), 8, True),
    ("dim0_inplace_deduplicate", (512, 1024), 8, True),
    ("dim0_inplace_deduplicate", (4096, 256), 8, True),
)


pytestmark = pytest.mark.skipif(
    base.vendor_name != "ascend" or base.device != "npu",
    reason="The AscendC variant A/B benchmark is only available on Ascend NPUs.",
)


def _positive_env(name, default):
    value = os.environ.get(name, str(default))
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _call_args(args, inplace):
    if inplace:
        return (args[0].clone(), *args[1:])
    return args


def _p50(op, args, inplace, samples):
    for _ in range(2):
        op(*_call_args(args, inplace))
    base.torch_device_fn.synchronize()

    latencies = []
    for _ in range(samples):
        call_args = _call_args(args, inplace)
        base.torch_device_fn.synchronize()
        start = time.perf_counter()
        result = op(*call_args)
        base.torch_device_fn.synchronize()
        latencies.append((time.perf_counter() - start) * 1e3)
        del result
    return statistics.median(latencies)


def _make_index(rows, index_len, duplicate):
    if duplicate:
        return torch.zeros(index_len, dtype=torch.long, device=base.device)
    return torch.arange(index_len, dtype=torch.long, device=base.device) % rows


def _benchmark_launcher(c_operators, path, args, inplace):
    def call(inp, dim, index, value):
        return c_operators.index_fill_ascendc_benchmark_scalar(
            inp, dim, index, value, inplace, path
        )

    return call


def _gems_op(inplace):
    return torch.Tensor.index_fill_ if inplace else torch.index_fill


def _aclnn_op(inplace):
    return aclnn_index_fill_ if inplace else aclnn_index_fill


def _assert_correct(c_operators, path, args, inplace):
    torch_op = _gems_op(inplace)
    expected = torch_op(*_call_args(args, inplace))
    base.torch_device_fn.synchronize()

    for forced_path in ("auto", "general"):
        actual = _benchmark_launcher(
            c_operators, forced_path, args, inplace
        )(*_call_args(args, inplace))
        base.torch_device_fn.synchronize()
        torch.testing.assert_close(actual, expected)

    with flag_gems.use_gems(exclude=["zero_"]):
        actual = torch_op(*_call_args(args, inplace))
    base.torch_device_fn.synchronize()
    torch.testing.assert_close(actual, expected)


def _decision(specialized_over_general, general_min_ratio, gems_min_ratio):
    if general_min_ratio < _GENERAL_MIN_RATIO:
        return "KEEP: general below floor"
    if gems_min_ratio < _GENERAL_MIN_RATIO:
        return "KEEP: FlagGems below floor"
    if specialized_over_general >= _SPECIALIZED_MIN_GAIN:
        return "KEEP: specialized gain"
    return "REMOVE candidate"


def _variant_decision(records):
    max_specialized_gain = max(record["specialized_over_general"] for record in records)
    min_general_ratio = min(record["general_min_ratio"] for record in records)
    min_gems_ratio = min(record["gems_min_ratio"] for record in records)
    if min_general_ratio < _GENERAL_MIN_RATIO:
        return "KEEP: general below floor"
    if min_gems_ratio < _GENERAL_MIN_RATIO:
        return "KEEP: FlagGems below floor"
    if max_specialized_gain >= _SPECIALIZED_MIN_GAIN:
        return "KEEP: specialized gain"
    return "REMOVE candidate"


def _print_variant_summary(records):
    print("\nvariant summary across all representative cases")
    print(
        f"{'path':<30} {'max gen/spec':>13} {'min ACLNN/gen':>14} "
        f"{'min ACLNN/Gems':>15} decision"
    )
    for path in sorted({record["path"] for record in records}):
        path_records = [record for record in records if record["path"] == path]
        max_specialized_gain = max(
            record["specialized_over_general"] for record in path_records
        )
        min_general_ratio = min(
            record["general_min_ratio"] for record in path_records
        )
        min_gems_ratio = min(
            record["gems_min_ratio"] for record in path_records
        )
        print(
            f"{path:<30} {max_specialized_gain:>13.3f} "
            f"{min_general_ratio:>14.3f} {min_gems_ratio:>15.3f} "
            f"{_variant_decision(path_records)}"
        )


def _run_case(c_operators, path, shape, index_len, duplicate, inplace, rounds, samples):
    inp = torch.randn(shape, dtype=torch.float16, device=base.device)
    index = _make_index(shape[0], index_len, duplicate)
    args = (inp, 0, index, 3.14159)
    _assert_correct(c_operators, path, args, inplace)

    auto_path = c_operators.index_fill_ascendc_debug_path(
        inp, 0, index, inplace
    )
    assert auto_path == path

    specialized = []
    general = []
    aclnn = []
    gems = []
    specialized_op = _benchmark_launcher(c_operators, "auto", args, inplace)
    general_op = _benchmark_launcher(c_operators, "general", args, inplace)
    aclnn_op = _aclnn_op(inplace)
    gems_op = _gems_op(inplace)

    for _ in range(rounds):
        specialized.append(_p50(specialized_op, args, inplace, samples))
        general.append(_p50(general_op, args, inplace, samples))
        aclnn.append(_p50(aclnn_op, args, inplace, samples))
        with flag_gems.use_gems(exclude=["zero_"]):
            gems.append(_p50(gems_op, args, inplace, samples))

    specialized_over_general = statistics.median(
        general_latency / specialized_latency
        for specialized_latency, general_latency in zip(specialized, general)
    )
    general_min_ratio = min(
        aclnn_latency / general_latency
        for aclnn_latency, general_latency in zip(aclnn, general)
    )
    gems_min_ratio = min(
        aclnn_latency / gems_latency
        for aclnn_latency, gems_latency in zip(aclnn, gems)
    )
    density = index_len / shape[0]
    decision = _decision(
        specialized_over_general, general_min_ratio, gems_min_ratio
    )
    print(
        f"{path:<30} {'inplace' if inplace else 'functional':<10} "
        f"{str(shape):<14} {index_len:>5} {density:>8.4f} "
        f"{statistics.median(specialized):>10.4f} "
        f"{statistics.median(general):>10.4f} "
        f"{specialized_over_general:>9.3f} "
        f"{general_min_ratio:>11.3f} {gems_min_ratio:>11.3f} {decision}"
    )
    return {
        "path": path,
        "specialized_over_general": specialized_over_general,
        "general_min_ratio": general_min_ratio,
        "gems_min_ratio": gems_min_ratio,
    }


@pytest.mark.index_fill
@pytest.mark.parametrize("inplace", [False, True])
def test_index_fill_ascendc_variant_ab(inplace):
    if os.environ.get(_ENABLE_ENV, "").lower() not in ("1", "true", "yes", "on"):
        pytest.skip(f"set {_ENABLE_ENV}=1 to run variant A/B measurements")
    if base.Config.mode != consts.BenchMode.OPERATOR:
        pytest.skip("AscendC variant A/B measurements require --mode=operator")

    from flag_gems.config import c_operators

    if c_operators is None or not hasattr(
        c_operators, "index_fill_ascendc_benchmark_scalar"
    ):
        pytest.skip("FlagGems was built without the AscendC benchmark extension")

    rounds = _positive_env(_ROUNDS_ENV, _DEFAULT_ROUNDS)
    samples = _positive_env(_SAMPLES_ENV, _DEFAULT_SAMPLES)
    print(
        "\npath                           mode       shape          index  density "
        "spec ms    gen ms  gen/spec ACLNN/gen min ACLNN/Gems min decision"
    )
    records = []
    for path, shape, index_len, duplicate in _CASES:
        expected_inplace = path.startswith("dim0_inplace")
        if inplace != expected_inplace:
            continue
        records.append(
            _run_case(
                c_operators,
                path,
                shape,
                index_len,
                duplicate,
                inplace,
                rounds,
                samples,
            )
        )
    _print_variant_summary(records)
