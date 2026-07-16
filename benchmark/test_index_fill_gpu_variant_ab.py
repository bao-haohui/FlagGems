"""Conservative A/B guardrails for CUDA index_fill small-inner variants.

The benchmark is opt-in. It compares the C++/Triton ``small_inner`` launcher
against the same launcher forced to ``general`` for inner sizes 2 and 4, while
also measuring the normal FlagGems operator against eager CUDA PyTorch.
"""

import os
import statistics
import time

import flag_gems
import pytest
import torch

from . import base, consts


_ENABLE_ENV = "FLAGGEMS_INDEX_FILL_GPU_VARIANT_AB"
_ROUNDS_ENV = "FLAGGEMS_INDEX_FILL_GPU_VARIANT_AB_ROUNDS"
_SAMPLES_ENV = "FLAGGEMS_INDEX_FILL_GPU_VARIANT_AB_SAMPLES"
_DEFAULT_ROUNDS = 5
_DEFAULT_SAMPLES = 11
_GENERAL_MIN_RATIO = 0.85
_SPECIALIZED_MIN_GAIN = 1.10


_CASES = (
    # inner size, shape, index length
    (2, (64, 4096, 2), 16),
    (2, (64, 4096, 2), 2048),
    (2, (64, 4096, 2), 4096),
    (4, (64, 4096, 4), 16),
    (4, (64, 4096, 4), 2048),
    (4, (64, 4096, 4), 4096),
)


pytestmark = pytest.mark.skipif(
    flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="The CUDA small-inner A/B benchmark requires a CUDA device.",
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
    for _ in range(3):
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


def _torch_op(inplace):
    return torch.Tensor.index_fill_ if inplace else torch.index_fill


def _benchmark_launcher(c_operators, inplace, variant):
    function = (
        c_operators.index_fill_scalar_benchmark_
        if inplace
        else c_operators.index_fill_scalar_benchmark
    )

    def call(inp, dim, index, value):
        return function(inp, dim, index, value, variant)

    return call


def _assert_correct(c_operators, args, inplace):
    torch_op = _torch_op(inplace)
    expected = torch_op(*_call_args(args, inplace))
    base.torch_device_fn.synchronize()

    for variant in ("small_inner", "general"):
        actual = _benchmark_launcher(c_operators, inplace, variant)(
            *_call_args(args, inplace)
        )
        base.torch_device_fn.synchronize()
        torch.testing.assert_close(actual, expected)

    with flag_gems.use_gems(exclude=["zero_"]):
        actual = torch_op(*_call_args(args, inplace))
    base.torch_device_fn.synchronize()
    torch.testing.assert_close(actual, expected)


def _case_decision(specialized_over_general, general_min_ratio, gems_min_ratio):
    if general_min_ratio < _GENERAL_MIN_RATIO:
        return "KEEP: general below floor"
    if gems_min_ratio < _GENERAL_MIN_RATIO:
        return "KEEP: FlagGems below floor"
    if specialized_over_general >= _SPECIALIZED_MIN_GAIN:
        return "KEEP: specialized gain"
    return "REMOVE candidate"


def _variant_decision(records):
    return _case_decision(
        max(record["specialized_over_general"] for record in records),
        min(record["general_min_ratio"] for record in records),
        min(record["gems_min_ratio"] for record in records),
    )


def _print_summary(records):
    print("\nvariant summary across all representative cases")
    print(
        f"{'variant':<16} {'max gen/spec':>13} {'min Torch/gen':>13} "
        f"{'min Torch/Gems':>15} decision"
    )
    for inner_size in sorted({record["inner_size"] for record in records}):
        inner_records = [
            record for record in records if record["inner_size"] == inner_size
        ]
        print(
            f"small_inner_{inner_size:<2} "
            f"{max(record['specialized_over_general'] for record in inner_records):>13.3f} "
            f"{min(record['general_min_ratio'] for record in inner_records):>13.3f} "
            f"{min(record['gems_min_ratio'] for record in inner_records):>15.3f} "
            f"{_variant_decision(inner_records)}"
        )


def _run_case(c_operators, inner_size, shape, index_len, inplace, rounds, samples):
    inp = torch.randn(shape, dtype=torch.float16, device=base.device)
    index = torch.arange(index_len, dtype=torch.long, device=base.device)
    args = (inp, 1, index, 3.14159)
    _assert_correct(c_operators, args, inplace)

    specialized = []
    general = []
    torch_latency = []
    gems = []
    specialized_op = _benchmark_launcher(c_operators, inplace, "small_inner")
    general_op = _benchmark_launcher(c_operators, inplace, "general")
    torch_op = _torch_op(inplace)

    for _ in range(rounds):
        specialized.append(_p50(specialized_op, args, inplace, samples))
        general.append(_p50(general_op, args, inplace, samples))
        torch_latency.append(_p50(torch_op, args, inplace, samples))
        with flag_gems.use_gems(exclude=["zero_"]):
            gems.append(_p50(torch_op, args, inplace, samples))

    specialized_over_general = statistics.median(
        general_latency / specialized_latency
        for specialized_latency, general_latency in zip(specialized, general)
    )
    general_min_ratio = min(
        reference_latency / general_latency
        for reference_latency, general_latency in zip(torch_latency, general)
    )
    gems_min_ratio = min(
        reference_latency / gems_latency
        for reference_latency, gems_latency in zip(torch_latency, gems)
    )
    density = index_len / shape[1]
    print(
        f"small_inner_{inner_size:<2} {'inplace' if inplace else 'functional':<10} "
        f"{str(shape):<16} {index_len:>5} {density:>8.4f} "
        f"{statistics.median(specialized):>10.4f} "
        f"{statistics.median(general):>10.4f} "
        f"{specialized_over_general:>9.3f} "
        f"{general_min_ratio:>10.3f} {gems_min_ratio:>11.3f} "
        f"{_case_decision(specialized_over_general, general_min_ratio, gems_min_ratio)}"
    )
    return {
        "inner_size": inner_size,
        "specialized_over_general": specialized_over_general,
        "general_min_ratio": general_min_ratio,
        "gems_min_ratio": gems_min_ratio,
    }


@pytest.mark.index_fill
@pytest.mark.parametrize("inplace", [False, True])
def test_index_fill_gpu_small_inner_variant_ab(inplace):
    if os.environ.get(_ENABLE_ENV, "").lower() not in ("1", "true", "yes", "on"):
        pytest.skip(f"set {_ENABLE_ENV}=1 to run GPU variant A/B measurements")
    if base.Config.mode != consts.BenchMode.OPERATOR:
        pytest.skip("GPU variant A/B measurements require --mode=operator")

    from flag_gems.config import c_operators

    if c_operators is None or not hasattr(
        c_operators, "index_fill_scalar_benchmark"
    ):
        pytest.skip("FlagGems was built without the CUDA index_fill launcher")

    rounds = _positive_env(_ROUNDS_ENV, _DEFAULT_ROUNDS)
    samples = _positive_env(_SAMPLES_ENV, _DEFAULT_SAMPLES)
    print(
        "\nvariant          mode       shape            index  density spec ms    gen ms "
        " gen/spec Torch/gen min Torch/Gems min decision"
    )
    records = []
    for inner_size, shape, index_len in _CASES:
        records.append(
            _run_case(
                c_operators,
                inner_size,
                shape,
                index_len,
                inplace,
                rounds,
                samples,
            )
        )
    _print_summary(records)
