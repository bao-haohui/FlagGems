# `index_fill` Ascend C P0

This is a standalone direct-invoke probe for the portable Ascend C route. It
does **not** register an operator, alter `index_fill` dispatch, or replace the
current Triton implementation.

The same copy/fill kernel source is built per target `NpuArch`. At runtime the
host obtains the active SoC with `aclrtGetSocName()`, the visible AIV count with
`aclrtGetDeviceInfo(..., ACL_DEV_ATTR_VECTOR_CORE_NUM)`, and the per-core UB
capacity through `PlatformAscendCManager`. Tiling is calculated from those
values; no device-name or fixed core/UB constant participates in the plan.

## Build and run

First activate the CANN environment of the target container. For an A2 /
`dav-2201` artifact:

```bash
cd experimental/ascendc_index_fill_p0
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
INDEX_FILL_P0_NPU_ARCH=dav-2201 \
INDEX_FILL_P0_NPU_ARCH_ID=2201 \
bash scripts/run_p0.sh
```

The program runs both a contiguous copy and a vector fill with a non-aligned
tail. It prints the runtime capability and the resulting two-level tiling.

For another architecture, rebuild with its matching pair of CANN target and
runtime `NpuArch` id. For example, a platform that reports `3510` must build a
separate `dav-3510` artifact:

```bash
rm -rf build-dav-3510
INDEX_FILL_P0_BUILD_DIR=build-dav-3510 \
INDEX_FILL_P0_NPU_ARCH=dav-3510 \
INDEX_FILL_P0_NPU_ARCH_ID=3510 \
bash scripts/run_p0.sh
```

If a runtime artifact does not match the queried `NpuArch`, the executable
refuses to launch it and reports the future production behavior: fall back to
the existing Triton path. P0 intentionally stops there; the real fallback is
introduced only with P6 production dispatch.

## P0 acceptance evidence

Run this unchanged source on at least two NPU families and retain the console
output. The expected evidence is:

1. different valid `npu_arch`, AIV, or UB values produce valid tiling;
2. the matching artifact completes copy and fill verification;
3. a mismatching artifact is not launched;
4. no `index_fill` production Python or Triton file changes.
