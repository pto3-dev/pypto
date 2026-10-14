# BackendHandler: principled backend dispatch

> Tracking: [issue #948](https://github.com/hw-native-sys/pypto/issues/948)

## Why

Earlier revisions of PyPTO branched on `backend::BackendType` directly inside
passes and codegen:

```cpp
if (backend::GetBackendType() != backend::BackendType::Ascend910B) { ... }
```

Every new backend therefore required hunting for these scattered conditionals
and adding another arm to each. `BackendHandler` replaces every such branch
with a single virtual call so that adding a new backend is a self-contained
change.

## What

`BackendHandler` (`include/pypto/backend/common/backend_handler.h`) is an
abstract interface that names every behavioural difference between backends.
Each `Backend` subclass owns a singleton `BackendHandler` subclass and exposes
it through the new pure-virtual `Backend::GetHandler()`.

```text
                       ┌────────────────────────────┐
        Pass / Codegen │  PassContext::Current()    │
                       │     ->GetBackendHandler()  │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │  BackendConfig::GetBackend │
                       │     ->GetHandler()         │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────┐
                │  Backend910B / Backend950 / ...      │
                │     -> Ascend910BHandler::Instance() │
                │     -> Ascend950Handler::Instance()  │
                └──────────────────────────────────────┘
```

The `PassContext` accessor satisfies the `pass-context-config` rule: passes
look up backend behaviour via the active `PassContext`, not by reaching out
to global state.

## Interface

| Method | Purpose | Ascend910B | Ascend950 |
| ------ | ------- | ---------- | --------- |
| `GetPtoTargetArch()` | `module attributes {pto.target_arch = "..."}` | `"a2a3"` | `"a5"` |
| `GetLaunchSpecCoreCountMethod()` | runtime API name on `launch_spec` | `"set_block_num"` | `"set_core_num"` |
| `GetDefaultSimPlatform()` | default simulator platform | `"a2a3sim"` | `"a5sim"` |
| `GetExtraPtoasFlags()` | extra ptoas flags | `[]` | `["--pto-arch", "a5"]` |
| `RequiresGMPipeBuffer()` | inject GM-backed pipe slot in `ExpandMixedKernel` | `true` | `false` |
| `RequiresSplitLoadTpopWorkaround()` | MemoryReuse load + tpop_from_aic in-place hazard guard | `true` | `false` |
| `RequiresLevel3TmpScratch()` | InitMemRef materialize level-3 explicit tmp; codegen static-view bridges | `true` | `false` |
| `RequiresVtoCFractalAdapt()` | AIV-side V-to-C fractal adapter `tile.move` | `false` | `true` |
| `RequiresRuntimeSubblockBridge()` | split AIV wrappers source subblock id from runtime | `true` | `false` |
| `RequiresNoSplitDualAivDispatch()` | `no_split` mixed kernels still dispatch on both AIV lanes | `true` | `false` |
| `GetL0cMAlignment(dtype)` | physical M-row alignment shared by L0C tiling, dbC capacity, and allocation | 32 for INT32; otherwise 16 | 16 |
| `BuildCrossCoreTransferView(dest, view)` | layout at cross-core transfer boundary | NZ for Mat/Left/Right; preserve for Vec | NZ for Mat/Left/Right; preserve for Vec (a5 hardware also requires fractal at the boundary) |

## Adding a new backend

1. Create `Backend<Arch>` in `src/backend/<arch>/backend_<arch>.cpp` (subclass
   `Backend`). Add the source to `CMakeLists.txt`.
2. Implement `Backend<Arch>Handler` in
   `src/backend/<arch>/backend_<arch>_handler.cpp` (subclass `BackendHandler`).
3. Override `Backend<Arch>::GetHandler()` to return your handler singleton.
4. Append the new backend to `BackendType` and to the factory switch in
   `src/backend/common/backend.cpp` (`GetBackendInstance` and
   `BackendTypeToString`). These are the only places that fan out by enum.

No pass or codegen file needs to change. Validate with
`tests/ut/backend/test_backend_handler.py` plus the regression tests in
`tests/ut/ir/transforms/` and `tests/ut/codegen/`.

## Python access

The handler is also exposed through the Python bindings:

```python
from pypto.pypto_core import backend as _backend_core

# When global config has been set:
handler = _backend_core.get_handler()

# When the caller already knows the desired backend:
handler = _backend_core.get_backend_instance(BackendType.Ascend950).get_handler()

handler.get_pto_target_arch()             # "a2a3" or "a5"
handler.requires_runtime_subblock_bridge()  # bool
handler.get_extra_ptoas_flags()           # list[str]
```

The runtime layer (`pypto.runtime.runner`, `pypto.ir.compiled_program`,
`pypto.backend.pto_backend`) consumes these accessors instead of branching on
`BackendType`, mirroring the C++ refactor.

## Ascend 310P compiler profile

`Backend310P` and `Ascend310PHandler` select the independent `a1` target.
The initial registry exposes FP16/FP32 vector load/add/subtract/multiply/store, tile allocation and
block indices. PTOAS validates static full aligned ND shapes and both memory
planners use conservative A1 budgets. The SoC records eight coupled AI cores
and seven additional vector cores, without assuming dual AIV dispatch.
Only proven DDR–Vec DMA routes are exposed initially. Conversion, BF16,
atomics, GEMM and cross-core operations need separate validated features.

Compile with `backend_type=BackendType.Ascend310P`; the compilation platform
defaults to `a1`. A legacy A3/A5 platform cannot override an explicit 310P
target. Use the matching A1 PTOAS and PTO-ISA feature branches. Backend
availability indicates compiler support. A compilation containing one static
InCore kernel is callable with contiguous CPU FP16/FP32 tensors on 310P using
`compiled(*tensors)` or `RunConfig(platform="a1", device_id=...)`. The synchronous
path uses the pinned A1 ISA and tensor descriptor runtime. Graph orchestration,
device-resident arguments, DFX and simulator dispatch remain unavailable.

A1 bandwidth hints remain disabled pending calibration.

Static ND partitions may load and store fewer columns than the parent tensor width. Row strides and column offsets must be 32-byte aligned, and the partition must fit the source view. The synchronous callable still takes a contiguous CPU parent tensor; output columns outside the partition are preserved.

Static full FP16/FP32 vector tiles no longer require a complete 256-byte arithmetic repeat. Add, subtract and multiply mask the final repeat; DMA rows still require 32-byte alignment. This enables one-block tiles without accepting dynamic valid shapes or unaligned column tails.

The bounded matrix profile supports K=N=16 with full static aligned M rows from 16 through 2048, FP16 Mat inputs and FP32 accumulators. Mixed matrix/vector functions remain one coupled AIC callable. Memory-space inference inserts an explicit FP32 ND Vec staging move before Acc stores, so liveness and address planning own the writeback buffer. Codegen emits `single_mode_vec0` for that move; ordinary Vec stores retain their path. Larger single-MAD K/N, partial rows, FP32 matrix inputs, split-AIV execution and other move families are rejected.

`requires_coupled_incore_dispatch()` selects this single-core mixed-body path. `supports_acc_to_gm_via_vec(FP32, FP32)` reports the explicit staging route; `supports_acc_to_gm_dtype()` continues to reject direct accumulator stores on A1.

A1 `pl.cast` supports static full FP16/FP32 ND Vec pairs with explicit `mode="none"`. The native cast uses nearest-even FP32-to-FP16 narrowing; signed zero, subnormals, overflow and NaN positions are covered. Both dtype rows must remain 32-byte aligned. Other rounding modes, scratch and dtype families are rejected. FP32 GEMM products can stage to Vec and narrow into FP16 outputs in the same coupled kernel.

`pl.matmul_acc(acc, lhs, rhs)` supports in-place FP32 accumulation for the same narrow matrix profile. The existing memory planner preserves accumulator storage. Literal `init_cond=True` reinitializes it, while `False` or an omitted condition adds another product. Unit flags remain unavailable. Reusing matrix operands can extend their L0 lifetimes; aggregate buffers must fit the configured budgets.

Aligned static Mat panel loads permit ordinary ND tensors with wider K/N extents. Each MAD remains K=N=16; explicitly load 16-wide panels, accumulate along K, and store successive N panels with aligned offsets. The native regression assembles `M x 32 @ 32 x 32` for M up to 1024 in one coupled kernel and validates the full output plus reload. This panel decomposition does not enable larger single-MAD shapes or dynamic/tail tiles.
