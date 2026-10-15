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
atomics, GEMM and cross-core operations need separate validated features; current extensions are described below.

Compile with `backend_type=BackendType.Ascend310P`; the compilation platform
defaults to `a1`. A legacy A3/A5 platform cannot override an explicit 310P
target. Use the matching A1 PTOAS and PTO-ISA feature branches. Backend
availability indicates compiler support. A compilation containing one static
InCore kernel is callable with contiguous CPU FP16/FP32 tensors on 310P using
`compiled(*tensors)` or `RunConfig(platform="a1", device_id=...)`. The synchronous
path uses the pinned A1 ISA and tensor descriptor runtime. Graph orchestration,
DFX and simulator dispatch remain unavailable; use `A1ResidentSession` below for explicit resident arguments.

A1 bandwidth hints remain disabled pending calibration.

Static ND partitions may load and store fewer columns than the parent tensor width. Row strides and column offsets must be 32-byte aligned, and the partition must fit the source view. The synchronous callable still takes a contiguous CPU parent tensor; output columns outside the partition are preserved.

Static full FP16/FP32 vector tiles no longer require a complete 256-byte arithmetic repeat. Add, subtract and multiply mask the final repeat; DMA rows still require 32-byte alignment. This enables one-block tiles without accepting dynamic valid shapes or unaligned column tails.

The bounded matrix profile supports K=N=16 with full static aligned M rows from 16 through 2048, FP16 Mat inputs and FP32 accumulators. Mixed matrix/vector functions remain one coupled AIC callable. Memory-space inference inserts an explicit FP32 ND Vec staging move before Acc stores, so liveness and address planning own the writeback buffer. Codegen emits `single_mode_vec0` for that move; ordinary Vec stores retain their path. Larger single-MAD K/N, partial rows, FP32 matrix inputs, split-AIV execution and other move families are rejected.

`requires_coupled_incore_dispatch()` selects this single-core mixed-body path. `supports_acc_to_gm_via_vec(FP32, FP32)` reports the explicit staging route; `supports_acc_to_gm_dtype()` continues to reject direct accumulator stores on A1.

A1 `pl.cast` supports static full FP16/FP32 ND Vec pairs with explicit `mode="none"`. The native cast uses nearest-even FP32-to-FP16 narrowing; signed zero, subnormals, overflow and NaN positions are covered. Both dtype rows must remain 32-byte aligned. Other rounding modes, scratch and dtype families are rejected. FP32 GEMM products can stage to Vec and narrow into FP16 outputs in the same coupled kernel.

`pl.matmul_acc(acc, lhs, rhs)` supports in-place FP32 accumulation for the same narrow matrix profile. The existing memory planner preserves accumulator storage. Literal `init_cond=True` reinitializes it, while `False` or an omitted condition adds another product. Unit flags remain unavailable. Reusing matrix operands can extend their L0 lifetimes; aggregate buffers must fit the configured budgets.

Aligned static Mat panel loads permit ordinary ND tensors with wider K/N extents. Each MAD remains K=N=16; explicitly load 16-wide panels, accumulate along K, and store successive N panels with aligned offsets. The native regression assembles `M x 32 @ 32 x 32` for M up to 1024 in one coupled kernel and validates the full output plus reload. This panel decomposition does not enable larger single-MAD shapes or dynamic/tail tiles.

A1 `pl.exp(tile)` and tile/tile `pl.div(lhs, rhs)` support full static ND FP16/FP32 Vec tiles with default precision. Division with `high_precision=True` is rejected. The native approximate arithmetic retains masked tails and repeat splitting. Tensor-form SiLU composes `zero - x`, `exp`, addition of an all-ones tensor and division; the native regression uses `@pl.jit.incore`, full compilation, public invocation and artifact reload. This validates an activation building block, not a complete YOLO model or portable IEEE edge-value semantics. Scalar division and other activation families remain unavailable.

A1 `pl.tile.full(shape, dtype, value)` fills static full ND FP16/FP32 Vec tiles from a compile-time scalar via `TEXPANDS`. SiLU can now materialize zero and one inside the kernel, removing the two host tensor inputs previously needed for these constants. Native tests cover both dtypes, masked repeats and public compile/call/reload. Runtime scalar arguments require separate support.

A1 exposes `pl.tile.row_sum`/`row_max` with full-size same-dtype scratch and `pl.tile.col_sum`/`col_max` for static full FP16/FP32 tiles. Row results preserve packed column-major `[M, 1]` layout through aligned contiguous stores; M must align the result to 32 bytes. Column results are `[1, N]`, and column sum currently rejects its optional binary-tree scratch form. Row buffers must be disjoint; column results may reuse the exact first input row, while partial overlap is rejected. Aggregate buffers must fit UB budgets. Arithmetic stays in the tile dtype; use FP32 for attention sum accumulation. Public native regressions check changed inputs, artifact reload and preservation outside aligned output partitions. Dynamic shapes, unaligned tails and special-value contracts remain unsupported.

Ascend310P additionally supports `tile.row_expand` from packed Mx1 reductions and `tile.col_expand` from row-major 1xN reductions into static aligned FP16/FP32 Vec tiles. Compose these with `sub`, `exp`, row/column `sum` and `div` for stable FP32 softmax. Row sources must be disjoint from the destination; column sources may share its exact first row. Fused broadcast arithmetic, dynamic and unaligned shapes remain unsupported.

`examples/models/yolo26_310p_softmax.py` validates an incremental YOLO26-nano integration using local Ultralytics weights. The two attention softmax modules execute as PTO3 FP32 kernels on 310P; convolution, matrix products and the remaining model run on CPU. Pass `--weights path/to/yolo26n.pt --size 128 --output-dir build_output/yolo26-hybrid` (or `--size 640`) with Ultralytics installed. It compares all tensor outputs for two inputs and reports warmed hybrid model and softmax-block host times, including allocation and transfers, separately from compilation/session creation. These are neither device-only timings nor full-PTO3 model FPS. Use the separate native runner below for complete 640x640 model execution.

Ascend310P standalone `@pl.jit.extern(core_type="aic", source=...)` entries preserve their source/include metadata through specialization and compile to a coupled-core wrapper. Relative sibling includes remain rooted at the original external source. Public CPU-carrier invocation and artifact reload use the same wrapper path. The runtime pin also exposes explicit device-resident tensors and registered multi-kernel images; device operations remain synchronous.

`A1ResidentSession(compiled)` prepares a standalone Ascend310P artifact. Use `session.upload`/`empty` for device tensors and `session.prepare(compiled)` to bind each exact static signature. Calling a prepared kernel accepts only matching session-owned tensors, with optional `blocks=1..8`; it performs no implicit activation transfer. Tensor `copy_from`/`to_cpu` make input/output transfers explicit. Images with identical source/include content share a registration within the session. Close drains the stream before releasing storage. This is synchronous host dispatch, not device graph capture.

The native YOLO26 operator plans in `examples/models/yolo26_native/operators.py` add static FP16 NHWC convolution for kernel 1/3 and stride 1/2. Channel padding, on-device boundary gathering, FP32 cube accumulation, bias and optional SiLU run on Ascend310P; CPU weight packing is constant preprocessing. Output pixel counts must be multiples of sixteen. The separate native runner below provides model-level validation and timing.

`examples/models/yolo26_native/layout.py` adds native input FP32 NCHW-to-FP16 NHWC packing, aligned channel slicing/concatenation, residual addition, nearest 2x upsampling, 5x5 max pooling and dense FP16/FP32 conversions. Static row strides and offsets keep feature-map routing on 310P, including half-rounded attention query scaling. Focused resident tests use eight blocks, changed inputs, artifact reload and adjacent guards.

`examples/models/yolo26_native/depthwise.py` provides aligned NHWC FP16 3x3 depthwise convolution with padding one, stride 1/2, FP32 accumulation, bias and optional SiLU. Packed constant weights use [kernel-height,kernel-width,channels] order; spatial gathering and all activation arithmetic execute on 310P. Resident tests cover changing inputs, artifact reload and output guards.

`examples/models/yolo26_native/matrix.py` adds native static FP16 matrix products with FP32 cube accumulation and optional transposed-right input. Sixteen-by-sixteen work tiles stream aligned row-major panels on 310P. The YOLO26 attention profiles cover QK products at N=16/400 with K=32 and probability/value products at N=16/400 with 64 output channels; inputs and outputs remain resident. Native cube kernels fence vector-side accumulator reads before cube reuse; the same lifecycle fix applies to convolution across repeated work tiles.

`examples/models/yolo26_native/detection.py` keeps the NMS-free YOLO26 head postprocessing on 310P: per-level FP16 box decoding, half-rounded sigmoid for 80 classes, then 300-anchor pruning and final 300 class/anchor selections. Dense outputs use FP32 [anchors,96] storage and final detections [300,6]. TopK uses deterministic candidate traversal for equal scores and runs as one native block; no host activation selection occurs.

### Complete native YOLO26-nano executor

`examples/models/yolo26_native/executor.py` records a static batch-one native trace for the fused 80-class NMS-free YOLO26-nano checkpoint. All input-dependent layout conversion, convolution, attention, feature routing, decoding and TopK execute on Ascend310P. CPU setup loads/fuses/packs constant weights; replay uses synchronous host dispatch with explicit transfers and device-resident intermediates. Unsupported modules raise an error. `RejectHostOperators` audits replay by rejecting any host ATen operator.

Create `NativeYOLO26(fused_half_model, 640, output_dir, device_id=0)` as a context manager and call `forward` with a contiguous CPU FP32 `[1,3,640,640]` input. Preparation compiles exact signatures and shares identical kernel images inside one resident session. The output is FP32 `[1,300,6]` detections. The 640x640 profile has been checked against all 23 feature outputs, every dense prediction and final TopK semantics for changing inputs; FP16 is used with FP32 cube accumulation and softmax. Hardware graph capture remains unsupported.

The additional 128x128 diagnostic geometry executes the full trace but currently has FP16 probability mismatches against the strict CPU reference. It is not included in the validated accuracy claim for 640x640.

### Native YOLO26 validation and performance

Run `python -m examples.models.yolo26_native.profile --weights path/to/yolo26n.pt --size 640 --output-dir build_output/yolo26-native --iterations 10` with the checkout, local Ultralytics dependency and matching Ascend310P toolchain available. The benchmark sets the one-to-one detection head before fusing weights, validates two changing synthetic uniform [0,1] inputs and audits native replay for zero host ATen operations. All 23 feature tensors, dense boxes/class probabilities and final detection scores are compared; probability tolerances remain rtol=0.03 and atol=1e-6. TopK membership and sorted scores are checked exactly against native dense predictions, allowing equal-score traversal differences from Torch. Numerical diagnostics are written before assertions, including strict mismatch counts.

The report includes checkpoint/native-source hashes, Ultralytics version, actual CANN logical/physical device identity, native launch coverage, raw samples and warmed median/mean/p95 latency and FPS. Full-forward timing includes input/output copies and synchronous host dispatch; resident replay excludes those copies and includes dispatch/synchronization. Native preparation includes trace/JIT creation, constant packing/upload and allocation; checkpoint loading and CPU fusion are excluded. Timings are unisolated host wall-clock observations. The default profile is batch-one 640x640; the optional 128x128 diagnostic retains its known strict numerical failure.

The native benchmark prepares constants with PTO2's BatchNorm folding order:
`scale = gamma / sqrt(var + eps)`, `weight *= scale`, and
`bias = (bias - mean) * scale + beta`, followed by FP16 narrowing. Its CPU
reference and native regular/depthwise convolution SiLU both materialize an
FP16 sigmoid before the final product. This differs from a fused SiLU with
only one output rounding. Preparation evaluates constants only; replay still
runs all activation arithmetic on Ascend310P. This arithmetic parity feature
does not relax the existing validation tolerances or establish complete PTO2
production conformance for every input profile.

`ConvPlan(..., four_way_accumulation=True)` selects an optional native vector
convolution with four independent FP32 partial sums over logical CHW terms.
The final one to three terms extend partial sum zero before the ordered merge,
matching the CPU FP16 golden's reduction policy without CPU model execution.
Weights use output-channel-block/CHW/output-lane packing with K padded to four.
`NativeYOLO26(..., four_way_accumulation=True)` selects this path for regular
convolutions. The default remains Cube accumulation. The vector policy trades
throughput for reproducible accumulation; other operators retain their own
rounding contracts.

The same `four_way_accumulation` policy also applies to aligned 3x3 depthwise
convolution. Its nine products accumulate into four FP32 vectors; the ninth
term extends partial sum zero, and the ordered merge precedes bias and FP16
rounding. Native model selection propagates the policy to both convolution
families. Omitting it retains the original sequential depthwise reduction.
