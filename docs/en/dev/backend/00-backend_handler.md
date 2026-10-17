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

`examples/models/yolo26_native/executor.py` records a static batch-one native trace for the fused 80-class NMS-free YOLO26-nano checkpoint. All input-dependent layout conversion, convolution, attention, feature routing, decoding and TopK execute on Ascend310P. CPU setup loads/fuses/packs constant weights; replay submits a static device model through one `A1L2Worker`, with explicit transfers and device-resident intermediates. Unsupported modules raise an error. `RejectHostOperators` audits replay by rejecting any host ATen operator.

Create `NativeYOLO26(fused_half_model, 640, output_dir, device_id=0)` as a context manager and call `forward` with a contiguous CPU FP32 `[1,3,640,640]` input. Preparation compiles exact signatures and shares identical kernel images inside one resident session. The output is FP32 `[1,300,6]` detections. The 640x640 profile has been checked against all 23 feature outputs, every dense prediction and final TopK semantics for changing inputs; FP16 is used with FP32 cube accumulation and softmax. Hardware graph capture remains unsupported.

The 128x128 geometry is validated with the combined PTO2 precision policy described below. Direct NativeYOLO26 callers select four_way_accumulation=True, linear_query_key=True and precise_small_softmax=True. All 23 features, both raw heads, dense scores and production detections pass the unchanged PTO2 checks for six synthetic inputs; strict probability checks also pass. The older Cube policy retains known 128 mismatches.

### Native YOLO26 validation and performance

Run `python -m examples.models.yolo26_native.profile --weights path/to/yolo26n.pt --size 640 --output-dir build_output/yolo26-native --iterations 10` with the checkout, local Ultralytics dependency and matching Ascend310P toolchain available. The benchmark sets the one-to-one detection head before fusing weights, validates two changing synthetic uniform [0,1] inputs and audits native replay for zero host ATen operations. All 23 feature tensors, dense boxes/class probabilities and final detection scores are compared; probability tolerances remain rtol=0.03 and atol=1e-6. TopK membership and sorted scores are checked exactly against native dense predictions, allowing equal-score traversal differences from Torch. Numerical diagnostics are written before assertions, including strict mismatch counts.

The report includes checkpoint/native-source hashes, Ultralytics version, actual CANN logical/physical device identity, native launch coverage, raw samples and warmed median/mean/p95 latency and FPS. Full-forward timing includes input/output copies and synchronous host dispatch; resident replay excludes those copies and includes dispatch/synchronization. Native preparation includes trace/JIT creation, constant packing/upload and allocation; checkpoint loading and CPU fusion are excluded. Timings are unisolated host wall-clock observations. The default profile is batch-one 640x640; the 128x128 profile selects the validated PTO2 precision policy by default.

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

An opt-in `MatrixPlan(..., linear_accumulation=True)` evaluates each dot product
in increasing K order with separate FP32 multiplication and addition, then
rounds once to FP16. It supports either B layout, uses native vector/scalar
instructions and keeps the Cube policy as default.
`NativeYOLO26(..., linear_query_key=True)` selects it for QK products only.
This matches the batched/transposed CPU golden reduction for the tested QK
layouts without changing attention scaling, activation rounding or budgets.

`NativeYOLO26(..., precise_small_softmax=True)` selects a compensated native
softmax for 16x16 finite FP16 scores (the 128 input's attention shape).
FP32 high/residual pairs retain rounding residuals through range-reduced
exponential evaluation, summation and normalization before final FP16 rounding.
This handles probabilities near FP16 midpoints without changing tolerances or
using CPU activation arithmetic. The regular DSL path remains the default and
continues to handle larger attention shapes.

The full-native profile CLI defaults to `--precision-policy auto`: 128x128 selects
the combined PTO2 policy (four-way regular/depthwise convolution, linear QK and
compensated small softmax); 640x640 retains Cube and the regular DSL softmax.
`--precision-policy cube` explicitly retains the previous 128 behavior, whose
CPU-golden comparison is not accepted. `--precision-policy pto2` is validated
for 128 only. Reports record the selected policy and keep every existing strict
probability assertion unchanged. The reference reduction policy trades throughput for
reproducibility and should not be compared with the dense Cube policy as if
they were the same execution policy.

`A1ResidentSession.prepare_sequence(calls)` binds ordered
`(prepared_kernel, resident_arguments, blocks)` tuples after checking session
ownership and static signatures. The sequence uploads descriptors once and
replays all kernels on the session stream with a single final wait. Direct sequence execution avoids per-kernel argument
allocation/upload and Python dispatch; native YOLO26 uses the device-model
worker described below. Input/output transfers
remain explicit; arithmetic and validation tolerances are unchanged. Reports
retain the kernel count and identify the single host replay submission. This
is stream-ordered execution, without hardware graph capture or AICPU scheduling.

The vector fallback for four-way convolution processes four or eight adjacent output pixels
per work item, selected by geometry and core occupancy. Weight rows and bias are broadcast across pixels using masked repeats;
each pixel retains its own four FP32 partial sums and the same ordered merge,
bias and FP16 activation boundaries. Static rectangular shapes, stride one/two,
logical channel tails and output guards remain supported. This reuses weights
and fills vector repeats without selecting a different precision policy.

Native detection TopK uses a bounded 300-element worst-first heap in each stage,
then sorts the retained candidates. Equal scores retain traversal order: anchor
order in stage one and selected-anchor rank/class order in stage two. Two
masked vector maxima reduce exactly the 80 class scores while excluding boxes
and padding. This replaces scalar scans and insertion shifts without changing
selected identities, scores, output geometry or the precision budget.

The four-way convolution loads one contiguous packed weight block for each
16 input channels, then converts it once to FP32 before consuming its ordered
four-term groups. The final transfer includes only the existing four-term
padding, so partial channel blocks never read the next output panel. Weight
cache storage ends at byte 82944 of private UB. Accumulation order, tail terms,
bias and activation rounding, packed weight ABI and validation tolerances are
unchanged. Exact tests cover repeated channel blocks and one/two/three-term
tails, changed inputs, artifact reload and output guards.

The four-way convolution places each pixel's four 16-channel partial sums in
one contiguous 64-lane segment. One vector instruction broadcasts the four
cached weight rows to the spatial tile. Strided 16-lane additions retain the
same left-to-right partial-sum merge, and tail terms still update partial zero
only. This changes private vector layout without changing input/output layout,
the packed weight ABI, arithmetic order or validation tolerances.

The four-way path expands each loaded FP16 patch once per channel block using
310P scatter transpose with repeated source rows. The expansion orders spatial
pixels inside each channel, so strided vector copies form a full four-term
group without scalar activation reads. FP16-to-FP32 conversion is exact; the
original ordered FP32 reduction and scalar tail are unchanged. The additional
private UB storage ends at byte 167936 and stays inside the 310P UB capacity.
Single-repeat 1x1 patches set both transpose repeat strides to zero. Exact
tests cover full and partial 1x1 channel blocks as well as repeated 3x3 blocks.

Only the channel block containing a one-to-three-term CHW tail prepares the
scalar FP32 patch buffer. Aligned blocks load every live FP16 patch lane and
expand them directly, skipping the unused clear and scalar-patch conversion.
The vector-to-load fence and the exact tail path remain in place.

For 1x1 convolutions with up to 512 padded reduction terms, each core retains
one complete FP32 output-channel panel across spatial work items. Traversal
visits output panels first, refreshes the cache when the panel changes, and
keeps the existing four-term reduction order. The FP32 cache ends at UB byte
229376; its FP16 staging buffer overlaps only the inactive per-block cache.
Larger panels and 3x3 convolutions keep the previous schedule and block cache.
Exact tests cover reuse, panel refresh, tails and both sides of the 512-term
boundary without changing existing expectations or tolerances.

Bounded 3x3 feature maps are staged once per core into NC1HWC0 L1 storage.
Hardware img2col then extracts aligned sixteen-pixel patches into a private UB
scratch tile; each work item selects its four live pixels. The image is reused
across spatial and output-channel work. Maps larger than the conservative
512 KiB L1 budget keep the GM gather path. Padding is explicitly zero, and
FP32 reduction order and FP16 activation boundaries remain unchanged.
The nine patch tiles occupy disjoint UB slots ending at byte 90624, so one
producer/consumer fence pair covers the whole extraction batch.

Input packing converts each live NCHW channel with vector FP32-to-FP16
conversion, then transposes a zero-padded 16x16 half tile into NHWC16 using
scatter transpose. Single-repeat strides are zero. This removes scalar
activation reads, casts and padding stores while retaining exact narrowing,
signed zero and padded channels; tests cover 1/3/16 channels and half midpoints.

Convolution consumes aligned channel slices directly using physical source
stride and offset metadata while keeping logical channel and weight geometry
separate. This removes materialized slice-copy kernels before ordinary
convolutions; depthwise inputs retain their compact path. Both the Cube and
ordered-vector kernels support sliced inputs. L1 staging packs only the live
channel window and falls back to GM gathers for DMA strides wider than 16 bits.
Exact tests poison surrounding source channels and retain output guards.

Cube attention products use two 1 KiB L1 slots. MTE2 prefetches a future A/B
panel after MTE1 releases its slot while MAD consumes separate L0 operands.
A separate MAD-to-MTE1 fence protects L0 reuse, and every slot-ready/release
flag is consumed, including the final one/two tiles. Aligned rows use batched
DMA; wide source gaps retain single-row transfers. The K traversal, MAD order,
transpose interpretation and accumulator writeback are unchanged.

Ordered attention products select two or four output rows per work item from
the row-panel count and launched core count. Small grids retain enough work
for all eight model cores. Batched row loads and a half-precision transpose
feed shared FP32 weight broadcasts; each output still adds products in strictly
increasing K order. Unused rows are zero-initialized and never stored. Exact
batched-golden tests cover both transpose modes and one/eight-core launches.

Ordered convolution writes its spatial tile with one strided MTE3 DMA.
The physical output stride retains padded channels and the output offset;
ConvPlan limits the DMA gap to 63 blocks. Vector-to-store and store-to-vector
fences still protect the rounded tile, with no changes to FP32 reduction or
FP16 activation boundaries. Existing exact convolution cases cover channel
padding, multiple output panels, repeated pixels and surrounding guards.

Cached 1x1 convolution processes eight spatial pixels per work item when
its eight-pixel row-panel count can populate every launched core. Small
grids and other shapes retain four pixels. Each output keeps its original
four ordered FP32 partial sums and scalar tail, followed by the same FP16
activation boundaries. The larger accumulator and transpose scratch remain
inside UB; the FP32 expansion is separated from the half scratch. Conversion
batches never exceed 144 repeats. Existing exact cases exercise both tile
widths, partial input channels, multiple output panels and channel views.

Ordered convolution specializes its validated 1x1 or 3x3 kernel size at entry.
Constant kernel area removes generic division/modulo and inactive geometry
branches from patch gathering and ordered-term address generation. Spatial
tile selection, runtime strides/padding, packed weights, buffer layout and all
FP32/FP16 arithmetic boundaries remain unchanged. Both kernel sizes retain
exact channel-tail, source-view and guard coverage.

The native model accepts opt-in `weight_l2_bypass=True`; its profile CLI exposes
`--weight-l2-bypass` with the 128x128 PTO2 precision policy. After initializing
the session, preparation queries `rtGetL2CacheOffset` for the current visible
runtime device. Four exact uint16 metadata words carry that offset to the
ordered convolution's read-only weight pointer. No device ordinal or offset is
hardcoded, and a missing offset or a dense Cube policy request fails explicitly. Activations,
biases, output writes, and the other operators retain their normal policy.

Cached weights remain the default: bypass execution does not imply a speedup.
Compare warmed calls with the same CANN profiler and unchanged precision checks.
Collect cache PMU data separately and retain raw events; an aggregate L2 hit rate
alone does not distinguish every bypass transaction. Concurrent jobs can affect
both task duration and chip-level DHA counters, even when rows are joined to
this model's stream/task IDs. This A1 native path transports an explicit runtime
offset; it does not rely on implicit SDK global configuration or claim frontend
`cache_policy` integration on A1.

Aligned 1x1, stride-one convolutions in the four-way policy use an ordered Cube
kernel when input channels are a multiple of 16 and at most 512. Each K16 MAD
has one nonzero weight row, so it adds exactly one FP16 product to the correct
one of four FP32 partial sums. The ordered merge, bias and FP16 SiLU boundaries
are retained. Other shapes and source DMA gaps exceeding 16 bits use the vector fallback.
Spatial tiles use 128, 64, 32 or
16 pixels according to divisibility and available work; each core reuses its
current output-channel weight panel in L1. Four L0C partials occupy at most
32 KiB and the L1 panel at most 256 KiB, with a separate 4 KiB input tile.

Ordered pointwise Cube weights now retain compact CHW constants in GM.
A private 8 KiB UB zero tile initializes the bounded sparse L1 panel once per
kernel. Strided GM-to-L1 transfers scatter sixteen compact weight rows into
their diagonal K16 positions; the panel remains resident across spatial work.
This removes the former 16-fold GM expansion without changing arithmetic,
source views, output guards or weight L2 bypass. The packed constant tensor
signature changes from `co * ci * 16` to `co * ci` FP16 elements. Recompile and
repack older native graphs when using this layout; existing compiled artifacts
retain their original tensor signatures.

The ordered 3x3 vector fallback also selects eight spatial pixels when its grid
has at least one work item per launched core, retaining four for small grids.
This reuses each im2col patch and converted weight block across twice as many
outputs without changing any output's reduction or rounding. The eight-pixel
3x3 expansion ends at UB byte 212992, inside the 310P capacity. It overlaps the
pointwise-only weight-cache region, which is inactive for 3x3 kernels; other
live buffers remain separate. Source views, channel tails, stride-two images,
oversized-image gathering and scalar reduction tails retain their behavior.

The four-way policy selects an ordered 3x3 Cube kernel when the complete
NC1HWC0 image plus sparse weight panels fit the Ascend310P 1 MiB L1 and the source DMA
gap fits 16 bits. GM weights retain the compact CHW layout: one zero tile
initializes the local L1 panels, then strided GM-to-L1 copies scatter each
logical weight row into its selected K16 row. No expanded GM weight allocation
is needed for this path. Shapes beyond that budget keep the vector fallback.

Direct hardware im2col feeds nine L0A patches from the resident L1 image for
16/32/64/128 spatial pixels. The kernel consumes channels and filter positions in
logical CHW order, retaining four FP32 sums and folding scalar tail terms into
partial zero. Bias and FP16 SiLU boundaries are unchanged. Nine live L0A patches
use at most 36 KiB; four L0C partials use at most 32 KiB. L1 panels survive until
the core changes its output-channel block. Source views, output guards, padded
channels, artifact reload and explicit weight L2 bypass remain supported.

The ordered Cube capacity check includes the 512-byte aligned image and every
locally sparse weight row. CANN reports `L1_SIZE=1048576` for Ascend310P3.
Using that capacity admits the 128x128 RGB stem and 128-channel 3x3 panels
without moving or rounding partial sums. Shapes beyond 1 MiB retain the
fallback; the capacity check also accounts for padded input channels.

When the full sparse weight panel exceeds L1 but the image plus two bounded
banks fit, the ordered 3x3 kernel streams 32 input channels per bank. Each bank
uses 144 KiB. MTE2 prefetches the next bank while MTE1 and Cube consume the
current one; independent ready/free events prevent overwrites, and both bank
lifetimes are drained before the next spatial work item. The resident image
is reused throughout. Four FP32 sums survive across every bank, including an
odd final C0 block and scalar CHW tail, with no partial-sum spill or narrowing.
Full panels still use their output-block cache; shapes that fit neither budget
retain the vector fallback. GM weights remain compact in both Cube paths.

Streamed Cube is selected only when the minimum M16-by-N16 grid provides at
least eight work items. A smaller grid cannot fill the 310P3's eight cores;
the measured 4x4,256-to16 head convolution keeps the faster vector schedule.
This utilization gate changes scheduling only, with unchanged accumulation
and precision budgets. Larger resident panels keep their existing policy.

Ordered pointwise and spatial Cube kernels select M128 when the output grid
still supplies at least one work item per launched core. Smaller grids retain
M64/M32/M16. Nine spatial L0A patches need at most 36 KiB of the 310P3's
64 KiB L0A; four L0C sums need 32 KiB. UB epilogue slots scale to 8 KiB,
while the zero tile at byte 65536 and metadata at byte 131072 remain separate.
This widens reuse without changing any output's accumulation or FP16 boundary.

### A1 static L2 task graphs

`A1L2Worker(compiled, device_id=0)` owns one chip's resident tensors and
registered kernel images. Call `prepare(compiled)` for each static kernel,
then `register_graph([(kernel, arguments, blocks), ...])` and `run(graph)`.
The graph retains immutable device addresses; update input values explicitly.
The A1 backend prepares a CANN device model and submits it once per replay.
Dependencies follow device stream order. Kernel preparation launches are
outside replay. Close invalidates all graphs and drains execution before
releasing images or tensors. Ownership/signature failures are rejected.

This static coupled-core backend is distinct from generic `ChipWorker` and
does not port A2/A3 TensorMap/AICPU dynamic orchestration to 310P. It requires
a runtime pin exposing `A1SingleLaunch.prepare_graph`.

The native YOLO26 executor registers its trace with one `A1L2Worker` and
replays the prepared device model. Coverage separates device compute tasks
from host submissions: 216 compute tasks remain 216 tasks; replay submits
one model with zero per-task host kernel launches. All tensor addresses,
arithmetic and precision tolerances are preserved.
