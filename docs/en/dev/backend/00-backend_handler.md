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
| `HonorsHighPrecisionAlgorithm()` | PTO-ISA selects a distinct algorithm from the `precisionType` attribute that `high_precision=True` emits; when false, both PTO emitters (Tile op callbacks and Buffer IR) warn that the request is dropped | `false` | `true` |
| `RequiresGMPipeBuffer()` | inject GM-backed pipe slot in `ExpandMixedKernel` | `true` | `false` |
| `RequiresSplitLoadTpopWorkaround()` | MemoryReuse load + tpop_from_aic in-place hazard guard | `true` | `false` |
| `RequiresLevel3TmpScratch()` | InitMemRef materialize level-3 explicit tmp; codegen static-view bridges | `true` | `false` |
| `RequiresVtoCFractalAdapt()` | AIV-side V-to-C fractal adapter `tile.move` | `false` | `true` |
| `RequiresRuntimeSubblockBridge()` | split AIV wrappers source subblock id from runtime | `true` | `false` |
| `RequiresNoSplitDualAivDispatch()` | `no_split` mixed kernels still dispatch on both AIV lanes | `true` | `false` |
| `SplitsCubeToVectorTransportInHardware()` | a split Cube-to-Vector push hands each AIV lane half of the transported extent, so that extent is the lane partition | `false` | `true` |
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

The bounded matrix profile supports full static M/K dimensions aligned to 16 in [16, 2048], with N=16 or 32, FP16 Mat inputs and FP32 accumulators. Each Left/Right operand must fit its 64 KiB L0 budget; Mat remains limited to the configured 512 KiB. A1 selects ZZ Left operands and supports legacy NZ Left at K16. Mixed matrix/vector functions remain one coupled AIC callable. Memory-space inference inserts explicit FP32 ND Vec staging before Acc stores, and codegen emits `single_mode_vec0`. Partial shapes, FP32 matrix inputs, single-MAD N>32, split-AIV execution are rejected.

A static `pl.pipeline(..., stage=2)` loop can place consecutive FP16 Mat loads in separate L1 slots before their GEMM consumers. Public 310P regressions cover a single tile, odd and even trip counts, exact output and buffer guards. This demonstrates the generated staging and dependencies for that profile; it does not guarantee a performance gain or overlap for every loop.

`requires_coupled_incore_dispatch()` selects this single-core mixed-body path. `supports_acc_to_gm_via_vec(FP32, FP32)` reports the explicit staging route; `supports_acc_to_gm_dtype()` continues to reject direct accumulator stores on A1.

A1 `pl.cast` supports static full FP16/FP32 ND Vec pairs with explicit `mode="none"`. The native cast uses nearest-even FP32-to-FP16 narrowing; signed zero, subnormals, overflow and NaN positions are covered. Both dtype rows must remain 32-byte aligned. Other rounding modes, scratch and dtype families are rejected. FP32 GEMM products can stage to Vec and narrow into FP16 outputs in the same coupled kernel.

`pl.matmul_acc(acc, lhs, rhs)` supports in-place FP32 accumulation for the same N16/N32 multi-K matrix profile. The existing memory planner preserves accumulator storage. Literal `init_cond=True` reinitializes it, while `False` or an omitted condition adds another product. Unit flags remain unavailable. Reusing matrix operands can extend their L0 lifetimes; aggregate buffers must fit the configured budgets.

Aligned static Mat loads accept ND tensors with wider K/N extents. A single MAD can consume multiple K16 fractals within the operand budget; native public compile/call/reload regressions cover K32, K144 and the K2048 boundary. N32 accumulator results must be explicitly sliced into full-height aligned N16 panels before Vec staging; direct N32 writeback and other accumulator slices remain unsupported. Dynamic/tail tiles and automatic extraction schedules are not enabled by this profile. Changing the K decomposition can change FP32 accumulation order and later FP16 rounding; primitive exact tests use representable dyadic sums and do not imply whole-model precision equivalence.

For full contiguous FP16/FP32 Vec tiles, `pl.tile.slice` supports a bounded static row interval spanning the complete row width. The slice aliases its parent storage and introduces no copy; dynamic offsets, partial columns and valid-shape overrides are rejected. The matching PTOAS path preserves parent alias dependencies when inserting synchronization.

The YOLO26n 160x160 CI32/CO32 ordered pointwise profile uses the public N32 matmul and N16 accumulator panels. Its single-core-per-image batch path pipelines complete spatial tiles; other core counts use a strided fallback. Weight packing retains the four ordered partials, and the profile is selected only for the matching contiguous geometry without an L2 alias. Native code remains available for other convolution shapes.

A1 `pl.tile.move(tile, pl.MemorySpace.Mat)` also moves full FP16 ND Vec tiles into same-shape ZZ/NZ Mat storage directly through MTE3. Both dimensions are static multiples of 16 in [16, 2048]; UB/L1 capacity and liveness checks remain active. The move repacks C16 panels without GM staging or scalar UB reads. Its MTE3 classification lets the matching PTOAS path order vector producers, L1 consumers and UB/L1 reuse. FP32 sources, partial valid shapes and non-ND Vec layouts are rejected. Public native regressions cover vector-produced GEMM operands, changed inputs, reloaded artifacts, output guards, repeated local lifetimes and eight independent images in one task.

A1 `pl.tile.assemble(left, panel, [row, column])` supports full FP16 C16 Mat panels inserted into Left storage through MTE1. The Left target must reuse its result buffer in place so untouched values survive; K greater than 16 requires ZZ layout. Existing full-shape and memory-budget checks remain active. Matching PTOAS support proves aligned, bounded offsets, including affine loop offsets, and orders L1 producers, MTE1 reads, Cube consumers and buffer reuse. Unsupported memory pairs, wider source panels and fused quantization/activation are rejected. Native tests exercise partial-window preservation, fourfold panel replication, repeated lifetimes, changed inputs, artifact reload, output guards and eight independent images in one task.

A1 `pl.exp(tile)` and tile/tile `pl.div(lhs, rhs)` support full static ND FP16/FP32 Vec tiles with default precision. Division with `high_precision=True` is rejected. The native approximate arithmetic retains masked tails and repeat splitting. Tensor-form SiLU composes `zero - x`, `exp`, addition of an all-ones tensor and division; the native regression uses `@pl.jit.incore`, full compilation, public invocation and artifact reload. This validates an activation building block, not a complete YOLO model or portable IEEE edge-value semantics. Scalar division and other activation families remain unavailable.

A1 `pl.tile.full(shape, dtype, value)` fills static full ND FP16/FP32 Vec tiles from a compile-time scalar via `TEXPANDS`. SiLU can now materialize zero and one inside the kernel, removing the two host tensor inputs previously needed for these constants. Native tests cover both dtypes, masked repeats and public compile/call/reload. Runtime scalar arguments require separate support.

A1 exposes `pl.tile.row_sum`/`row_max` with full-size same-dtype scratch and `pl.tile.col_sum`/`col_max` for static full FP16/FP32 tiles. Row results preserve packed column-major `[M, 1]` layout through aligned contiguous stores; M must align the result to 32 bytes. Column results are `[1, N]`, and column sum currently rejects its optional binary-tree scratch form. Row buffers must be disjoint; column results may reuse the exact first input row, while partial overlap is rejected. Aggregate buffers must fit UB budgets. Arithmetic stays in the tile dtype; use FP32 for attention sum accumulation. Public native regressions check changed inputs, artifact reload and preservation outside aligned output partitions. Dynamic shapes, unaligned tails and special-value contracts remain unsupported.

Ascend310P additionally supports `tile.row_expand` from packed Mx1 reductions and `tile.col_expand` from row-major 1xN reductions into static aligned FP16/FP32 Vec tiles. Compose these with `sub`, `exp`, row/column `sum` and `div` for stable FP32 softmax. Row sources must be disjoint from the destination; column sources may share its exact first row. Fused broadcast arithmetic, dynamic and unaligned shapes remain unsupported.

The pinned A1 column broadcast uses raw vector copies with zero source repeat stride. It preserves FP16/FP32 signed zero, subnormals and NaN payloads, with bounded repeat groups and a wide-row fallback. Compile/reload regressions cover bit patterns and output guards without changing precision thresholds.

`pl.tile.maximum` supports matching full aligned FP16/FP32 row-major Vec operands through public A1 `TMAX`, with V-pipe dependencies and artifact reload. Either exact operand alias is supported; shifted output overlaps are rejected by the ISA. Finite values including subnormals are qualified. Broadcast forms and NaN selection require separate support.

`pl.tile.reshape` supports zero-copy full contiguous row-major FP16/FP32 Vec views with matching types and element counts. Both endpoints require static full valid shapes, a zero Tile view origin, unboxed storage and 32-byte-aligned rows within the UB budget. Padded or strided Tile views, matrix storage and dtype reinterpretation are rejected. Public compile/call/reload tests cover shape restoration, repeated lifetimes, input preservation and output guards with one or eight coupled blocks. Matching PTOAS/ISA support is required.

Ascend310P standalone `@pl.jit.extern(core_type="aic", source=...)` entries preserve their source/include metadata through specialization and compile to a coupled-core wrapper. Relative sibling includes remain rooted at the original external source. Public CPU-carrier invocation and artifact reload use the same wrapper path. The runtime pin also exposes explicit device-resident tensors and registered multi-kernel images; device operations remain synchronous.

`A1ResidentSession(compiled)` prepares a standalone Ascend310P artifact. Use `session.upload`/`empty` for device tensors and `session.prepare(compiled)` to bind each exact static signature. Calling a prepared kernel accepts only matching session-owned tensors, with optional `blocks=1..8`; it performs no implicit activation transfer. Tensor `copy_from`/`to_cpu` make input/output transfers explicit. Images with identical source/include content share a registration within the session. Close drains the stream before releasing storage. This is synchronous host dispatch, not device graph capture.

The native YOLO26 operator plans in `examples/models/yolo26_native/operators.py` add static FP16 NHWC convolution for kernel 1/3 and stride 1/2. Channel padding, on-device boundary gathering, FP32 cube accumulation, bias and optional SiLU run on Ascend310P; CPU weight packing is constant preprocessing. Output pixel counts must be multiples of sixteen. The separate native runner below provides model-level validation and timing.

`examples/models/yolo26_native/layout.py` adds native input FP32 NCHW-to-FP16 NHWC packing, aligned channel slicing/concatenation, residual addition, nearest 2x upsampling, 5x5 max pooling and dense FP16/FP32 conversions. Static row strides and offsets keep feature-map routing on 310P, including half-rounded attention query scaling. Focused resident tests use eight blocks, changed inputs, artifact reload and adjacent guards.

`examples/models/yolo26_native/depthwise.py` provides aligned NHWC FP16 3x3 depthwise convolution with padding one, stride 1/2, FP32 accumulation, bias and optional SiLU. Packed constant weights use [kernel-height,kernel-width,channels] order; spatial gathering and all activation arithmetic execute on 310P. Resident tests cover changing inputs, artifact reload and output guards.

`examples/models/yolo26_native/matrix.py` selects public PyPTO for the production non-transposed probability/value product `[400,400] @ [400,64]`. It uses M=80, N=16 work tiles and the original K=16 FP32 Cube accumulation order, retaining the flat four-argument ABI and one task for eight independent images. Changed-input, artifact-reload and output-guard regressions compare FP16 bit patterns against the native parent. QK, linear-accumulation and other geometries retain their native kernels; these stream 16x16 work tiles and fence vector-side accumulator reads before Cube reuse.

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

Depthwise convolution selects full 64/80/128/256-channel vectors when the
pixel grid can populate the launched cores; smaller grids and other channel
counts retain 16-channel panels. Channel-panel-major work keeps nine converted
weight vectors and bias in UB across pixels instead of reloading them per pixel.
The largest private weight/bias cache ends at UB byte 87040. Each lane retains
the same sequential or four-part FP32 reduction, padding and FP16 SiLU boundaries.
Tensor packing, device-task count and public signatures are unchanged.

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

Native detection TopK uses bounded 300-element worst-first heaps for streamed
anchor selection and class selection, then sorts retained candidates with A1
proposal instructions. The 16-element
`vbitsort` primitive and staged `vmrgsort4` merges replace scalar heap extraction.
FP32 scores remain unchanged; equal-score runs are ordered by ascending traversal
index because proposal sorting alone does not guarantee stable ties. Two private
16 KiB proposal banks occupy UB bytes 32768..65535 after metadata is consumed. Equal scores retain traversal order: anchor
order in stage one and selected-anchor rank/class order in stage two. Two
masked vector maxima reduce exactly the 80 class scores while excluding boxes
and padding. This replaces scalar scans and insertion shifts without changing
selected identities, scores, output geometry or the precision budget.

Detection TopK reuses its sorted anchor maxima as an upper bound: once the
next row's maximum cannot beat the current worst retained class candidate,
all remaining rows are skipped. The comparison includes traversal indices,
so equal-score ties keep the same exact two-stage semantics. For at most
384 anchors, one GM-to-UB transfer caches the dense [anchors,96] input for
maxima, class selection and final box gather. The cache spans UB bytes
65536..212991 at capacity and does not overlap heaps, metadata or output.
Larger inputs retain row DMA. Neither path changes values or uses host
selection; the single-block task and output tensor ABI are unchanged.

For at most 384 anchors, both class-maximum reductions batch rows with physical
stride 96 and split repeats at 255. One vector-to-scalar handoff replaces
per-anchor waits. All anchor maxima are proposal-sorted and equal-score groups
reordered before truncating to 300, preserving tied boundary membership. The
two proposal banks temporarily hold reduction results before sorting; larger
inputs retain the streamed heap path.

After the class heap is full, two FP32 vector comparisons identify row scores
strictly above its row-start threshold. Later traversal indices cannot replace
equal scores, and the threshold can only rise, so this is a safe candidate
superset. Scalar heap updates visit set bits in increasing class order and
recheck the current threshold. Initial heap filling retains the original scan.
The 64-bit masks exclude box coordinates and padding, with resident and streamed
rows both covered; no score narrowing or tolerance change is involved.

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
kernel when input channels are a multiple of 16 and at most 512. Each K16 fractal
has one nonzero weight row, so wider-K MAD batches add FP16 products in the
correct logical order to one of four FP32 partial sums. The ordered merge, bias and FP16 SiLU boundaries
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

Ordered 1x1 Cube batches increasing logical terms of each FP32 partial into
K64..K2048 MADs. The four replicas of each C0 input tile use M-major L0A
fractal storage; strided L1-to-L0B loads select every fourth sparse weight
fractal. The batch bound `8192 / Rows` keeps both L0 operands within 64 KiB
and reduces per-term instruction/fence overhead without changing the logical
sum order or compact GM weights. Every L1 weight read is drained before
MTE2 can replace an output panel. The final four-part merge and FP16
boundaries are unchanged; this local extern sequence does not widen the
generic A1 narrow-matrix Tile API.

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

For input channel counts divisible by four, ordered 3x3 Cube stores the nine
im2col patches in four phase groups: positions [0,4,8], [1,5], [2,6] and [3,7].
Each group uses M-major L0A fractals; strided L0B loads select the matching
sparse weights. Four K48/K32 MADs replace nine K16 MADs per channel without
adding MAC terms or expanding GM/L1 weights. The destination partial rotates
with the logical channel, preserving CHW summation. Scalar CHW tails retain
the previous path. All nine patches still fit 36 KiB L0A, and L0B needs at most
1536 bytes. Resident/streamed L1 lifetimes and FP16 boundaries are unchanged.

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

Paired pointwise output blocks share a complete input tile in L1 when pairing
preserves at least one work item per launched core. C0-major inputs occupy at
most 128 KiB below byte 262144; two sparse weight panels above that address
occupy at most 512 KiB. Each input tile is fetched once for two output blocks,
while each output keeps the same ordered MAD batches and FP16 epilogue.
A Vector-to-MTE2 fence drains previous bias reads before the shared bias
buffer is overwritten for the next output panel.
The existing M128/M64/M32/M16 selection is retained. Odd output-block counts
and grids that would underfill after pairing use the original single-panel
path. Source views, output guards, constant packing and explicit weight L2
bypass retain the same interfaces.

Resident ordered 3x3 convolutions launch at most one block per minimum M16/N16
output tile, capped at eight. If the grid is smaller, inactive blocks no longer
stage duplicate NC1HWC0 images and sparse-panel initialization. The chosen M16
arithmetic and the tensor/graph interfaces are unchanged. Streamed and vector
fallback schedules retain their existing block counts.

When the launched block stride is a multiple of the spatial tile count,
every task assigned to one core uses the same output pixels. Keep only that
core's input row band, including the 3x3 halo, in L1 and reuse the converted
NC1HWC0 band across its output-channel panels. Source-view strides and offsets
remain supported; irregular grids retain full-image staging, and the existing
large C16 per-task row-band schedule is unchanged. MAD order and FP16 boundaries
are identical to the full-image path.

Non-ordered 3x3 convolutions with 64 output pixels and C16-aligned input may
select dense K144 hardware im2col when the image and one/two dense N16 panels
fit 1 MiB L1. Dense GM weight packing is unchanged. Contiguous copies retain
the panel in L1; strided L0B loads match nine direct image-to-L0A patches.
There is no UB patch packing, sparse weight expansion or K-zero MAC work.
One FP32 accumulator replaces the four ordered sums, so this path is never
silently selected for the exact four-way policy. Bias/FP16 SiLU boundaries remain.

`NativeYOLO26(..., compact_box_head=True)` explicitly enables this path only in
supported box-head convolutions of the 128x128 ordered model; backbone, neck,
score and unsupported box convolutions retain their policies. This option is
off by default and requires `four_way_accumulation=True`. The profile entry
accepts `--compact-box-head --precision-policy pto2`. Existing generic convolution
and full-model validation tolerances are unchanged. Dense backbone/neck routing
is not enabled: it exceeded the strict model probability budget in hardware
experiments, even though isolated convolution checks passed.

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

Call `graph.rebuild()` after starting or stopping CANN profiling to refresh
the model's task profiling state. The YOLO26 executor exposes the same action
as `native.rebuild_graph()`. The graph handle, task order and resident bindings
remain fixed; no model arithmetic runs during rebuilding. Exclude rebuilding
and subsequent warmup from replay timing. CANN 9.1 pipe counters can remain
zero if a device model was prepared before profiling started. This API requires
a runtime pin exposing `A1PreparedSequence.rebuild`.

Native aligned COPY routing batches complete channel rows into occupancy-aware
panels instead of transferring one C16 pixel fragment at a time. Two 16 KiB UB
slots overlap MTE2 reads with MTE3 writes; ready/free events protect both slots
and are drained before return. The final panel may contain fewer rows. Channels
above 1024 or DMA gaps wider than 16 bits retain the original fragment path.
Source/destination offsets, surrounding channels and every FP16 bit are preserved.
This reduces routing transfer overhead without changing task count or concat ABI.

Grouped 3x3 Cube alternates two 2 KiB L0B slots. Each slot has an independent
MTE1-to-M ready event and M-to-MTE1 free event, allowing the next operand load
to overlap the current MAD. Both slots drain at each C0 boundary before shared
L0A patches change or an L1 weight bank is released. Existing resident L1
weight reuse and streamed 32-channel double buffering are retained. Spatial
tiles, physical MAC terms, four FP32 partials and FP16 boundaries are unchanged.

For grouped M16 spatial tiles, sixteen im2col patches follow the period-nine
sequence `4*j % 9`. Four overlapping nine-fractal windows starting at 0/7/5/3
supply K144 MADs for four logical channels at a time. Each window retains one
nonzero product per K16 fractal and increasing CHW order for its FP32 partial;
a full C0 uses 16 MADs instead of 64, with unchanged physical MAC terms.
The windows share 8 KiB L0A; two 8 KiB L0B slots hold their weight operands.
L1 resident/streamed weights and compact GM packing remain unchanged. Larger
M tiles and scalar CHW tails retain their existing paths.

The native YOLO26 executor fuses concat with two to four aligned source views
into one device task when total channels are at most 4096. Every source is a
formal read tensor so resident-graph dependency analysis retains its producer.
Two 64 KiB UB banks assemble up to sixteen full output rows; strided MTE2 loads
write each source directly to its final UB columns, followed by one contiguous
MTE3 store. Independent ready/free events overlap adjacent panels. Source views
may share an allocation; exact FP16 bits and channel order are preserved.
Larger fan-in, channel totals or DMA gaps retain per-source COPY tasks. This is
concat task fusion with physical copying, not producer-side zero-copy storage.

Streamed spatial Cube also accepts a 4x4 image/output with one padded
output-channel tile and complete channel quartets. This detection-head case
uses the tested M16 K144 windows even though only one core has work.
Two/three-tile fallbacks, other geometry thresholds and the 1 MiB L1 bound
remain unchanged; small shapes need not fill eight cores to benefit from Cube.

Spatial Cube can pair adjacent N16 output panels in an M16/N32 tile, sharing
one im2col operand across both panels. The bounded path requires at most 64 output pixels, at least 128 padded
output channels, grouped channel quartets, an even output-panel count and at
least eight work tiles. Each output retains its ordered K144 operands and four FP32 partials.
Two L0B slots reserve 16 KiB each; L1 holds two resident sparse panels or two
32-channel banks per output panel. The dispatcher bounds image plus both
panels to 1 MiB and keeps the N16 path for other shapes. Physical sparse MAC
terms are unchanged; operand generation and MAD instruction overhead shrink.

Spatial input staging with at least two C0 blocks reads complete aligned NHWC
channel rows into two 32 KiB UB banks before local MTE3 scatter to resident
NC1HWC0 L1. Panels adapt to channel count and retain short final rows, strided
source views and offsets. Separate ready/free events prevent bank reuse while
MTE3 reads it; both banks drain before im2col. Single-C0 inputs retain the direct
GM-to-L1 path. UB staging ends before the unchanged vector epilogue begins.

For single-C0, single-output-panel spatial convolution with input height at
least 32, each work tile loads only its input row band and kernel halo.
The L1 image descriptor uses the band's height, and im2col subtracts its
clipped global row origin. This preserves top/bottom padding and source views
without replicating the complete image in every core. Weight-cache addresses,
ordered partials, FP16 boundaries and other shapes remain unchanged.

Resident detection TopK reuses the existing vcmax value/index pairs to seed
the final heap with one real class maximum from
each of the 300 selected anchors. This raises its initial cutoff without
changing candidate eligibility. Vector comparison includes equality because
a future-anchor seed can lose to an earlier traversal index. Seeded pairs are
not inserted twice; heap replacement and final proposal sorting keep the
existing exact FP32 and tie order. The streaming path is unchanged.

N32 small-grid convolution uses M32/K144 when at least eight work tiles
remain. Four independent M-major im2col windows occupy 36 KiB of L0A,
so one L1-to-L0B K144 load serves two spatial fragments. Other eligible
grids retain the M16 shared-window path. Resident/streamed weight banks,
source views, padded output columns and the four ordered sums stay exact.

Larger ordered spatial grids can also share M16 im2col operands across N32
output panels. This reuses sixteen overlapping K144 patches and alternates
two L0B banks while retaining the four FP32 sums in CHW order. The dispatch
requires at least C32, paired output panels, sufficient work for active cores,
and a bounded image window plus both sparse weight panels within 1 MiB of L1.
Low-channel full images that already exceed the single-panel L1 budget keep
their larger M tiles and existing input reuse; blindly selecting M16 increases
staging and weight traffic on those shapes. Dense reduction and the existing
small-grid dispatch are unchanged.

Large ordered 3x3 convolutions can use bounded input row windows when a full
image plus the sparse weight banks exceeds L1. The planner budgets the
worst-case halo for up to 128 consecutive output pixels, aligned to 512 bytes,
and retains the 1 MiB image-plus-weight limit. Each work tile stages only its
clipped input rows in NC1HWC0, updates the local image descriptor and subtracts
the global row origin from hardware im2col coordinates. Multi-C0 input uses
the existing two UB staging banks. Logical CHW order, four FP32 partials,
source slices, output guards and FP16 boundaries are unchanged.

Windowed M128 work with at most 64 padded input channels assigns contiguous
spatial stripes to cores when there are multiple output panels or the output
width is at least 256. It masks the final partial stripe and consumes all
output-channel panels before advancing to the next tile. The clipped row
bounds key the L1 input cache, so adjacent tiles with the same halo can reuse
it too. Weight panels reload when their channel changes. Wider input channels,
smaller M tiles and narrow single-panel grids keep the weight-resident traversal
to avoid extra sparse-weight loads. Full-image paths are unchanged.

Residual addition uses aligned full-channel row panels when channels are at
most 1024 and all three DMA gaps fit 16 bits. Two 8 KiB slots per operand
pipeline input DMA, FP16 vector addition and output DMA without touching
metadata. Tail masks cover only actual elements; output slice gaps stay
untouched. Wider channels and unsupported gaps retain the C16 fallback.

The remaining FP16 convolution SiLU epilogues refine the sigmoid for
`abs(x) < 1/16` using a degree-five exponential polynomial. This avoids A1
exponential approximation moving the FP16 sigmoid rounding boundary near zero.
The clamped polynomial and a binary mask over widened FP16 inputs are evaluated
across whole vector tiles. Other inputs retain the hardware sigmoid result.
The helper reuses dead epilogue scratch and preserves separate FP16
sigmoid/product rounding; it introduces no tolerance changes or CPU work.
All finite FP16 input patterns are checked through the real convolution and
depthwise kernels. External plans register the shared header include directory.

Cube convolution bias epilogues issue one repeated C16 vector add per NZ
column panel. A zero bias-source repeat stride reuses the same sixteen FP32
values across M rows, eliminating per-row mask setup and scalar loop control.
The arithmetic, FP16 narrowing and fused SiLU boundaries are unchanged.

Full-channel depthwise panels retain a three-row NHWC input ring in the upper
128 KiB of UB when it fits. Each core traverses one contiguous output stripe,
loads its own halo, and reuses input rows across adjacent pixels. Converted
weights and bias remain resident; the four FP32 partial sums and SiLU rounding
are unchanged. Oversized rows and channel-panel geometries keep the prior path.

Bounded 5x5 max pooling retains a compact full-channel input map in UB when
it occupies at most 128 KiB and channels are at most 1024. Input DMA chunks
respect the 4095-row burst limit and source slice gaps. Full-channel vector
maxima reuse the resident map, retaining the original ky/kx comparison order,
finite padding value and output slice boundaries. Metadata and result scratch
remain below the input cache. Larger maps or unsupported source DMA gaps keep
the C16 fallback; pooling does not change values or introduce host arithmetic.

Nearest 2x upsampling stages one complete input-channel row, duplicates it
with integer vector OR and reuses the resulting pair for both output rows.
Two input/pair UB slots pipeline input DMA and output stores. This path
requires at most 1024 channels and a representable destination DMA gap;
larger channels or gaps retain C16 routing. Integer copies preserve every
FP16 bit, including signed zeros, infinities and NaN payloads, while strided
source/destination slices and their untouched gaps remain intact.

Full-channel depthwise convolution processes groups of four adjacent interior
pixels as independent vector lanes for C64, C80 and C128 stride-one maps that
fit the resident three-row ring. Replicated FP32 weights and bias remain in
bounded UB caches, preserving each pixel's four-part accumulation and FP16
SiLU rounding boundaries. A separate scalar pass owns boundary columns and
short right tails, retaining skipped invalid taps rather than introducing
zero-filled products. Oversized rows and other channel/stride geometries
retain their previous path.

Streamed N32 spatial convolution consumes all output-channel panels before
replacing each bounded L1 input halo. Its two weight banks already reload
for every work tile in either traversal, so this schedule saves input staging
without adding weight transfers. Per-core spatial stripes mask ragged groups;
four ordered FP32 partial sums, im2col geometry and FP16 activation boundaries
remain unchanged. Resident-weight paths retain their existing reuse policy.

Input packing stages 256 NCHW pixels per channel, performs the unchanged FP32
to FP16 cast, and interleaves sixteen C16/M16 fragments with one repeated
transpose. Two 8 KiB half banks and a 16 KiB float bank end below metadata at
32 KiB. Zero-initialized padded channel rows survive each panel. A V-to-MTE2
fence protects float-bank reuse; the output fence protects transpose-buffer
reuse. Small core grids and non-256 pixel tails keep the existing C16 path.

Streaming detection TopK stages at most 255 dense rows per input chunk and
reduces their 80 class scores with repeated vector maxima. Box coordinates
and padded fields stay masked. Scalar heap updates consume the maxima in the
original anchor order, preserving FP32 scores and exact tie traversal. The
bounded chunk and reduction buffers finish before reuse; selected-row class
selection, proposal sorting and final gathering retain their prior semantics.

Paired ordered pointwise output panels share one N32 Cube operation rather
than rebuilding L0A and issuing separate N16 MADs. The second weight fractal
is interleaved into K-major L0B, and NZ column blocks stay adjacent through
bias, activation and strided output DMA. The K batch bounds both operand
buffers to 64 KiB; four partial accumulators occupy at most 64 KiB and vector
scratch slots at most 16 KiB. Logical K order and FP16 narrowing are unchanged;
unpaired output geometries and small work grids retain the N16 path.

Large-grid ordered 3x3 convolution shares the existing M64/M128 K32/K48
im2col windows across N32 output columns. Phase rotation keeps every CHW
term in its original FP32 partial and order. L1 selection accounts for both
weight columns, retaining resident or double-buffered streamed weights and
bounded input halos; unsupported or small grids keep earlier paths. Four
L0C partials occupy at most 64 KiB, with vector scratch slots at most 16 KiB.
The A1 external entry is forced inline because single-launch loads the
compiler object without a device link step; an outlined wrapper call would
otherwise leave an executable relocation. FP16 activation is unchanged.

Detection decode groups eight contiguous anchors per sigmoid/transfer panel
when the pixel count is divisible by eight and the grid fills every core.
The same FP32 sigmoid instructions, FP16 probability narrowing, scalar FP16
box arithmetic and dense [anchor,96] layout are preserved. Live panel buffers
end below metadata at 32 KiB. A completion fence precedes buffer reuse;
small and ragged pixel grids retain the original one-anchor path.

Large-M N32 spatial convolution consumes four-channel groups in phase-major
order. One period-nine K144 hardware-im2col window remains in L0A while all
groups for that FP32 partial reuse it; M128 occupies 36 KiB rather than four
simultaneous windows. Both MAD slots finish before replacing A, and two
16 KiB L0B slots alternate while Cube consumes weights. Each partial retains
its increasing CHW term order. The existing L1 double buffer is released only
after all four phases have finished; FP16 activation and fallback paths remain.

Ordered pointwise Cube SiLU uses an exact single-boundary correction instead
of the general near-zero exponential polynomial. At widened FP16 input
`-15/2048`, A1 vexp is one FP32 ULP low near one; adding `2^-23` before the
reciprocal preserves the CPU FP16 sigmoid rounding tie. An arithmetic mask
selects only that value: neighboring FP16 inputs are at least `2^-18` away.
Every other finite input retains the hardware exponential, and the separate
FP16 sigmoid/product casts are unchanged. Exhaustive checks cover all 63,488
finite patterns for N16 and N32, multiple output panel pairs, and one/eight
cores; nonfinite bias cases check infinity, NaN and signed-zero behavior.
Other convolution and depthwise kernels retain the general correction.

Ordered N32 spatial convolution can use M128 tiles for grids containing a
whole number of M16 fragments but a partial larger tile. Input channels must
be divisible by four, padded channels at least 64, output channels at least
128, and the grid at least 256 pixels with enough panel work for the launch.
The existing L1 capacity checks still choose resident or streamed weights.
Inactive M16 fragments issue no im2col reads and only valid rows are stored.
Each active M row retains its four ordered FP32 partials. Tail predicates are
compiled only into the ragged specialization; existing full-tile loops remain
constant. This avoids forcing 400-pixel detection grids down to M16, reducing
repeated sparse L0B loads without changing FP16 activation or tolerances.

`worker.prepare_independent(compiled, instances)` binds two to eight independent
single-core invocations to physical cores; one instance reuses `prepare`.
Flatten complete signatures in instance-major order and use `blocks=instances`.
Each body sees logical grid `(0, 1)`; the physical core selects its own canonical
tensor descriptors. The flattened signature must contain at most 64 tensors.
Shared read-only inputs are allowed; outputs and scratch must be independent.
The kernel must be correct on one core and must not communicate between instances
or read physical core identity inside precompiled/header helpers. This API makes
no tensor-layout, arithmetic or precision-policy changes. Count, signature,
ownership and mismatched block-count failures are rejected before graph binding.
Registered images are cached separately by source/includes, arity and instance
count; preparation and compilation remain outside measured replay.

`NativeYOLO26Batch(model, 640, output_dir, batch_size=8)` records the production
batch on one physical 310P chip. It uses one independent image per coupled core,
shared immutable weights and separate activation storage. Layer-major order
preserves every image's dependencies. Each native or generated operation binds
eight complete signatures through `prepare_independent`; 287 positions become
287 device tasks in one model submission, rather than eight repeated traces.
Use `copy_input` and `detections_cpu` for explicit transfers outside replay;
`replay` itself performs no host tensor arithmetic. `rebuild_graph` refreshes
profiling state outside timing. The original single-image executor remains
available for strict intermediate checks. This batch API currently requires
640x640 batch eight and retains ordered convolution, precise small softmax,
original tensor layouts and exact native TopK, without tolerance changes.

Spatial Cube SiLU shares the pointwise single-input exponent correction in
`fp16_activation.hpp`, replacing the general near-zero polynomial only in
these two ordered epilogues. The exact mask adds 2^-23 to `exp(-x)` at the
finite FP16 input -15/2048. Reciprocal, separately rounded FP16 sigmoid and
product, bias and four FP32 partials are unchanged. Other consumers retain
the original general refinement helper. Exhaustive spatial N32 checks cover
all 63,488 finite patterns with one/eight cores, both output panels and panel
replacement; nonfinite post-Cube bias checks retain NaN and signed-zero behavior.

Decode copies the 80 already-computed FP32 class probabilities with aligned
64-bit scalar loads/stores and clears padding the same way. The class field
starts 16 bytes into a 96-float row, so a normal 32-byte-aligned vector copy
cannot be substituted directly. Unrolling packed copies reduces scalar-loop
work without changing sigmoid, FP16 narrowing, boxes, output layout or any
probability bits. Finite-FP16 score coverage checks both class-field edges,
padding, output row guards and one/eight-core dispatch under the existing budget.

Large M64/M128 spatial N16 kernels now share the phase-major K144 windows
already used by N32. For input channel counts divisible by four, each full C16 block
issues sixteen K144 MADs instead of sixty-four K32/K48 MADs, retaining the
increasing CHW term order within each of the four FP32 partials. One resident
L0A window and alternating L0B slots retain their existing bounds and fences.
Non-divisible channel counts keep the CHW-tail implementation. This reduces
operand setup and instruction overhead; it does not remove the ordered sparse
K padding. N16 checks cover partial C0, streamed weights, stride-two halos,
source views, reload and one/eight-core dispatch under exact existing checks.

Short spatial grids that would otherwise use M16 can use bounded M64/N32
ragged tiles: 256–512 whole-M16 pixels, 64 padded output channels, at least
64 padded input channels, actual input channels divisible by four, and valid
L1/core guards.
This covers the production 400-pixel co64 sites that otherwise retain M16.
Only live M16 fragments issue im2col reads and only valid rows are stored;
the existing four ordered partials and FP16 SiLU boundaries remain unchanged.
Other channel counts and grids outside this interval retain their prior paths.
Exact tests cover stride-one/two, partial channels, dispatch bounds, CHW-tail
fallback, reload, source/output guards and one/eight-core activation.

Single-core pointwise bodies can retain a complete NC1HWC0 input in L1 across
all output-channel panels when the rounded input and both sparse weight panels
fit in 1 MiB. Coalesced NHWC row reads and double-buffered UB staging reuse the
same input helper as spatial convolution. Independent-image batch dispatch
still uses all eight coupled cores on one chip. The selected body retains the
MTE2-to-MTE1 weight-ready edge after every weight replacement; removing input
copies must not remove that synchronization. Other shapes/core grids retain
the existing tile loader. Four ordered FP32 partials, activation boundaries,
source/output views and all precision budgets remain unchanged.

Depthwise SiLU now shares the Cube single-boundary exponent correction. It
retains all nine-tap accumulation, resident row/panel schedules and separate
FP16 sigmoid/product casts. Exhaustive finite-input checks cover C16 and
C64/C80/C128 panel, boundary and tail paths at one/eight blocks; C80 exercises
partial vector masks. No precision tolerance or host fallback changes.

Three-input-channel spatial Cube convolution groups the first 24 CHW terms
into four ordered K96 MADs, then appends the final three terms to partial zero
with K48. Each K16 keeps one useful weight, preserving sparse products and
every FP32 partial order. This replaces 27 MAD/fence pairs with five, while
assembling phase-major im2col windows. Existing tile, halo, view, bias and
activation behavior stays unchanged; other channel counts retain their path.

Ordered spatial convolution uses native Load3D M-mode repeats for adjacent
M16 fragments. The destination jump follows each phase-major K layout; the
hardware advances the sliding-window coordinates through row wraps. Whole-M16
ragged tails clip the repeat count. Loaded values, K order, partial sums and
all precision boundaries stay unchanged; compact convolution retains its loop.

Short pointwise grids with 256..512 whole-M16 pixels, aligned input channels
64..512 and padded output channels >=64 with an even N16 panel count use
bounded M128 tails. Only live M16
fragments load and store; inactive rows do not change any live reduction.
This amortizes sparse B reloads at the cost of inactive MAC rows. Retain
whole-input L1 residency when the one-core body fits its existing bound,
and the tile loader otherwise. Other grids retain their prior dispatch;
all four FP32 partials and separate FP16 activation boundaries are unchanged.

Occupied large N32 spatial grids may use M224 tiles: at least 256 whole-M16
pixels, padded input channels >=64 and at most 15% inactive rows. Existing
source-view, core-count and image-plus-two-weight-bank L1 bounds still apply.
A K144 patch uses 64512 bytes of L0A. Four 32 KiB partial slots preserve the
ordered accumulation; after merging, dead UB partial slots become activation
scratch, correction mask and rounded output. The bias ends at byte 130688,
below metadata at 131072. Restore partial views before each subsequent tile.
Windowed C64 and streamed N32 paths retain their spatial-first halo reuse.
Other grids retain prior dispatch; this reduces repeated B transfers without
changing sparse K arithmetic, activation rounding or precision tolerances.

Contiguous NHWC input staging uses one GM burst per bounded panel instead of
one burst per pixel. Multi-C0 UB panels contain at most 16384 FP16 elements;
single-C0 direct-to-L1 chunks retain the 4095-block limit. Both flattened
lengths fit the hardware descriptor. Strided source views keep their prior
row-gather descriptors. Bytes, NC1HWC0 layout, double-buffer ownership and
all readiness fences remain unchanged. This shared helper serves spatial
convolution and resident pointwise inputs without changing arithmetic.

Ordered pointwise convolution pipelines B operands for M64/M128. Two 32 KiB
L0B slots alternate across the four independent FP32 partials while sharing
the same A batch. Each slot waits for its previous MAD before replacement;
both drain before A changes or L1 weights are released. Smaller M retains
the single-slot loop because its larger K batch can occupy all 64 KiB L0B.
K order, partial merge, L1 input residency, clipped tails and activation
boundaries remain unchanged; no precision budget changes.

Native decode assembles complete 96-float records with m200 U16 gather,
copying both halves of each FP32 score bit pattern. A static byte-offset
table spans coordinates, the 16-byte-shifted score region and zero padding;
scalar coordinate writes then replace the first four fields. At eight rows,
the 1536-entry table ends at byte 22528, below output at 24576 and metadata
at 32768. One-row tails mask the final 64 halfwords. Coordinate arithmetic,
FP16 sigmoid narrowing, output ABI and all precision budgets are unchanged.

Depthwise convolution enumerates only boundary columns and the incomplete
right panel in its scalar pass, rather than scanning pixels already owned by
the four-pixel interior pass. Compact work indices map back to original NHWC
coordinates before row-cache lookup. Channel ownership, valid-tap checks,
four-part/sequential arithmetic and FP16 activation boundaries stay unchanged.

Depthwise 80/40/20-column grids use bounded wider panels: C64/P13 at width 80,
C80/P12 at widths 80/40, C128/P6 at widths 40/20, and C256/P3 at width 20.
Other shapes retain their existing paths. Up to 1024 FP32 lanes fit separate
4 KiB scratch spans; replicated weights end below byte 126976, bias below
61440, metadata remains at 32768, and resident rows begin at 131072. Tap order,
scalar invalid-tap skips, activation narrowing and precision budgets are unchanged.

Ordered pointwise convolution keeps the current output panel's bias in its
existing private UB span across spatial tiles. A panel change waits for the
previous vector consumer before loading replacement bias; every kernel replay
starts with an empty cache. Input staging, four partials and activation scratch
do not alias this span. Bias addition, K order, FP16 narrowing and precision
budgets are unchanged.

The FP16 sigmoid exponent correction uses packed comparison bits and native
FP16 tensor-select on complete FP32 repeats. Its exactly representable 2^-23
FP16 delta widens before the existing divide and sigmoid/product narrowing.
A masked final FP16 repeat stays within private scratch; partial FP32 repeats
retain the distance/clamp mask. Finite/nonfinite behavior, arithmetic boundaries
and all precision budgets remain unchanged.

The 400-position attention softmax retains the original 16-row FP32
reduction shape inside a native device loop and fuses its FP16 input/output
casts. One task replaces the previous 27-task chain, without materializing
two FP32 score matrices in GM. Cores own disjoint row blocks; the batch worker
binds independent images to cores. The sixteen-position compensated path and
all numerical budgets remain unchanged.

Ordered N32 3x3 convolution uses M112 tiles when the existing occupancy,
view and L1 guards allow them. Two disjoint 32 KiB L0A slots overlap the
next K144 phase's im2col transfer with the current phase's Cube work;
existing L0B slots retain their separate ownership protocol. Every phase
releases A after its last MAD and drains both slots before leaving the
channel block. Four-part K order, live-row clipping and FP16 boundaries
remain unchanged; private activation scratch stays within its existing span.
