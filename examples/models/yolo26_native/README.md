# YOLO26 on Ascend310P

## Native model execution

`executor.py` provides the single-image 128x128/640x640 executor.
`executor_batch.py` provides `NativeYOLO26Batch` for 640x640, batch eight,
on one physical Ascend310P chip. It shares immutable weights and compiled
artifacts while keeping each image's activations separate. Replay uses one
static device model; model arithmetic runs on the device. The batch wrapper
selects four-way accumulation and the precise small-Softmax profile.

`profile.py` validates single-image runs against the CPU FP16 oracle and
reports synchronous host replay timings. Use matched low-overhead msprof
captures for device-compute comparisons and actual batch replay for batch
results. Keep feature, box, probability, score and exact TopK checks enabled.
Dynamic AICPU ready-queue scheduling requires separate runtime support.

## Chip observations

This standalone 128x128, batch-one diagnostic uses the existing YOLO26n
weight/graph builder and external C++ operator bodies. It combines the
bodies into one persistent AICore entry with an AICPU observer and a
sequential task barrier. It leaves the default device-model replay unchanged.

After installing PyPTO and its pinned runtime/ISA, source the CANN environment
and use an aarch64 host with Ascend310P3. Select an idle device through the
usual device isolation/queue workflow. Supply the exact input and a saved
31-array reference from the strictly validated default replay:

```bash
python -m examples.models.yolo26_native.chip_profile \
  --weights /path/to/yolo26n.pt --input /path/to/input128.npy \
  --reference /path/to/validated-readback --output /path/to/new-capture
```

The output contains raw versioned captures, physical-core Perfetto traces,
critical-path reports, 31 exact-array readbacks per measured round, and the
existing strict CPU-oracle checks for the supplied input and six fixed seeds.
Probability checks retain rtol=0.03 and atol=1e-6; TopK retains exact semantics.
A mismatch fails the command. Model arithmetic during replay stays on 310P.

The private JIT source-path and resident-buffer adapters are diagnostic
integration points, not a general replacement for the runtime's public
session API. The support libraries are built from the current runtime's
common loader/dispatcher and native A1 observer sources; no A2/A3 device code
or prebuilt library is required.

Timestamps come from the queried AICPU clock. Core bars are observed
receipt/release/completion envelopes, including polling/coherence/barrier
costs, not instruction timestamps or pure compute. `early_dispatch` shows a
real queue-publication phase; it does not prove core-side descriptor prefetch.
The critical path describes this sequential diagnostic implementation.
Use separate matched low-overhead msprof captures for CANN speed comparisons.
No elapsed-time-to-FPS conversion is made for this diagnostic.

## Public classifier projections

The 640x640 classifier uses public PyPTO for its three final, non-activated
80-to-80 pointwise projections. An M80/K80/N16 kernel retains compact constant
weights in L1 and reuses Left across the five output-channel panels. Dense FP32
reduction replaces sparse four-part reduction for this bounded profile; all
existing CPU FP16 feature, box, probability and score budgets and exact TopK
checks remain required. Other geometries, sliced inputs and L2-alias launches
retain their existing convolution profiles. No model-local C++ entry is added.

## Public FP16 activation

`pypto.library.fp16_silu(rows)` returns a public PyPTO inline helper for finite
FP16 M64/M128/M160/M256 by N16 tiles. It rounds the sigmoid to FP16 before
multiplying and narrows the product separately. An exact finite-domain
squared-distance clamp corrects the hardware exponent boundary at -15/2048.
Every other finite FP16 input is at least 2^-18 away, making
`max(0, 2^-23 - 8192 * distance^2)` exactly zero there.
All 63,488 finite FP16 patterns, signed zeros, reordered inputs, artifact
reload, output guards and one/eight-core grids retain bitwise CPU checks.
Convolution callers are migrated and qualified separately.
The remaining native ordered/depthwise callers share one boundary-SiLU
epilogue in `fp16_activation.hpp`; its explicit rounding and pipeline order
are common to all three callers. Legacy polynomial paths remain separate.

`pypto.library.fp16_silu_constants(rows)` shares the same inline calculation but
accepts five caller-owned FP32 constant tiles: zero, offset 15/2048, scale
8192, one, and epsilon 2^-23. Allocate them outside a spatial loop to retain
values in UB and avoid repeated fills; this uses `5 * rows * 16 * 4` bytes
in addition to live work buffers. The Tile parameters remain local to
the caller and add no GM dispatch arguments. Both helpers support M160,
which permits a 60 KiB ordered K192 Left tile.

## Public ordered pointwise convolution

The 160x160 CI48/CO64 1x1 SiLU profile uses public PyPTO with M160/N16 Cube
panels. A 60 KiB K192 Left operand is assembled once per spatial tile and
shared across all four output panels. Expanded ordered weights stay in L1;
five activation constants stay in UB. Four scalar-partial reduction order,
FP16 sigmoid/product rounding and the six-tensor dispatch ABI are retained.
Other geometries, source slices and L2-alias launches retain existing plans.

The 160x160 CI32/CO32 1x1 SiLU profile also uses public PyPTO. Four ordered
K128 partials use N32 Cube operations, then explicit N16 accumulator panels
preserve the original reduction and activation boundaries. The batch graph's
single-core-per-image path pipelines spatial tiles; other core counts use a
complete strided fallback. Source slices and L2-alias launches remain on the
existing native plan. This profile uses no external C++ kernel.

## Public Softmax algorithms

Both attention Softmax profiles use `pypto.library` inline algorithms. The
16-column profile retains compensated FP32 scalar arithmetic and exact FP16
regressions; the 400-column profile keeps the original vector reduction shape
and FP16 boundaries. Neither profile registers an external C++ source.
The chip diagnostic compiles generated entries through the same backend,
so it does not require a separate handwritten Softmax implementation.

## Pointwise plan reuse

The public ordered pointwise plans share resident-view validation. The CI32
and CI64 N32 profiles also share one packed-weight implementation and the
existing kernel factory, while retaining their separate geometry checks.
The fixed CI48 and CI32/CI64 kernels also share one inline factory for the
four ordered N32 Cube partials and N16 writeback panels. Only tile dimensions
are specialized; reduction order, activation and memory placement remain with
each caller. Plan names, metadata, packed buffers and generated device
arithmetic stay unchanged; this is a code cleanup.

## Experimental public residual ADD

`PublicAddYOLO26` inherits the existing single-image constructor and selects
public FP16 ADD for residual routes. `NativeYOLO26Batch` exposes
`experimental_public_add=True` for the same experiment at 640x640, batch eight.
The default remains the existing implementation until strict device precision
and matched performance qualification pass.

This path reuses `pl.load`, `pl.add`, `pl.store` and `pl.pipeline(stage=2)`.
The single logical block per image uses two-stage panels; multiple logical
blocks own disjoint row panels, and block zero owns any final partial panel.
Source, second-input and destination channel strides/offsets remain independent.
It admits up to 1024 aligned channels and 16-bit DMA row gaps. The existing
metadata and four-tensor ABI stay intact. SCALE and other vector routes retain
their existing implementation; SCALE requires its FP32 multiply boundary.

## Experimental public decode

`PublicDecodeYOLO26` uses public PyPTO operations for box/sigmoid decode.
`NativeYOLO26Batch(..., experimental_public_decode=True)` enables the same
candidate at 640x640, batch eight, including with SPMD attention packing or
experimental public residual ADD. Both default executors keep native decode
until real-device precision and performance qualification is complete.

The candidate preserves the sigmoid FP16 narrowing and every native coordinate
FP16 rounding step. It uses existing tile arithmetic, casts, scalar access,
assembly and indexed gather; no model C++ kernel or new public operator is
introduced. The four-argument ABI carries INT32 immutable index metadata;
setup shares this table across all batch images and feature levels. Index DMA
is outside panel loops, and full eight-row panels fall back to one-row work
when the logical core grid is too large. Activations remain independent, with
unchanged task ordering and count.

Generated-wrapper CPU bitwise tests and dav-m200 offline compilation cover
small/odd grids, 128-shape decode and production 80/40/20 feature levels, with
nonzero Tensor start offsets, output guards and read-only inputs. They do not
establish device precision, asynchronous timing or model performance. The
candidate currently adds aligned UB packing and box conversion; those costs
must be measured against native decode before selecting a default. The live
native implementation remains required by the default path.

## Experimental public attention conversions

`PublicConversionYOLO26` selects public FP32-to-FP16, FP16-to-FP32 and scaled
FP16 conversion. Batch eight exposes `experimental_public_conversion=True`.
The existing metadata and four-tensor ABI remain unchanged. FP16 inputs widen
to FP32 before multiplying by the FP32 metadata scale; scaled FP16 output is
narrowed only after that multiply. The conversion-to-FP32 path also retains
the multiply, including its default scale of one.

The implementation uses existing public `cast`, scalar read, `mul`, load/store
and `pipeline(stage=2)` operations. Full 4096-element panels reduce repeated
DMA setup for production 12,800-element query vectors; 256-element-aligned
tails remain bounded. A1 retains replicated static double buffers under all
memory planners because dynamic multi-slot selection is not yet supported.
These are generation properties, not measured device overlap or speedups.
SPMD attention packing already fuses the same scale into QKV routing; the
optional conversion path preserves that fusion and its task count.

The native default remains until strict device precision and matched latency
are qualified. CPU raw-bit checks cover all finite FP16 bit patterns, FP32
half-rounding midpoints, signed zeros, scale signs, read-only inputs, output
guards and nonzero Tensor start offsets. All three planners compile offline
for dav-m200. Host graph checks do not execute whole-model numerics.
