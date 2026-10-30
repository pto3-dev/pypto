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
Plan names, metadata, packed buffers and generated device arithmetic stay
unchanged; this is a host-side code cleanup.
