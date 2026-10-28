# Native YOLO26 chip observations on Ascend310P

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
