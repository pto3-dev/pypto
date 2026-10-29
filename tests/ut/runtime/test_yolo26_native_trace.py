# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep native YOLO26 replay free from silent host-operator fallbacks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.models.yolo26_native.executor import (
    Feature,
    Launch,
    NativeYOLO26,
    RejectHostOperators,
    _runtime_l2_cache_offset,
)
from examples.models.yolo26_native.operators import ConvPlan
from examples.models.yolo26_native.pointwise import PublicOrderedPointwisePlan, select_ordered_pointwise
from examples.models.yolo26_native.precision import FP16SiLU, prepare_fp16_model
from examples.models.yolo26_native.projection import ClassifierProjectionPlan, select_classifier_projection
from examples.models.yolo26_native.softmax import Softmax16Plan, Softmax400Plan


def test_native_trace_allows_host_dispatch_without_tensor_arithmetic():
    seen = []

    def kernel(value: Any, *, blocks: int) -> None:
        seen.append((value, blocks))

    native = NativeYOLO26.__new__(NativeYOLO26)
    native.trace = [Launch(kernel, ("resident-tensor",), 8, "test")]
    guard = RejectHostOperators()
    with guard:
        native.replay()
    assert seen == [("resident-tensor", 8)]
    assert guard.calls == 0


def test_native_replay_uses_registered_graph_without_per_task_dispatch():
    seen = []
    native = NativeYOLO26.__new__(NativeYOLO26)
    native._sequence = lambda: seen.append("device-model")
    native.trace = [
        Launch(lambda **_kwargs: pytest.fail("per-task dispatch is forbidden"), (), 8, "forbidden")
    ] * 216
    with RejectHostOperators():
        native.replay()
    assert seen == ["device-model"]


def test_native_trace_rejects_host_tensor_arithmetic():
    tensor = torch.ones(4)

    def kernel(*, blocks: int) -> None:
        torch.add(tensor, blocks)

    native = NativeYOLO26.__new__(NativeYOLO26)
    native.trace = [Launch(kernel, (), 1, "invalid-fallback")]
    guard = RejectHostOperators()
    with pytest.raises(RuntimeError, match="host ATen operation"), guard:
        native.replay()
    assert guard.calls == 1


def test_native_unknown_module_fails_without_calling_cpu_forward():
    class Unsupported(torch.nn.Module):
        def forward(self, _value):
            raise AssertionError("CPU forward must never run")

    native = NativeYOLO26.__new__(NativeYOLO26)
    with pytest.raises(ValueError, match="unsupported native YOLO26 module"):
        native._module(Unsupported(), None)


def test_fp16_silu_preserves_sigmoid_narrowing_boundary():
    x = torch.linspace(-8, 8, 4096).half()
    expected = (x.float() * torch.sigmoid(x.float()).half().float()).half()
    actual = FP16SiLU()(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.any(actual != torch.nn.functional.silu(x))


@pytest.mark.parametrize("affine,tracked", [(False, True), (True, False)])
def test_fp16_preparation_rejects_batchnorm_without_constants(affine, tracked):
    class Invalid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(16, 16, 1)
            self.bn = torch.nn.BatchNorm2d(16, affine=affine, track_running_stats=tracked)

        def fuse(self, *, verbose):
            raise AssertionError("invalid BatchNorm must fail before fusion")

    with pytest.raises(ValueError, match="affine BatchNorm with tracked"):
        prepare_fp16_model(Invalid().eval())


@pytest.mark.parametrize("bypass", [False, True])
def test_native_convolution_consumes_source_view_without_copy(monkeypatch, bypass):
    native = NativeYOLO26.__new__(NativeYOLO26)
    native.four_way_accumulation = True
    native._compact_box_modules = set()
    native.weight_l2_bypass = bypass
    native.device_id = 0
    native.weight_l2_cache_offset = 0
    native.trace = []
    native.session = SimpleNamespace(upload=lambda value: value)
    monkeypatch.setattr(native, "_prepare", lambda *_args: "prepared-convolution")
    monkeypatch.setattr(native, "_empty", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(native, "_dummy", lambda: object())

    def reject_compaction(_value):
        raise AssertionError("ordinary convolution must consume its source view")

    monkeypatch.setattr(native, "_compact", reject_compaction)
    queried = []

    def offset(device):
        queried.append(device)
        return 1 << 44

    monkeypatch.setattr("examples.models.yolo26_native.executor._runtime_l2_cache_offset", offset)
    source = object()
    view = Feature(source, 4, 4, 16, 64, 16)
    result = native._convolution(torch.nn.Conv2d(16, 16, 1).eval(), view, False)
    assert len(native.trace) == 1 and native.trace[0].label == "convolution"
    assert native.trace[0].arguments[1] is source
    assert native.trace[0].arguments[0][13:15].tolist() == [64.0, 16.0]
    assert queried == ([0] if bypass else [])
    assert native.trace[0].arguments[0][16:20].tolist() == ([0, 0, 4096, 0] if bypass else [0] * 4)
    assert (result.height, result.width, result.channels, result.row_stride, result.offset) == (
        4,
        4,
        16,
        16,
        0,
    )


@pytest.mark.parametrize("offset", [0, 1 << 44, 0xFEDCBA9876543210])
def test_convolution_weight_offset_round_trips_without_fp32_loss(offset):
    plan = ConvPlan(4, 4, 16, 16, 1, 1, 0, four_way_accumulation=True)
    words = plan.metadata(weight_l2_offset=offset)[16:20]
    assert sum(int(word) << (16 * i) for i, word in enumerate(words)) == offset


@pytest.mark.parametrize("offset", [True, -1, 1 << 64, 1.0])
def test_convolution_rejects_invalid_weight_l2_offset(offset):
    plan = ConvPlan(4, 4, 16, 16, 1, 1, 0, four_way_accumulation=True)
    with pytest.raises(ValueError, match="uint64"):
        plan.metadata(weight_l2_offset=offset)


def test_cube_convolution_rejects_weight_l2_offset():
    plan = ConvPlan(4, 4, 16, 16, 1, 1, 0)
    with pytest.raises(ValueError, match="four-way"):
        plan.metadata(weight_l2_offset=1 << 44)


@pytest.mark.parametrize(
    "current,status,offset", [(0, 0, 1 << 44), (1, 0, 1 << 44), (0, 107001, 0), (0, 0, 0)]
)
def test_weight_l2_query_uses_visible_runtime_device(monkeypatch, current, status, offset):
    seen = []

    def get_device(pointer):
        pointer._obj.value = current
        return 0

    def get_offset(device, pointer):
        seen.append(device)
        pointer._obj.value = offset
        return status

    library = SimpleNamespace(aclrtGetDevice=get_device, rtGetL2CacheOffset=get_offset)
    monkeypatch.setattr("examples.models.yolo26_native.executor.ctypes.CDLL", lambda _name: library)
    if current == 0 and status == 0 and offset:
        assert _runtime_l2_cache_offset(0) == offset
    else:
        with pytest.raises(RuntimeError):
            _runtime_l2_cache_offset(0)
    assert seen == ([] if current != 0 else [0])


@pytest.mark.parametrize(
    "cin,kernel,stride,ordered,selected",
    [
        (16, 1, 1, True, True),
        (512, 1, 1, True, True),
        (528, 1, 1, True, False),
        (17, 1, 1, True, False),
        (32, 3, 1, True, False),
        (32, 1, 2, True, False),
        (32, 1, 1, False, False),
    ],
)
def test_ordered_pointwise_cube_bounds(cin, kernel, stride, ordered, selected):
    plan = ConvPlan(8, 8, cin, 17, kernel, stride, kernel // 2, four_way_accumulation=ordered)
    assert plan.ordered_pointwise_cube is selected
    packed = plan.pack_weight(torch.ones((17, cin, kernel, kernel)))
    assert packed.numel() == plan.weight_size
    if selected:
        assert packed.numel() == plan.co * cin
        assert torch.count_nonzero(packed) == 17 * cin


@pytest.mark.parametrize("row_stride,selected", [(1048576, True), (1048592, False)])
def test_ordered_pointwise_cube_bounds_dma_source_stride(row_stride, selected):
    plan = ConvPlan(4, 4, 16, 16, 1, 1, 0, four_way_accumulation=True, source_stride=row_stride)
    assert plan.ordered_pointwise_cube is selected


@pytest.mark.parametrize(
    "height,cin,stride,selected",
    [
        (4, 3, 1, True),
        (32, 64, 2, True),
        (128, 3, 2, True),
        (4, 112, 1, True),
        (4, 113, 1, True),
        (4, 225, 1, True),
        (4, 226, 1, False),
        (8, 256, 1, False),
    ],
)
def test_ordered_spatial_cube_bounds(height, cin, stride, selected):
    plan = ConvPlan(height, height, cin, 17, 3, stride, 1, four_way_accumulation=True)
    assert plan.ordered_spatial_cube is selected
    assert plan.pack_weight(torch.ones((17, cin, 3, 3))).numel() == plan.co * plan.reduction_size


@pytest.mark.parametrize(
    "height,width,cin,selected",
    [
        (4, 4, 226, False),
        (8, 8, 256, True),
        (4, 4, 1024, False),
        (4, 16, 1024, True),
        (32, 46, 256, True),
        (32, 47, 256, False),
        (128, 128, 3, False),
        (4, 4, 128, False),
    ],
)
def test_streamed_spatial_cube_bounds(height, width, cin, selected):
    plan = ConvPlan(height, width, cin, 17, 3, 1, 1, four_way_accumulation=True)
    assert plan.streamed_spatial_cube is selected
    assert not (plan.streamed_spatial_cube and plan.ordered_spatial_cube)


@pytest.mark.parametrize("row_stride,selected", [(16 * 65536, True), (16 * 65537, False)])
def test_streamed_spatial_cube_bounds_dma_stride(row_stride, selected):
    plan = ConvPlan(8, 8, 256, 17, 3, 1, 1, four_way_accumulation=True, source_stride=row_stride)
    assert plan.streamed_spatial_cube is selected


def test_native_graph_rebuild_performs_no_replay_or_host_arithmetic():
    seen = []
    native = NativeYOLO26.__new__(NativeYOLO26)
    native._sequence = SimpleNamespace(rebuild=lambda: seen.append("rebuild"))
    with RejectHostOperators():
        native.rebuild_graph()
    assert seen == ["rebuild"]


def test_native_graph_rebuild_requires_registered_graph():
    native = NativeYOLO26.__new__(NativeYOLO26)
    with pytest.raises(RuntimeError, match="not prepared"):
        native.rebuild_graph()


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_native_concat_exposes_all_read_sources_and_falls_back(count):
    native = NativeYOLO26.__new__(NativeYOLO26)
    result = object()
    dummy = object()
    emitted = []
    copied = []
    native._empty = lambda count, dtype=torch.float16: result
    native._dummy = lambda: dummy
    native._emit = lambda plan, inputs, output, label, blocks=8: emitted.append((plan, inputs, output, label))
    native._copy = lambda x, out: copied.append((x, out))
    values = [Feature(object(), 4, 4, 16, 48, 16) for _ in range(count)]
    actual = native._concat(values)
    assert actual == Feature(result, 4, 4, count * 16, count * 16)
    if count <= 4:
        assert not copied and len(emitted) == 1
        plan, inputs, output, label = emitted[0]
        assert plan.sources == ((16, 48, 16),) * count
        assert inputs == tuple(v.data for v in values) + (dummy,) * (4 - count)
        assert output is result and label == "concat"
    else:
        assert not emitted and len(copied) == count
        assert [target.offset for _, target in copied] == [i * 16 for i in range(count)]


@pytest.mark.parametrize(
    "cin,cout,selected",
    [
        (226, 16, False),
        (228, 16, True),
        (256, 16, True),
        (512, 16, True),
        (1024, 16, True),
        (256, 17, False),
        (256, 64, False),
    ],
)
def test_streamed_spatial_cube_single_tile_grid(cin, cout, selected):
    plan = ConvPlan(4, 4, cin, cout, 3, 1, 1, four_way_accumulation=True)
    assert not plan.ordered_spatial_cube
    assert plan.streamed_spatial_cube is selected


@pytest.mark.parametrize(
    "height,cin,cout,blocks",
    [(8, 128, 16, 4), (4, 16, 16, 1), (4, 16, 32, 2), (16, 16, 16, 8), (4, 256, 16, 8)],
)
def test_spatial_convolution_skips_inactive_cube_blocks(monkeypatch, height, cin, cout, blocks):
    native = NativeYOLO26.__new__(NativeYOLO26)
    native.four_way_accumulation = True
    native._compact_box_modules = set()
    native.weight_l2_bypass = False
    native.weight_l2_cache_offset = 0
    native.trace = []
    native.session = SimpleNamespace(upload=lambda value: value)
    monkeypatch.setattr(native, "_prepare", lambda *_args: "prepared-convolution")
    monkeypatch.setattr(native, "_empty", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(native, "_dummy", lambda: object())
    module = torch.nn.Conv2d(cin, cout, 3, padding=1).eval()
    source = Feature(object(), height, height, cin, cin)
    result = native._convolution(module, source, False)
    assert len(native.trace) == 1 and native.trace[0].blocks == blocks
    assert (result.height, result.width, result.channels) == (height, height, cout)


def test_dense_spatial_policy_preserves_exact_ordered_selection():
    ordered = ConvPlan(8, 8, 128, 16, 3, 1, 1, four_way_accumulation=True)
    dense = ConvPlan(8, 8, 128, 16, 3, 1, 1)
    assert ordered.ordered_spatial_cube and not ordered.compact_spatial_cube
    assert dense.compact_spatial_cube and not dense.ordered_spatial_cube
    assert ordered.metadata()[15].item() == 0 and dense.metadata()[15].item() == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_compact_box_policy_does_not_change_score_convolution(monkeypatch, enabled):
    box = torch.nn.Conv2d(128, 16, 3, padding=1).eval()
    score = torch.nn.Conv2d(128, 16, 3, padding=1).eval()
    native = NativeYOLO26.__new__(NativeYOLO26)
    native.four_way_accumulation = True
    native._compact_box_modules = {id(box)} if enabled else set()
    native.weight_l2_bypass = False
    native.weight_l2_cache_offset = 0
    native.trace = []
    native.session = SimpleNamespace(upload=lambda value: value)
    monkeypatch.setattr(native, "_prepare", lambda *_args: "prepared-convolution")
    monkeypatch.setattr(native, "_empty", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(native, "_dummy", lambda: object())
    source = Feature(object(), 8, 8, 128, 160, 16)
    for module in (box, score):
        native._convolution(module, source, True)
    assert native.four_way_accumulation is True
    assert [int(call.arguments[0][15]) for call in native.trace] == [int(enabled), 0]
    assert [call.blocks for call in native.trace] == [4, 4]
    assert all(call.arguments[1] is source.data for call in native.trace)


@pytest.mark.parametrize(
    "height,width,cin,cout,stride",
    [
        (640, 640, 3, 16, 2),
        (320, 320, 16, 32, 2),
        (160, 160, 64, 64, 2),
        (80, 80, 128, 128, 2),
        (80, 80, 64, 64, 2),
        (80, 80, 64, 16, 1),
    ],
)
def test_windowed_spatial_cube_production_geometry(height, width, cin, cout, stride):
    plan = ConvPlan(height, width, cin, cout, 3, stride, 1, four_way_accumulation=True)
    assert not plan.ordered_spatial_cube and not plan.streamed_spatial_cube
    assert plan.windowed_spatial_cube
    assert Path(plan.jit_kernel()._external_source_paths()[0]).name == "conv_spatial_ordered.cpp"
    assert plan.window_image_bytes + min(plan.reduction_size, 576) * 512 <= 1024 * 1024


def test_windowed_spatial_cube_preserves_existing_choices():
    small = ConvPlan(8, 8, 64, 128, 3, 1, 1, four_way_accumulation=True)
    wide = ConvPlan(16, 4096, 1024, 16, 3, 1, 1, four_way_accumulation=True)
    unordered = ConvPlan(640, 640, 3, 16, 3, 2, 1)
    assert small.ordered_spatial_cube and not small.windowed_spatial_cube
    assert not wide.windowed_spatial_cube
    assert not unordered.windowed_spatial_cube


def test_native_batch_coalesces_dependencies_and_retains_distinct_bindings(monkeypatch, tmp_path):
    from examples.models.yolo26_native import executor_batch  # noqa: PLC0415

    events = []

    # Kernel objects are identity keys, matching the real resident handles.
    class Kernel:
        def __init__(self, params):
            self._params = params

    originals = [Kernel([{"shape": [16], "dtype": "fp16"}]) for _ in range(2)]

    class Worker:
        def prepare_independent(self, artifact, count):
            events.append(("prepare", artifact, count))
            return f"independent-{artifact}"

        def register_graph(self, calls):
            events.append(("graph", calls))

            class Graph:
                def __call__(self):
                    events.append("replay")

                def rebuild(self):
                    events.append("rebuild")

            return Graph()

        def close(self):
            events.append("close")

    worker = Worker()
    images = []

    def image(*_args, shared, **_kwargs):
        index = len(images)
        assert shared is (images[0] if images else None)
        result = SimpleNamespace(
            session=SimpleNamespace(worker=worker, constants={}),
            trace=[Launch(k, (f"image-{index}-task-{j}",), 8, f"task-{j}") for j, k in enumerate(originals)],
            _artifacts={k: j for j, k in enumerate(originals)},
        )
        images.append(result)
        return result

    monkeypatch.setattr(executor_batch, "_SharedImage", image)
    with executor_batch.NativeYOLO26Batch(None, 640, tmp_path) as batch:
        assert events[:2] == [("prepare", 0, 8), ("prepare", 1, 8)]
        calls = events[2][1]
        assert len(calls) == 2
        for stage, (kernel, arguments, blocks) in enumerate(calls):
            assert kernel == f"independent-{stage}" and blocks == 8
            assert arguments == tuple(f"image-{i}-task-{stage}" for i in range(8))
        with RejectHostOperators():
            batch.replay()
        batch.rebuild_graph()
        assert events[-2:] == ["replay", "rebuild"]
        coverage = batch.coverage()
        assert coverage["physical_chip_count"] == 1 and coverage["device_compute_tasks"] == 2
        assert coverage["model_execute_calls_per_replay"] == 1
        assert coverage["runtime_kernel_launches_per_replay"] == 0
    assert events[-1] == "close"


@pytest.mark.parametrize("size,batch", [(128, 8), (640, 1), (640, 9), (True, 8), (640, True)])
def test_native_batch_rejects_unqualified_shapes_before_setup(tmp_path, size, batch):
    from examples.models.yolo26_native.executor_batch import NativeYOLO26Batch  # noqa: PLC0415

    with pytest.raises(ValueError, match="size=640 and batch_size=8"):
        NativeYOLO26Batch(None, size, tmp_path, batch_size=batch)


def test_batch_constant_uploads_do_not_alias_activation_allocations():
    from examples.models.yolo26_native.executor_batch import _ConstantUploads  # noqa: PLC0415

    uploaded = []
    worker = SimpleNamespace(
        upload=lambda tensor: uploaded.append(tensor.clone()) or object(),
        empty=lambda *_args: object(),
    )
    constants = _ConstantUploads(worker)
    a = torch.arange(16).half()
    first = constants.upload(a)
    assert constants.upload(a.clone()) is first
    assert constants.upload(a.float()) is not first
    assert constants.upload(a.reshape(4, 4)) is not first
    assert len(uploaded) == 3
    assert constants.empty((16,), torch.float16) is not constants.empty((16,), torch.float16)
    # The worker itself retains ordinary upload behavior; only the private
    # recording proxy deduplicates immutable setup values.
    assert worker.upload(a) is not first
    assert len(uploaded) == 4


def test_native_batch_explicit_transfers_keep_image_identity_and_validate_input():
    from examples.models.yolo26_native.executor_batch import NativeYOLO26Batch  # noqa: PLC0415

    batch: Any = NativeYOLO26Batch.__new__(NativeYOLO26Batch)
    batch.size, batch.batch_size = 640, 8
    copied = []
    batch.images = [
        SimpleNamespace(
            input=SimpleNamespace(copy_from=lambda tensor: copied.append(tensor)),
            output=SimpleNamespace(to_cpu=lambda i=i: torch.full((1800,), float(i))),
        )
        for i in range(8)
    ]
    # Expanded input has no materialized image storage. Explicit transfer must
    # make each image contiguous while preserving its distinct batch value.
    x = torch.arange(8).float().reshape(8, 1, 1, 1).expand(8, 3, 640, 640)
    batch.copy_input(x)
    assert len(copied) == 8
    for image, tensor in enumerate(copied):
        assert tensor.shape == (3 * 640 * 640,) and tensor.is_contiguous()
        assert torch.all(tensor == image)
    output = batch.detections_cpu()
    assert output.shape == (8, 300, 6)
    for image in range(8):
        assert torch.all(output[image] == image)
    for invalid in (torch.zeros(1), x.half(), torch.empty(x.shape, device="meta")):
        with pytest.raises(ValueError, match="CPU FP32"):
            batch.copy_input(invalid)
    assert len(copied) == 8


@pytest.mark.parametrize("positions", [16, 400])
def test_native_attention_softmax_has_one_task_and_no_host_arithmetic(monkeypatch, positions):
    """Both qualified attention sizes emit one resident task, not per-row launches."""
    native = NativeYOLO26.__new__(NativeYOLO26)
    native.precise_small_softmax = True
    score, output = object(), object()
    seen = []
    monkeypatch.setattr(native, "_empty", lambda *_args: output)
    monkeypatch.setattr(native, "_emit", lambda *args: seen.append(args))
    with RejectHostOperators():
        result = native._softmax(score, positions)
    assert result is output and len(seen) == 1
    plan, inputs, destination, label = seen[0]
    assert isinstance(plan, Softmax16Plan if positions == 16 else Softmax400Plan)
    assert inputs == (score,) and destination is output
    assert label == ("softmax:pair" if positions == 16 else "softmax:matrix")


@pytest.mark.parametrize("plan", [Softmax16Plan(), Softmax400Plan()])
def test_attention_softmax_uses_public_compiler_only(plan):
    assert not plan.jit_kernel()._external_source_paths()


def test_public_classifier_selection_and_constant_packing():
    original = ConvPlan(80, 80, 80, 80, 1, 1, 0, False, True)
    selected = select_classifier_projection(original, weight_l2_bypass=False)
    assert isinstance(selected, ClassifierProjectionPlan)
    assert not selected.four_way_accumulation
    assert selected.weight_size == original.weight_size == 6400
    weight = torch.arange(6400).reshape(80, 80, 1, 1).half()
    assert torch.equal(selected.pack_weight(weight), original.pack_weight(weight))
    assert not selected.jit_kernel()._external_source_paths()
    for plan in (
        replace(original, silu=True),
        replace(original, height=40),
        replace(original, source_stride=96),
    ):
        assert select_classifier_projection(plan, weight_l2_bypass=False) is plan
    assert select_classifier_projection(original, weight_l2_bypass=True) is original
    with pytest.raises(ValueError, match="offset"):
        selected.metadata(offset=16)
    with pytest.raises(ValueError, match="guards"):
        selected.jit_kernel(output_padding=16)


def test_public_ordered_pointwise_profile_selection():
    original = ConvPlan(160, 160, 48, 64, 1, 1, 0, True, True)
    selected = select_ordered_pointwise(original, weight_l2_bypass=False)
    assert isinstance(selected, PublicOrderedPointwisePlan)
    assert selected.weight_size == 49152
    assert selected.four_way_accumulation
    assert torch.equal(selected.metadata(), original.metadata())
    assert not selected.jit_kernel()._external_source_paths()
    packed = selected.pack_weight(torch.ones((64, 48, 1, 1)))
    assert packed.dtype == torch.float16 and packed.numel() == selected.weight_size
    assert int(torch.count_nonzero(packed)) == 64 * 48
    for plan in (
        replace(original, silu=False),
        replace(original, four_way_accumulation=False),
        replace(original, height=80),
        replace(original, source_stride=64),
    ):
        assert select_ordered_pointwise(plan, weight_l2_bypass=False) is plan
    assert select_ordered_pointwise(original, weight_l2_bypass=True) is original
    with pytest.raises(ValueError, match="offset"):
        selected.metadata(offset=16)
    with pytest.raises(ValueError, match="guards"):
        selected.jit_kernel(output_padding=16)
    with pytest.raises(ValueError, match="weights"):
        selected.pack_weight(torch.ones((64, 48)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
