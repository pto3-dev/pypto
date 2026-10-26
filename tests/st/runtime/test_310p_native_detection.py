# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native dense decoding/sigmoid and exact two-stage detection TopK."""

import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import load_plan, require_310p


def _compile(plan, tmp_path):
    return ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )


@pytest.mark.parametrize("stride", [8, 16, 32])
def test_310p_native_decode(tmp_path, stride):
    require_310p()
    plan = load_plan("detection").DecodePlan(4, 4, stride, 16, 336)
    compiled = _compile(plan, tmp_path)
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        db = session.empty((16 * 16,), torch.float16)
        dc = session.empty((16 * 80,), torch.float16)
        out = session.empty((336 * 96,), torch.float32)
        meta = session.upload(plan.metadata())
        for seed in (283, 293):
            g = torch.Generator().manual_seed(seed)
            boxes = (torch.randn((16, 16), generator=g) * 2).half()
            classes = (torch.randn((16, 80), generator=g) * 2).half()
            db.copy_from(boxes.reshape(-1))
            dc.copy_from(classes.reshape(-1))
            out.copy_from(torch.full((336 * 96,), 7.0))
            kernel(meta, db, dc, out, blocks=8)
            actual = out.to_cpu().reshape(336, 96)
            y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
            anchors = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1).half() + 0.5
            expected_boxes = torch.cat((anchors - boxes[:, :2], anchors + boxes[:, 2:4]), dim=1) * stride
            torch.testing.assert_close(actual[16:32, :4], expected_boxes.float(), rtol=0, atol=0)
            torch.testing.assert_close(
                actual[16:32, 4:84], classes.sigmoid().float(), rtol=0.001, atol=0.00003
            )
            assert torch.count_nonzero(actual[16:32, 84:]) == 0
            assert torch.equal(actual[:16], torch.full((16, 96), 7.0))
            assert torch.equal(actual[32:], torch.full((304, 96), 7.0))


def test_310p_native_topk(tmp_path):
    require_310p()
    plan = load_plan("detection").TopKPlan(336)
    compiled = _compile(plan, tmp_path)
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        dx = session.empty((336 * 96,), torch.float32)
        guard = session.upload(torch.full((1800 + 32,), 7.0))
        out = guard.view((1800,), 16)
        meta = session.upload(plan.metadata())
        for seed in (307, 311):
            g = torch.Generator().manual_seed(seed)
            dense = torch.zeros(336, 96)
            dense[:, :4] = torch.arange(336 * 4).reshape(336, 4).float()
            dense[:, 4:84] = torch.randperm(336 * 80, generator=g).reshape(336, 80).float() / (336 * 80)
            candidate = dense[:, 4:84].amax(1).topk(300).indices
            values, indices = dense[candidate, 4:84].reshape(-1).topk(300)
            anchors = candidate[indices // 80]
            expected = torch.cat(
                (dense[anchors, :4], values[:, None], (indices % 80).float()[:, None]), dim=1
            )
            dx.copy_from(dense.reshape(-1))
            kernel(meta, dx, out)
            torch.testing.assert_close(out.to_cpu().reshape(300, 6), expected, rtol=0, atol=0)
            full = guard.to_cpu()
            assert torch.equal(full[:16], torch.full((16,), 7.0))
            assert torch.equal(full[-16:], torch.full((16,), 7.0))

        # All-zero probabilities exercise tie retention and the negative sentinel.
        dense[:, 4:84].zero_()
        dx.copy_from(dense.reshape(-1))
        kernel(meta, dx, out)
        indices = torch.arange(300)
        expected = torch.cat(
            (dense[indices // 80, :4], torch.zeros(300, 1), (indices % 80).float()[:, None]), dim=1
        )
        torch.testing.assert_close(out.to_cpu().reshape(300, 6), expected, rtol=0, atol=0)


def test_detection_rejects_invalid_geometry():
    module = load_plan("detection")
    with pytest.raises(ValueError, match="300"):
        module.TopKPlan(299)
    with pytest.raises(ValueError, match="interval"):
        module.DecodePlan(20, 20, 32, 0, 336)


@pytest.mark.parametrize("anchors", [300, 336, 384, 385])
def test_310p_topk_preserves_mixed_tie_traversal(tmp_path, anchors):
    require_310p()
    plan = load_plan("detection").TopKPlan(anchors)
    compiled = _compile(plan, tmp_path)
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        meta = session.upload(plan.metadata())
        dx = session.empty((anchors * 96,), torch.float32)
        storage = session.upload(torch.full((1832,), 7.0))
        output = storage.view((1800,), 16)
        for seed in (379, 383):
            g = torch.Generator().manual_seed(seed)
            dense = torch.zeros((anchors, 96))
            dense[:, :4] = torch.arange(anchors * 4).reshape(anchors, 4)
            dense[:, 4:84] = 0.5
            high = torch.randperm(anchors, generator=g)[:10]
            dense[high, 4] = 0.75 + torch.arange(10) / 64
            maximum = dense[:, 4:84].amax(1)
            selected = sorted(range(anchors), key=lambda i: (-float(maximum[i]), i))[:300]
            # Stable two-stage sorting is independent of the device heap implementation.
            pairs = [(float(dense[a, 4 + c]), rank, c) for rank, a in enumerate(selected) for c in range(80)]
            best = sorted(pairs, key=lambda p: (-p[0], p[1], p[2]))[:300]
            expected = torch.stack(
                [
                    torch.cat((dense[selected[rank], :4], torch.tensor([score, float(c)])))
                    for score, rank, c in best
                ]
            )
            dx.copy_from(dense.reshape(-1))
            kernel(meta, dx, output)
            torch.testing.assert_close(output.to_cpu().reshape(300, 6), expected, rtol=0, atol=0)
            full = storage.to_cpu()
            assert torch.equal(full[:16], torch.full((16,), 7.0))
            assert torch.equal(full[-16:], torch.full((16,), 7.0))


@pytest.mark.parametrize("anchors", [300, 301, 319, 320, 321, 336, 383, 384, 385, 512, 1344])
def test_310p_topk_fp32_boundary_ties(tmp_path, anchors):
    """Keep FP32 distinctions and exact ties across proposal merge boundaries."""
    require_310p()
    plan = load_plan("detection").TopKPlan(anchors)
    compiled = _compile(plan, tmp_path)
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        meta = session.upload(plan.metadata())
        dx = session.empty((anchors * 96,), torch.float32)
        guard = session.upload(torch.full((1832,), 7.0))
        output = guard.view((1800,), 16)
        for seed in (409, 419):
            g = torch.Generator().manual_seed(seed)
            dense = torch.zeros((anchors, 96))
            dense[:, :4] = torch.arange(anchors * 4).reshape(anchors, 4)
            # All these distinct FP32 scores round to the same FP16 value.
            dense[:, 4:84] = 0.5 + torch.randint(0, 32, (anchors, 80), generator=g) * 2**-24
            maximum = dense[:, 4:84].amax(1)
            selected = sorted(range(anchors), key=lambda i: (-float(maximum[i]), i))[:300]
            pairs = [(float(dense[a, 4 + c]), rank, c) for rank, a in enumerate(selected) for c in range(80)]
            best = sorted(pairs, key=lambda p: (-p[0], p[1], p[2]))[:300]
            expected = torch.stack(
                [
                    torch.cat((dense[selected[rank], :4], torch.tensor([score, float(c)])))
                    for score, rank, c in best
                ]
            )
            dx.copy_from(dense.reshape(-1))
            kernel(meta, dx, output)
            torch.testing.assert_close(output.to_cpu().reshape(300, 6), expected, rtol=0, atol=0)
            actual = guard.to_cpu()
            assert torch.equal(actual[:16], torch.full((16,), 7.0))
            assert torch.equal(actual[-16:], torch.full((16,), 7.0))


@pytest.mark.parametrize("anchors", [336, 385])
def test_310p_topk_late_class_mask_boundaries(tmp_path, anchors):
    """Later anchor rows must retain winners on both compare-mask boundaries."""
    require_310p()
    plan = load_plan("detection").TopKPlan(anchors)
    compiled = _compile(plan, tmp_path)
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        meta = session.upload(plan.metadata())
        dx = session.empty((anchors * 96,), torch.float32)
        guard = session.upload(torch.full((1832,), 7.0))
        output = guard.view((1800,), 16)
        for classes, scale in (
            ([0, 59, 60, 63, 64, 79], 1.0),
            ([3, 4, 58, 59, 60, 78, 79], 1.0),
            ([0, 59, 60, 79], 2**-140),
        ):
            dense = torch.zeros((anchors, 96))
            dense[:, :4] = torch.arange(anchors * 4).reshape(anchors, 4)
            dense[:10, 4 + torch.tensor(classes)] = 0.75
            dense[:10, 4] = 0.9 - torch.arange(10) / 100
            dense[:, 4:84] *= scale
            selected = dense[:, 4:84].amax(1).argsort(descending=True, stable=True)[:300]
            flat = dense[selected, 4:84].reshape(-1)
            indices = flat.argsort(descending=True, stable=True)[:300]
            expected = torch.cat(
                (dense[selected[indices // 80], :4], flat[indices, None], (indices % 80).float()[:, None]),
                dim=1,
            )
            dx.copy_from(dense.reshape(-1))
            kernel(meta, dx, output)
            torch.testing.assert_close(output.to_cpu().reshape(300, 6), expected, rtol=0, atol=0)
            actual = guard.to_cpu()
            assert torch.equal(actual[:16], torch.full((16,), 7.0))
            assert torch.equal(actual[-16:], torch.full((16,), 7.0))


@pytest.mark.parametrize("anchors", [385, 509, 510, 511, 512, 765, 766, 8400])
def test_310p_topk_stream_chunks_exact_ties(tmp_path, anchors):
    """Keep 255-row chunks, short tails and production selection bitwise exact.

    Large box coordinates and padded fields must stay outside class reductions.
    Stable CPU sorting independently defines both anchor and flattened-class
    tie order; changed quantized/late/zero scores exercise heap updates across
    chunk boundaries and artifact reload without narrowing FP32 scores.
    """
    require_310p()
    plan = load_plan("detection").TopKPlan(anchors)
    compiled = _compile(plan, tmp_path)
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        source = session.empty((anchors * 96,), torch.float32)
        meta = session.upload(plan.metadata())
        guard = session.upload(torch.full((1832,), 7.0))
        output = guard.view((1800,), 16)
        for mode in range(3):
            g = torch.Generator().manual_seed(503 + mode)
            dense = torch.full((anchors, 96), 17.0)
            ids = torch.arange(anchors).float()
            dense[:, :4] = torch.stack((ids, ids + 0.25 + mode, ids + 100000, -ids - 100000), 1)
            classes = dense[:, 4:84]
            if mode == 0:
                classes.copy_(torch.randint(0, 64, (anchors, 80), generator=g).float() / 64)
                for anchor in (254, 255, 509, 510, 764, 765):
                    if anchor < anchors:
                        classes[anchor, 79] = 1.0
            elif mode == 1:
                classes.fill_(0.5)
                late = torch.arange(anchors - 300, anchors)
                boundary = torch.tensor([0, 59, 60, 79])[late % 4]
                classes[late, boundary] = 0.75
            else:
                classes.zero_()
                classes[254::255, 79] = 0.5
                classes[255::255, 0] = 0.5
            maximum = classes.amax(1)
            selected = torch.argsort(maximum, descending=True, stable=True)[:300]
            flat = classes[selected].reshape(-1)
            best = torch.argsort(flat, descending=True, stable=True)[:300]
            expected = torch.cat(
                (dense[selected[best // 80], :4], flat[best, None], (best % 80).float()[:, None]), 1
            )
            source.copy_from(dense.reshape(-1))
            output.copy_from(torch.full((1800,), 7.0))
            kernel(meta, source, output)
            assert torch.equal(output.to_cpu().view(torch.int32), expected.reshape(-1).view(torch.int32))
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
