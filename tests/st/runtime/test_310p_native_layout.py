# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native resident input packing, routing, residuals, pooling and casts."""

import pytest
import torch
import torch.nn.functional as F
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import load_plan, require_310p

_PACKS = ("pack", "pack_single", "pack_full")


def _plan(name):
    module = load_plan("layout")
    op = module.VectorOp
    constructors = {
        "pack": lambda: module.VectorPlan(op.PACK, 8, 8, channels=3),
        "pack_single": lambda: module.VectorPlan(op.PACK, 4, 4, channels=1),
        "pack_full": lambda: module.VectorPlan(op.PACK, 4, 4, channels=16),
        "copy": lambda: module.VectorPlan(op.COPY, 4, 4, 16, 48, 64, 16, 32),
        "upsample": lambda: module.VectorPlan(op.UPSAMPLE, 8, 8, 32, 32, 32, input_height=4, input_width=4),
        "add": lambda: module.VectorPlan(op.ADD, 4, 4, 16, 48, 32, 16, 16, second_stride=32),
        "maxpool": lambda: module.VectorPlan(op.MAXPOOL, 4, 4, 32, 32, 32),
        "float_to_half": lambda: module.VectorPlan(op.FLOAT_TO_HALF, 512),
        "half_to_float": lambda: module.VectorPlan(op.HALF_TO_FLOAT, 160000, scale=0.125),
        "scale_half": lambda: module.VectorPlan(op.SCALE_HALF, 512, scale=32**-0.5),
    }
    return constructors[name]()


def _expected(plan, name, x, b):
    result = torch.full(
        (plan.output_size,), 7.0, dtype=torch.float32 if name == "half_to_float" else torch.float16
    )
    if name in _PACKS:
        result = result.reshape(plan.height, plan.width, 16)
        result.zero_()
        result[:, :, : plan.channels] = (
            x.reshape(plan.channels, plan.height, plan.width).permute(1, 2, 0).half()
        )
    elif name in ("float_to_half", "half_to_float", "scale_half"):
        result = x.half() if name == "float_to_half" else (x.float() * plan.scale).to(result.dtype)
    else:
        source = x.reshape(-1, plan.source_stride)[:, plan.source_offset : plan.source_offset + plan.channels]
        if name == "upsample":
            source = (
                source.reshape(plan.input_height, plan.input_width, plan.channels)
                .repeat_interleave(2, 0)
                .repeat_interleave(2, 1)
                .reshape(-1, plan.channels)
            )
        elif name == "add":
            source = (
                source
                + b.reshape(-1, plan.second_stride)[
                    :, plan.second_offset : plan.second_offset + plan.channels
                ]
            )
        elif name == "maxpool":
            source = (
                F.max_pool2d(
                    source.reshape(plan.height, plan.width, plan.channels).permute(2, 0, 1).unsqueeze(0),
                    5,
                    1,
                    2,
                )[0]
                .permute(1, 2, 0)
                .reshape(-1, plan.channels)
            )
        result.reshape(-1, plan.destination_stride)[
            :, plan.destination_offset : plan.destination_offset + plan.channels
        ] = source
    return result.reshape(-1)


@pytest.mark.parametrize(
    "name", [*_PACKS, "copy", "upsample", "add", "maxpool", "float_to_half", "half_to_float", "scale_half"]
)
def test_310p_native_layout(tmp_path, name):
    require_310p()
    plan = _plan(name)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    dtype = torch.float32 if name in (*_PACKS, "float_to_half") else torch.float16
    output_dtype = torch.float32 if name == "half_to_float" else torch.float16
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        dx = session.empty((plan.input_size,), dtype)
        db = session.empty((plan.second_size,), torch.float16)
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=output_dtype))
        out = guard.view((plan.output_size,), 16)
        metadata = session.upload(plan.metadata())
        for seed in (257, 263):
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(plan.input_size, generator=g).to(dtype)
            b = torch.randn(plan.second_size, generator=g).half()
            if name in _PACKS:
                x[:12] = torch.tensor(
                    [
                        0.0,
                        -0.0,
                        65504.0,
                        -65504.0,
                        1.00048828125,
                        1.00146484375,
                        2**-24,
                        2**-25,
                        3 * 2**-25,
                        -(2**-24),
                        -(2**-25),
                        -3 * 2**-25,
                    ]
                )
            dx.copy_from(x)
            db.copy_from(b)
            out.copy_from(torch.full((plan.output_size,), 7.0, dtype=output_dtype))
            kernel(metadata, dx, db, out, blocks=8)
            expected = _expected(plan, name, x, b)
            actual_output = out.to_cpu()
            torch.testing.assert_close(actual_output, expected, rtol=0, atol=0)
            if name in _PACKS:
                assert torch.equal(actual_output.view(torch.int16), expected.view(torch.int16))
            actual = guard.to_cpu()
            assert torch.equal(actual[:16], torch.full((16,), 7.0, dtype=output_dtype))
            assert torch.equal(actual[-16:], torch.full((16,), 7.0, dtype=output_dtype))


def test_layout_rejects_invalid_slices_and_upsampling():
    module = load_plan("layout")
    with pytest.raises(ValueError, match="slice"):
        module.VectorPlan(module.VectorOp.COPY, 4, 4, 16, 16, 16, 16)
    with pytest.raises(ValueError, match="twice"):
        module.VectorPlan(module.VectorOp.UPSAMPLE, 8, 8, input_height=3, input_width=4)
    with pytest.raises(ValueError, match="256"):
        module.VectorPlan(module.VectorOp.FLOAT_TO_HALF, 255)


@pytest.mark.parametrize(
    "pixels,channels,blocks", [(1, 16, 8), (17, 48, 8), (257, 512, 8), (257, 1024, 1), (33, 1040, 8)]
)
def test_310p_copy_row_panels_exact_bits(tmp_path, pixels, channels, blocks):
    require_310p()
    module = load_plan("layout")
    plan = module.VectorPlan(module.VectorOp.COPY, pixels, 1, channels, channels + 32, channels + 64, 16, 32)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        call = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        source = session.empty((plan.input_size,), torch.float16)
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.output_size,), 16)
        zero = session.upload(torch.zeros(16, dtype=torch.float16))
        meta = session.upload(plan.metadata())
        for seed in (421, 431):
            g = torch.Generator().manual_seed(seed)
            bits = torch.randint(-32768, 32768, (plan.input_size,), generator=g, dtype=torch.int16)
            x = bits.view(torch.float16)
            source.copy_from(x)
            output.copy_from(torch.full((plan.output_size,), 7.0, dtype=torch.float16))
            call(meta, source, zero, output, blocks=blocks)
            expected = torch.full((plan.output_size,), 7.0, dtype=torch.float16).reshape(pixels, -1)
            expected[:, 32 : 32 + channels] = x.reshape(pixels, -1)[:, 16 : 16 + channels]
            assert torch.equal(output.to_cpu().view(torch.int16), expected.reshape(-1).view(torch.int16))
            actual = guard.to_cpu()
            assert torch.all(actual[:16] == 7) and torch.all(actual[-16:] == 7)


@pytest.mark.parametrize(
    "pixels,specs,blocks,shared",
    [
        (1, ((16, 48, 16), (32, 64, 32)), 8, False),
        (17, ((16, 48, 16), (48, 80, 16), (32, 64, 32)), 8, False),
        (257, ((512, 544, 16), (128, 160, 16), (256, 288, 32), (16, 48, 16)), 1, False),
        (257, ((1024, 1040, 16),) * 4, 1, False),
        (16, ((16, 64, 16), (32, 64, 32)), 8, True),
    ],
)
def test_310p_concat_row_panels_exact_bits(tmp_path, pixels, specs, blocks, shared):
    require_310p()
    plan = load_plan("layout").ConcatPlan(pixels, 1, specs)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        call = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        sources = [session.empty((n,), torch.float16) for n in plan.input_sizes]
        if shared:
            sources[1] = sources[0]
        dummy = session.upload(torch.zeros(16, dtype=torch.float16))
        inputs = tuple(sources) + (dummy,) * (4 - len(sources))
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.output_size,), 16)
        meta = session.upload(plan.metadata())
        for seed in (443, 449):
            g = torch.Generator().manual_seed(seed)
            values = [
                torch.randint(-32768, 32768, (n,), generator=g, dtype=torch.int16).view(torch.float16)
                for n in plan.input_sizes
            ]
            if shared:
                values[1] = values[0]
            for source, value in zip(sources, values, strict=True):
                source.copy_from(value)
            call(meta, *inputs, output, blocks=blocks)
            pieces = [
                value.reshape(pixels, stride)[:, offset : offset + channels]
                for value, (channels, stride, offset) in zip(values, specs, strict=True)
            ]
            expected = torch.cat(pieces, dim=1).reshape(-1)
            assert torch.equal(output.to_cpu().view(torch.int16), expected.view(torch.int16))
            actual = guard.to_cpu()
            assert torch.all(actual[:16] == 7) and torch.all(actual[-16:] == 7)


@pytest.mark.parametrize(
    "height,specs",
    [
        (1, ((16, 16, 0),)),
        (1, ((16, 16, 0),) * 5),
        (1, ((3, 16, 0), (16, 16, 0))),
        (1, ((16, 16, 16), (16, 16, 0))),
        (1, ((4096, 4096, 0), (16, 16, 0))),
        (1, ((16, 1048592, 0), (16, 16, 0))),
        (2**24 + 1, ((16, 16, 0), (16, 16, 0))),
    ],
)
def test_concat_rejects_unbounded_geometry(height, specs):
    with pytest.raises(ValueError, match="concat"):
        load_plan("layout").ConcatPlan(height, 1, specs)


@pytest.mark.parametrize("pixels,channels", [(1, 16), (17, 48), (257, 128), (65, 1024), (17, 1040)])
@pytest.mark.parametrize("blocks", [1, 8])
def test_310p_add_row_panels_exact_bits(tmp_path, pixels, channels, blocks):
    """Preserve both sliced inputs, ragged panels and untouched output gaps."""
    require_310p()
    module = load_plan("layout")
    plan = module.VectorPlan(
        module.VectorOp.ADD,
        pixels,
        1,
        channels,
        channels + 32,
        channels + 64,
        16,
        32,
        second_stride=channels + 48,
        second_offset=32,
    )
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        call = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        left = session.empty((plan.input_size,), torch.float16)
        right = session.empty((plan.second_size,), torch.float16)
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.output_size,), 16)
        meta = session.upload(plan.metadata())
        for seed in (1213, 1217):
            generator = torch.Generator().manual_seed(seed)
            x = torch.randn(plan.input_size, generator=generator).half()
            b = torch.randn(plan.second_size, generator=generator).half()
            # Include signed zero, cancellation and the smallest FP16 values.
            edge = torch.tensor([0.0, -0.0, 2**-24, -(2**-24), 1.0, -1.0]).half()
            x[16:22] = edge
            b[32:38] = -edge
            left.copy_from(x)
            right.copy_from(b)
            output.copy_from(torch.full((plan.output_size,), 7.0, dtype=torch.float16))
            call(meta, left, right, output, blocks=blocks)
            expected = _expected(plan, "add", x, b)
            assert torch.equal(output.to_cpu().view(torch.int16), expected.view(torch.int16))
            actual = guard.to_cpu()
            assert torch.all(actual[:16] == 7) and torch.all(actual[-16:] == 7)


@pytest.mark.parametrize("blocks", [1, 8])
@pytest.mark.parametrize(
    "height,width,channels",
    [
        (1, 1, 16),
        (3, 5, 80),
        (20, 20, 128),
        (16, 16, 256),
        (17, 16, 256),
        (1, 4096, 16),
        (1, 4097, 16),
        (5, 7, 1024),
        (9, 7, 176),
    ],
)
def test_310p_maxpool_resident_input_exact_bits(tmp_path, height, width, channels, blocks):
    """Check full/tail vectors, UB capacity, DMA chunk boundaries and fallback.

    Strided input/output slices and guards catch full-channel DMA overruns.
    Independent CPU pooling checks two changed finite inputs and artifact reload.
    """
    require_310p()
    module = load_plan("layout")
    plan = module.VectorPlan(
        module.VectorOp.MAXPOOL, height, width, channels, channels + 32, channels + 64, 16, 32
    )
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        call = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        source = session.empty((plan.input_size,), torch.float16)
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.output_size,), 16)
        zero = session.upload(torch.zeros(16, dtype=torch.float16))
        meta = session.upload(plan.metadata())
        for seed in (457, 461):
            g = torch.Generator().manual_seed(seed)
            x = -torch.randn(plan.input_size, generator=g).abs().half()
            edge = torch.tensor([-65504.0, 65504.0, 2**-24, -(2**-24), 0.0, 1.0, -1.0]).half()
            x[16:23] = edge
            source.copy_from(x)
            output.copy_from(torch.full((plan.output_size,), 7.0, dtype=torch.float16))
            call(meta, source, zero, output, blocks=blocks)
            expected = _expected(plan, "maxpool", x, torch.zeros(plan.second_size).half())
            assert torch.equal(output.to_cpu().view(torch.int16), expected.view(torch.int16))
            actual = guard.to_cpu()
            assert torch.all(actual[:16] == 7) and torch.all(actual[-16:] == 7)


@pytest.mark.parametrize("blocks", [1, 8])
@pytest.mark.parametrize(
    "height,width,channels",
    [
        (1, 1, 16),
        (3, 5, 80),
        (5, 7, 176),
        (20, 20, 256),
        (40, 40, 128),
        (1, 17, 1024),
        (1, 3, 1040),
    ],
)
def test_310p_upsample_channel_pairs_exact_bits(tmp_path, height, width, channels, blocks):
    """Reuse both pairs without altering FP16 bits or strided destination gaps."""
    require_310p()
    module = load_plan("layout")
    plan = module.VectorPlan(
        module.VectorOp.UPSAMPLE,
        height * 2,
        width * 2,
        channels,
        channels + 32,
        channels + 64,
        16,
        32,
        input_height=height,
        input_width=width,
    )
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        call = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        source = session.empty((plan.input_size,), torch.float16)
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.output_size,), 16)
        zero = session.upload(torch.zeros(16, dtype=torch.float16))
        meta = session.upload(plan.metadata())
        for seed in (467, 479):
            g = torch.Generator().manual_seed(seed)
            bits = torch.randint(-32768, 32768, (plan.input_size,), generator=g, dtype=torch.int16)
            # Arithmetic copies can change these zero, infinity and NaN bits.
            bits[16:24] = torch.tensor([0, -32768, 31744, -1024, 31745, -1023, 1, -32767], dtype=torch.int16)
            x = bits.view(torch.float16)
            source.copy_from(x)
            output.copy_from(torch.full((plan.output_size,), 7.0, dtype=torch.float16))
            call(meta, source, zero, output, blocks=blocks)
            expected = _expected(plan, "upsample", x, torch.zeros(plan.second_size).half())
            assert torch.equal(output.to_cpu().view(torch.int16), expected.view(torch.int16))
            actual = guard.to_cpu()
            assert torch.all(actual[:16] == 7) and torch.all(actual[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
