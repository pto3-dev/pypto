# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pack independent attention routing work with the public SPMD primitive."""

from typing import Any

import pypto.language as pl
from pypto import ir
from pypto.backend import BackendType

from .executor import Launch


@pl.jit.inline
def _pack_heads(
    x: pl.Tensor[[400, 256], pl.FP16],
    q0: pl.Tensor[[400, 32], pl.FP16],
    k0: pl.Tensor[[400, 32], pl.FP16],
    v0: pl.Tensor[[400, 64], pl.FP16],
    q1: pl.Tensor[[400, 32], pl.FP16],
    k1: pl.Tensor[[400, 32], pl.FP16],
    v1: pl.Tensor[[400, 64], pl.FP16],
):
    for work in pl.range(25):
        row = work * 16
        scale = pl.tile.full([16, 32], pl.FP32, 0.1767766952966369)
        q = pl.load(x, [row, 0], [16, 32])
        q0 = pl.store(
            pl.cast(pl.mul(pl.cast(q, pl.FP32, mode="none"), scale), pl.FP16, mode="none"), [row, 0], q0
        )
        k0 = pl.store(pl.load(x, [row, 32], [16, 32]), [row, 0], k0)
        v0 = pl.store(pl.load(x, [row, 64], [16, 64]), [row, 0], v0)
        q = pl.load(x, [row, 128], [16, 32])
        q1 = pl.store(
            pl.cast(pl.mul(pl.cast(q, pl.FP32, mode="none"), scale), pl.FP16, mode="none"), [row, 0], q1
        )
        k1 = pl.store(pl.load(x, [row, 160], [16, 32]), [row, 0], k1)
        v1 = pl.store(pl.load(x, [row, 192], [16, 64]), [row, 0], v1)
    return q0, k0, v0, q1, k1, v1


# A1 uses flat tensor descriptors, so each image binds one source and six
# outputs explicitly. Tensor tuples are not part of this dispatch ABI.
@pl.jit
def pack_attention8(  # noqa: PLR0913
    x0: pl.Tensor[[400, 256], pl.FP16],
    q0_0: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k0_0: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v0_0: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    q1_0: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k1_0: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v1_0: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    x1: pl.Tensor[[400, 256], pl.FP16],
    q0_1: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k0_1: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v0_1: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    q1_1: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k1_1: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v1_1: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    x2: pl.Tensor[[400, 256], pl.FP16],
    q0_2: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k0_2: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v0_2: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    q1_2: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k1_2: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v1_2: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    x3: pl.Tensor[[400, 256], pl.FP16],
    q0_3: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k0_3: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v0_3: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    q1_3: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k1_3: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v1_3: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    x4: pl.Tensor[[400, 256], pl.FP16],
    q0_4: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k0_4: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v0_4: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    q1_4: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k1_4: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v1_4: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    x5: pl.Tensor[[400, 256], pl.FP16],
    q0_5: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k0_5: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v0_5: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    q1_5: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k1_5: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v1_5: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    x6: pl.Tensor[[400, 256], pl.FP16],
    q0_6: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k0_6: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v0_6: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    q1_6: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k1_6: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v1_6: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    x7: pl.Tensor[[400, 256], pl.FP16],
    q0_7: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k0_7: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v0_7: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
    q1_7: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    k1_7: pl.Out[pl.Tensor[[400, 32], pl.FP16]],
    v1_7: pl.Out[pl.Tensor[[400, 64], pl.FP16]],
):
    for block in pl.spmd(8):
        if block == 0:
            q0_0, k0_0, v0_0, q1_0, k1_0, v1_0 = _pack_heads(x0, q0_0, k0_0, v0_0, q1_0, k1_0, v1_0)
        elif block == 1:
            q0_1, k0_1, v0_1, q1_1, k1_1, v1_1 = _pack_heads(x1, q0_1, k0_1, v0_1, q1_1, k1_1, v1_1)
        elif block == 2:
            q0_2, k0_2, v0_2, q1_2, k1_2, v1_2 = _pack_heads(x2, q0_2, k0_2, v0_2, q1_2, k1_2, v1_2)
        elif block == 3:
            q0_3, k0_3, v0_3, q1_3, k1_3, v1_3 = _pack_heads(x3, q0_3, k0_3, v0_3, q1_3, k1_3, v1_3)
        elif block == 4:
            q0_4, k0_4, v0_4, q1_4, k1_4, v1_4 = _pack_heads(x4, q0_4, k0_4, v0_4, q1_4, k1_4, v1_4)
        elif block == 5:
            q0_5, k0_5, v0_5, q1_5, k1_5, v1_5 = _pack_heads(x5, q0_5, k0_5, v0_5, q1_5, k1_5, v1_5)
        elif block == 6:
            q0_6, k0_6, v0_6, q1_6, k1_6, v1_6 = _pack_heads(x6, q0_6, k0_6, v0_6, q1_6, k1_6, v1_6)
        elif block == 7:
            q0_7, k0_7, v0_7, q1_7, k1_7, v1_7 = _pack_heads(x7, q0_7, k0_7, v0_7, q1_7, k1_7, v1_7)

    return (
        q0_0,
        k0_0,
        v0_0,
        q1_0,
        k1_0,
        v1_0,
        q0_1,
        k0_1,
        v0_1,
        q1_1,
        k1_1,
        v1_1,
        q0_2,
        k0_2,
        v0_2,
        q1_2,
        k1_2,
        v1_2,
        q0_3,
        k0_3,
        v0_3,
        q1_3,
        k1_3,
        v1_3,
        q0_4,
        k0_4,
        v0_4,
        q1_4,
        k1_4,
        v1_4,
        q0_5,
        k0_5,
        v0_5,
        q1_5,
        k1_5,
        v1_5,
        q0_6,
        k0_6,
        v0_6,
        q1_6,
        k1_6,
        v1_6,
        q0_7,
        k0_7,
        v0_7,
        q1_7,
        k1_7,
        v1_7,
    )


def pack_batch_attention(batch: Any) -> list[Launch]:
    """Replace only audited ready QKV routes; retain every matrix dependency."""
    pattern = [
        "vector:COPY",
        "vector:COPY",
        "vector:COPY",
        "vector:SCALE_HALF",
        "matrix",
        "softmax:matrix",
        "matrix",
        "vector:COPY",
        "vector:COPY",
        "vector:COPY",
        "vector:SCALE_HALF",
        "matrix",
        "softmax:matrix",
        "matrix",
    ]
    trace = batch.images[0].trace
    starts = [
        i
        for i in range(len(trace) - len(pattern) + 1)
        if [t.label for t in trace[i : i + len(pattern)]] == pattern
    ]
    if len(starts) != 2:
        raise ValueError("SPMD attention packing requires two audited 400-position attention groups")
    compiled = ir.compile(
        pack_attention8.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(batch.images[0].output_dir / "spmd-attention"),
        dump_passes=False,
    )
    kernel = batch.session.prepare(compiled)
    remove = {start + offset for start in starts for offset in (0, 1, 2, 3, 7, 8, 9, 10)}
    replacements: dict[int, Launch] = {}
    for start in starts:
        arguments: list[Any] = []
        for image in batch.images:
            group = image.trace[start : start + len(pattern)]
            if [t.label for t in group] != pattern:
                raise ValueError("SPMD attention image traces must match")
            source = group[0].arguments[1]
            for index, channels, offset in (
                (0, 32, 0),
                (1, 32, 32),
                (2, 64, 64),
                (7, 32, 128),
                (8, 32, 160),
                (9, 64, 192),
            ):
                metadata = group[index].arguments[0].to_cpu().reshape(-1).tolist()
                if (
                    metadata[:8] != [1, 20, 20, channels, 256, channels, offset, 0]
                    or group[index].arguments[1] is not source
                ):
                    raise ValueError("SPMD attention requires unmodified aligned QKV slices")
            for copy, scale in ((0, 3), (7, 10)):
                metadata = group[scale].arguments[0].to_cpu().reshape(-1).tolist()
                if (
                    group[scale].arguments[1] is not group[copy].arguments[-1]
                    or metadata[15] != 0.1767766922712326
                ):
                    raise ValueError("SPMD attention requires the original FP32 query scale")
            arguments.append(source.view((400, 256)))
            for index, columns in ((3, 32), (1, 32), (2, 64), (10, 32), (8, 32), (9, 64)):
                arguments.append(group[index].arguments[-1].view((400, columns)))
        replacements[start] = Launch(kernel, tuple(arguments), 8, "spmd:attention-pack")
    batch._spmd_attention_replacements = [
        dict(parent_indices=[start + j for j in (0, 1, 2, 3, 7, 8, 9, 10)]) for start in starts
    ]
    return [
        replacements[i] if i in replacements else task
        for i, task in enumerate(batch.trace)
        if i not in remove or i in replacements
    ]
