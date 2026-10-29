# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""310P runtime branch joins using public PyPTO vector tiles."""

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import require_310p


def test_310p_vec_copy_controlflow(tmp_path):
    require_310p()

    @pl.jit.incore
    def joined(a: pl.Tensor[[8, 80], pl.FP32], output: pl.Out[pl.Tensor[[8, 80], pl.FP32]]):
        index = pl.tile.get_block_idx()
        value = pl.load(a, [index, 0], [1, 80])
        if index % 2 == 0:
            value = pl.add(value, value)
        else:
            value = pl.sub(value, value)
        output = pl.store(value, [index, 0], output)
        return output

    compiled = ir.compile(joined.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    source = next(tmp_path.rglob("*.pto")).read_text()
    assert "pto.tmov" in source
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(compiled)
        inp = session.empty((8, 80), torch.float32)
        out = session.empty((8, 80), torch.float32)
        for seed in (3263, 3269):
            values = torch.randn((8, 80), generator=torch.Generator().manual_seed(seed))
            expected = values.clone()
            expected[::2] = values[::2] + values[::2]
            expected[1::2] = values[1::2] - values[1::2]
            inp.copy_from(values)
            out.copy_from(torch.full((8, 80), float("nan")))
            kernel(inp, out, blocks=8)
            assert torch.equal(out.to_cpu().view(torch.int32), expected.view(torch.int32))
