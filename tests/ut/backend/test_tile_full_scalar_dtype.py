# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The scalar emitted for a floating tile.full matches the requested tile dtype."""

import re

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("dtype,expected", [(pl.FP16, "f16"), (pl.FP32, "f32")])
@pytest.mark.parametrize("backend", [BackendType.Ascend310P, BackendType.Ascend910B])
def test_full_float_scalar_dtype(tmp_path, keyword, dtype, expected, backend):
    if keyword:

        @pl.jit.incore
        def kernel(out: pl.Out[pl.Tensor[[16, 16], dtype]]):
            value = pl.tile.full([16, 16], dtype=dtype, value=0.125)
            return pl.store(value, [0, 0], out)
    else:

        @pl.jit.incore
        def kernel(out: pl.Out[pl.Tensor[[16, 16], dtype]]):
            value = pl.tile.full([16, 16], dtype, 0.125)
            return pl.store(value, [0, 0], out)

    ir.compile(kernel.specialize(), backend_type=backend, output_dir=str(tmp_path), skip_ptoas=True)
    pto = "\n".join(path.read_text() for path in tmp_path.rglob("*.pto"))
    assert re.search(rf"pto\.texpands ins\([^:]+: {expected}\)", pto)


@pytest.mark.parametrize("fill", [0, 7])
@pytest.mark.parametrize("dtype,expected", [(pl.FP16, "f16"), (pl.FP32, "f32")])
@pytest.mark.parametrize("backend", [BackendType.Ascend310P, BackendType.Ascend910B])
def test_integer_literal_for_float_fill(tmp_path, fill, dtype, expected, backend):
    @pl.jit.incore
    def kernel(out: pl.Out[pl.Tensor[[16, 16], dtype]]):
        value = pl.tile.full([16, 16], dtype, fill)
        return pl.store(value, [0, 0], out)

    ir.compile(kernel.specialize(), backend_type=backend, output_dir=str(tmp_path), skip_ptoas=True)
    pto = "\n".join(path.read_text() for path in tmp_path.rglob("*.pto"))
    assert re.search(rf"pto\.texpands ins\([^:]+: {expected}\)", pto)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
