# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public standalone external-source execution on Ascend310P."""

import ctypes
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession


def test_external_compile_and_call(tmp_path):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    if not acl.aclrtGetSocName().startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    source = Path(__file__).parent / "kernels/a1/resident_add.cpp"

    @pl.jit.extern(core_type="aic", source=source)
    def native(
        a: pl.Tensor[[1, 256], pl.FP32],
        b: pl.Tensor[[1, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
    ): ...

    compiled = ir.compile(
        native.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    assert (tmp_path / "a1_external_kernel.json").is_file()
    assert len(list((tmp_path / "kernels").glob("*/*.cpp"))) == 1
    x = torch.arange(256).reshape(1, 256).float() / 16
    y = x.flip([1]).contiguous()
    out = torch.zeros_like(x)
    compiled(x, y, out)
    assert torch.equal(out, x + y)
    reloaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    reloaded(x * 2, y, out)
    assert torch.equal(out, x * 2 + y)

    with A1ResidentSession(compiled) as session:
        add = session.prepare(compiled)
        da, db = session.upload(x), session.upload(y)
        dc = session.empty((1, 256), torch.float32)
        add(da, db, dc)
        add(dc, db, dc)
        assert torch.equal(dc.to_cpu(), x + 2 * y)
        da.copy_from(x * 2)
        reloaded_add = session.prepare(reloaded)
        reloaded_add(da, db, dc)
        assert torch.equal(dc.to_cpu(), x * 2 + y)
        with pytest.raises(ValueError, match="signature"):
            add(da.view((16, 16)), db, dc)
    with pytest.raises(RuntimeError, match="closed"):
        add(da, db, dc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
