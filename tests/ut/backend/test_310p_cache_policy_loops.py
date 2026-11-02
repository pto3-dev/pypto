# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A1 bypass offset through real SPMD/pipeline entries, checked as raw bytes."""

import json

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto._a1_cache_policy import a1_cache_argument_count
from pypto.backend import BackendType
from pypto.runtime import a1_runner
from pypto.runtime.pto_isa import ensure_pto_isa_root

from .a1_cache_policy_support import capture_entries, run_cpu_oracle


def cache_kernel(dtype, mode):
    @pl.jit.inline
    def pair(block: pl.Scalar[pl.INDEX], source: pl.Tensor[[16, 16], dtype], out: pl.Tensor[[16, 16], dtype]):
        first = pl.load(source, [block * 2, 0], [1, 16], cache=pl.CachePolicy.BYPASS)
        second = pl.load(source, [block * 2 + 1, 0], [1, 16], cache=pl.CachePolicy.DEFAULT)
        pl.store(first, [block * 2, 0], out)
        pl.store(second, [block * 2 + 1, 0], out)
        return out

    if mode == "spmd":

        @pl.jit
        def cache_spmd(source: pl.Tensor[[16, 16], dtype], out: pl.Out[pl.Tensor[[16, 16], dtype]]):
            for block in pl.spmd(8):
                out = pair(block, source, out)
            return out

        return cache_spmd

    @pl.jit.incore
    def cache_pipeline(source: pl.Tensor[[16, 16], dtype], out: pl.Out[pl.Tensor[[16, 16], dtype]]):
        for block in pl.pipeline(8, stage=2):
            out = pair(block, source, out)
        return out

    return cache_pipeline


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32, pl.INT32, pl.UINT32])
@pytest.mark.parametrize("mode", ["spmd", "pipeline"])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP]
)
def test_cache_offset_raw_bits_and_native_entry(
    tmp_path, dtype, mode, planner, pass_verification_instruments, monkeypatch
):
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(
            cache_kernel(dtype, mode).specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
        )
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    assert a1_cache_argument_count(tmp_path, sources[0]) == 2
    pto = (tmp_path / "ptoas" / f"{sources[0].stem}.pto").read_text()
    hinted = pto.count("cache_policy = #pto.load_cache_policy<l2_bypass>")
    assert hinted > 0 and pto.count("offset = %__pypto_l2_cache_offset : i64") == hinted
    blocks, order = 1, [0, 1]
    if mode == "spmd":
        manifest = json.loads((tmp_path / "a1_spmd.json").read_text())
        assert manifest["blocks"] == 8 and manifest["task_count"] == 1
        assert manifest["argument_count"] == 2
        order = manifest["argument_order"]
        assert sorted(order) == [0, 1]
        blocks = 8
    captured = capture_entries(monkeypatch, tmp_path)
    image = a1_runner._compile_image(tmp_path, sources[0], ensure_pto_isa_root())
    assert image.startswith(b"\x7fELF")
    run_cpu_oracle(captured[-1], tmp_path / "cpu", dtype, rows=16, blocks=blocks, order=order)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
