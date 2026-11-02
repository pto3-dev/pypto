# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Existing public bypass reads on A1: assembly, static ABI and native entry."""

import json

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto._a1_cache_policy import a1_cache_argument_count
from pypto.backend import BackendType
from pypto.runtime import a1_runner
from pypto.runtime.pto_isa import ensure_pto_isa_root

from .a1_cache_policy_support import capture_entries, run_cpu_oracle


def copy_kernel(dtype, bypass):
    policy = pl.CachePolicy.BYPASS if bypass else pl.CachePolicy.DEFAULT

    @pl.jit.incore
    def copy(x: pl.Tensor[[2, 16], dtype], out: pl.Out[pl.Tensor[[2, 16], dtype]]):
        first = pl.load(x, [0, 0], [1, 16], cache=policy)
        second = pl.load(x, [1, 0], [1, 16], cache=pl.CachePolicy.DEFAULT)
        pl.store(first, [0, 0], out)
        pl.store(second, [1, 0], out)
        return out

    return copy


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32, pl.INT32, pl.UINT32])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP]
)
def test_bypass_uses_one_scalar_and_only_adjusts_declared_read(
    tmp_path, dtype, planner, pass_verification_instruments, monkeypatch
):
    captured = capture_entries(monkeypatch, tmp_path)
    for bypass in (False, True):
        directory = tmp_path / ("bypass" if bypass else "plain")
        with passes.PassContext(pass_verification_instruments, memory_planner=planner):
            ir.compile(
                copy_kernel(dtype, bypass).specialize(),
                backend_type=BackendType.Ascend310P,
                output_dir=str(directory),
            )
        sources = list((directory / "kernels").glob("*/*.cpp"))
        assert len(sources) == 1
        pto = (directory / "ptoas" / f"{sources[0].stem}.pto").read_text()
        code = sources[0].read_text()
        assert code.count("TLOAD(") == 2 and code.count("TSTORE(") == 2
        assert '#include "intrinsic.h"' not in code
        if bypass:
            assert pto.count("cache_policy = #pto.load_cache_policy<l2_bypass>") == 1
            assert pto.count("offset = %__pypto_l2_cache_offset : i64") == 1
            assert "uint64_t __pypto_l2_cache_offset" in code
            assert a1_cache_argument_count(directory, sources[0]) == 2
        else:
            assert "__pypto_l2_cache_offset" not in code
            assert a1_cache_argument_count(directory, sources[0]) is None
        params = json.loads((directory / "compiled_meta.json").read_text())["params"]
        assert len(params) == 2 and all(param["shape"] == [2, 16] for param in params)
        image = a1_runner._compile_image(directory, sources[0], ensure_pto_isa_root())
        assert image.startswith(b"\x7fELF")
        run_cpu_oracle(captured[-1], directory / "cpu-single", dtype, rows=2, blocks=1, bypass=bypass)
        # Both legacy ABI1 and cache ABI2 use the same public independent-entry
        # builder. Check its actual source for eight distinct image signatures.
        batch = a1_runner._compile_image(
            directory, sources[0], ensure_pto_isa_root(), independent_instances=8, argument_count=2
        )
        assert batch.startswith(b"\x7fELF")
        run_cpu_oracle(
            captured[-1], directory / "cpu-batch", dtype, rows=2, blocks=8, bypass=bypass, instances=8
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
