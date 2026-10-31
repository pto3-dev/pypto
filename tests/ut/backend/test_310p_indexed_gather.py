# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Existing public indexed gather on A1: full compilation and real Tensor ABI."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir


def flat_gather_kernel(dtype: Any, panels: int = 5) -> Any:
    rows = panels * 8

    @pl.jit.incore
    def indexed_gather(
        source: pl.Tensor[[8 * 112], dtype],
        indices: pl.Tensor[[rows * 112], pl.INT32],
        output: pl.Out[pl.Tensor[[rows * 128], dtype]],
    ):
        source_view = pl.tensor.view(source, [8, 112])
        index_view = pl.tensor.view(indices, [rows, 112])
        output_view = pl.tensor.view(output, [rows, 128])
        table = pl.load(source_view, [0, 16], [8, 80])
        for block in pl.range(pl.tile.get_block_idx(), panels, pl.tile.get_block_num()):
            positions = pl.load(index_view, [block * 8, 8], [8, 96])
            values = pl.gather(table, index=positions)
            pl.tensor.assemble(output_view, values, [block * 8, 16])
        return output

    return indexed_gather


def test_public_flat_gather_compile(tmp_path):
    ir.compile(
        flat_gather_kernel(pl.FP32).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
    )
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    assert "TGATHER(" in sources[0].read_text()


def tile_gather_kernel(dtype: Any, scratch_columns: int = 192, scratch_dtype: Any = pl.INT32) -> Any:
    @pl.jit.incore
    def tile_indexed_gather(
        source: pl.Tensor[[8 * 112], dtype],
        indices: pl.Tensor[[40 * 112], pl.INT32],
        output: pl.Out[pl.Tensor[[40 * 128], dtype]],
    ):
        source_view = pl.tensor.view(source, [8, 112])
        index_view = pl.tensor.view(indices, [40, 112])
        output_view = pl.tensor.view(output, [40, 128])
        table = pl.load(source_view, [0, 16], [8, 80])
        for block in pl.range(pl.tile.get_block_idx(), 5, pl.tile.get_block_num()):
            positions = pl.load(index_view, [block * 8, 8], [8, 96])
            scratch = pl.create_tile([8, scratch_columns], scratch_dtype)
            values = pl.tile.gather(table, positions, scratch)
            pl.store(values, [block * 8, 16], output_view)
        return output

    return tile_indexed_gather


def spmd_gather_kernel(dtype: Any) -> Any:
    @pl.jit
    def gather_spmd(
        source: pl.Tensor[[8 * 112], dtype],
        indices: pl.Tensor[[64 * 112], pl.INT32],
        output: pl.Out[pl.Tensor[[64 * 128], dtype]],
    ):
        for block in pl.spmd(8):
            source_view = pl.tensor.view(source, [8, 112])
            index_view = pl.tensor.view(indices, [64, 112])
            output_view = pl.tensor.view(output, [64, 128])
            table = pl.load(source_view, [0, 16], [8, 80])
            positions = pl.load(index_view, [block * 8, 8], [8, 96])
            values = pl.gather(table, index=positions)
            pl.tensor.assemble(output_view, values, [block * 8, 16])
        return output

    return gather_spmd


def pipeline_gather_kernel(dtype: Any) -> Any:
    @pl.jit.incore
    def gather_pipeline(
        source: pl.Tensor[[8 * 112], dtype],
        indices: pl.Tensor[[40 * 112], pl.INT32],
        output: pl.Out[pl.Tensor[[40 * 128], dtype]],
    ):
        source_view = pl.tensor.view(source, [8, 112])
        index_view = pl.tensor.view(indices, [40, 112])
        output_view = pl.tensor.view(output, [40, 128])
        table = pl.load(source_view, [0, 16], [8, 80])
        for block in pl.pipeline(5, stage=2):
            positions = pl.load(index_view, [block * 8, 8], [8, 96])
            values = pl.gather(table, index=positions)
            pl.tensor.assemble(output_view, values, [block * 8, 16])
        return output

    return gather_pipeline


_CASES = [
    (dtype, mode) for dtype in (pl.FP16, pl.FP32, pl.INT32) for mode in ("flat", "tile", "spmd", "pipeline")
]
_FACTORIES = {
    "flat": flat_gather_kernel,
    "tile": tile_gather_kernel,
    "spmd": spmd_gather_kernel,
    "pipeline": pipeline_gather_kernel,
}


@pytest.fixture(scope="module", params=_CASES)
def compiled_gather(request, tmp_path_factory):
    assert shutil.which("ptoas"), "requires PTOAS for complete public compilation"
    dtype, mode = request.param
    output = tmp_path_factory.mktemp("public-gather")
    ir.compile(
        _FACTORIES[mode](dtype).specialize(), backend_type=BackendType.Ascend310P, output_dir=str(output)
    )
    sources = list((output / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert "TGATHER(" in code
    assert "set_flag(PIPE_MTE2, PIPE_S," in code and "wait_flag(PIPE_MTE2, PIPE_S," in code
    assert "wait_flag(PIPE_MTE2, PIPE_V," in code
    assert "wait_flag(PIPE_V, PIPE_MTE3," in code
    # Public full-shape A1 lowering needs neither metadata mutation nor a copy
    # through floating extract/integer arithmetic to pack a dense producer.
    assert "TSETVALIDSHAPE(" not in code and "TEXTRACT(" not in code
    metadata = json.loads((output / "compiled_meta.json").read_text())
    expected = "fp16" if dtype == pl.FP16 else "fp32" if dtype == pl.FP32 else "int32"
    assert [param["dtype"] for param in metadata["params"]] == [expected, "int32", expected]
    assert metadata["params"][0]["direction"] == metadata["params"][1]["direction"] == "In"
    assert metadata["params"][2]["direction"] in {"Out", "InOut"}
    order = [0, 1, 2]
    if mode == "spmd":
        manifest = json.loads((output / "a1_spmd.json").read_text())
        assert manifest["blocks"] == 8 and manifest["task_count"] == 1 and manifest["argument_count"] == 3
        assert (output / manifest["kernel"]).resolve() == sources[0].resolve()
        order = manifest["argument_order"]
        assert sorted(order) == [0, 1, 2]
    return dtype, mode, sources[0], order


def _includes() -> list[str]:
    runtime = Path(__file__).resolve().parents[3] / "runtime"
    return [
        f"-I{p}"
        for p in (
            pto_isa_include_dir(),
            runtime / "src/a1/platform/include",
            runtime / "src/common",
            runtime / "src/common/task_interface",
            runtime / "src/common/platform/include",
        )
    ]


def test_public_gather_cpu_tensor_abi(compiled_gather, tmp_path):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for CPU instruction semantics"
    dtype, mode, generated, order = compiled_gather
    ctype = "half" if dtype == pl.FP16 else "float" if dtype == pl.FP32 else "int32_t"
    tag = "FLOAT16" if dtype == pl.FP16 else "FLOAT32" if dtype == pl.FP32 else "INT32"
    rows = 64 if mode == "spmd" else 40
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("indexed_gather_oracle.cpp").read_text()
    )
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            f"-DGATHER_CTYPE={ctype}",
            f"-DGATHER_DTYPE=DataType::{tag}",
            f"-DGATHER_ROWS={rows}",
            *[f"-DGATHER_ARG{i}={value}" for i, value in enumerate(order)],
            *_includes(),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cores = (8,) if mode == "spmd" else (1,) if mode == "pipeline" else (1, 3, 8)
    for count in cores:
        subprocess.run(
            [str(binary)],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PTO_CPU_SIM_NUM_CORES": str(count),
                "PTO_CPU_SIM_TRACE_ENABLE": "0",
                "OMP_NUM_THREADS": "1",
            },
        )


def test_public_gather_native_compile(compiled_gather, tmp_path):
    compiler = shutil.which("ccec")
    assert compiler, "requires CCEC for offline dav-m200 compilation"
    _dtype, _mode, source, _order = compiled_gather
    subprocess.run(
        [
            compiler,
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-m200",
            *_includes(),
            str(source),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "columns,dtype,diagnostic",
    [
        (96, pl.INT32, "scratch needs a 32-bit offset"),
        (192, pl.FP32, "requires INT32 indices"),
    ],
)
def test_public_gather_rejects_invalid_scratch(tmp_path, columns, dtype, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        ir.compile(
            tile_gather_kernel(pl.FP32, columns, dtype).specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
