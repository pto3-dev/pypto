# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only independent checks for public channel-slice copy."""

import os
import shutil
import subprocess

import pytest
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir

from examples.models.yolo26_native.executor import NativeYOLO26
from examples.models.yolo26_native.layout import VectorOp, VectorPlan
from examples.models.yolo26_native.public_copy import PublicCopyPlan, copy_kernel


@pytest.mark.parametrize(
    "size,channels,src_stride,src_offset,dst_stride,dst_offset,cores",
    [
        (4, 16, 16, 0, 16, 0, 1),
        (4, 512, 512, 0, 512, 0, 1),
        (4, 16, 1048592, 0, 16, 0, 1),
        (4, 16, 16, 0, 1048592, 0, 1),
        (8, 16, 16, 0, 16, 0, 1),
        (16, 16, 16, 0, 16, 0, 1),
        (16, 32, 128, 32, 64, 16, 1),
        (16, 64, 128, 64, 256, 128, 1),
        *((20, 32, 256, offset, 32, 0, 1) for offset in (0, 32, 128, 160)),
        *((20, 64, 256, offset, 64, 0, 1) for offset in (64, 192)),
    ],
)
def test_public_copy_cpu(tmp_path, size, channels, src_stride, src_offset, dst_stride, dst_offset, cores):
    ptoas, compiler = shutil.which("ptoas"), shutil.which("clang++-19")
    if not ptoas or not compiler:
        pytest.skip("requires PTOAS and clang++-19")
    isa_include = pto_isa_include_dir()
    pixels = size * size
    plan = VectorPlan(VectorOp.COPY, size, size, channels, src_stride, dst_stride, src_offset, dst_offset)
    ir.compile(
        copy_kernel(plan).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
    )
    generated = tmp_path / "generated.cpp"
    subprocess.run(
        [
            ptoas,
            "--pto-arch=a1",
            "--pto-level=level3",
            str(next(tmp_path.rglob("*.pto"))),
            "-o",
            str(generated),
        ],
        check=True,
    )
    # Also instantiate the A1 implementation for the largest supported tile
    # and for both directions of the wide-stride row-DMA fallback.
    ccec = shutil.which("ccec")
    if ccec and (channels == 512 or max(src_stride, dst_stride) > 65535 * 16):
        subprocess.run(
            [
                ccec,
                "-c",
                "-O2",
                "-x",
                "cce",
                "-std=c++17",
                "--cce-aicore-only",
                "--cce-aicore-arch=dav-m200",
                f"-I{isa_include}",
                str(generated),
                "-o",
                str(tmp_path / "boundary.o"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    source = generated.read_text()
    oracle = r"""
int main() {
 constexpr int p=PIXELS, c=CHANNELS, ss=SRC_STRIDE, so=SRC_OFFSET, ds=DST_STRIDE, d=DST_OFFSET;
 std::vector<half> input(p*ss+32,half(7)), output(p*ds+32,half(9)), unused(16,half(3));
 std::vector<float> metadata(32,5.0f);
 for(int i=0;i<p*ss;++i) {
   uint16_t bits=uint16_t(i*251u+0x8000u); std::memcpy(&input[16+i],&bits,sizeof(bits));
 }
 auto original=input;
 for(int b=0;b<CORES;++b) { pto::cpu_sim::set_execution_context(b,0,1);
   copy(metadata.data(),input.data()+16,unused.data(),output.data()+16); }
 for(int i=0;i<p;++i) for(int j=0;j<ds;++j) {
   half expected = (j>=d && j<d+c) ? input[16+i*ss+so+j-d] : half(9);
   if(std::memcmp(&output[16+i*ds+j],&expected,sizeof(half))) return 1;
 }
 if(std::memcmp(input.data(),original.data(),input.size()*sizeof(half))) return 2;
 for(int i=0;i<16;++i) if(output[i]!=half(9)||output[16+p*ds+i]!=half(9)) return 3;
 for(auto v:unused) if(v!=half(3)) return 4;
 for(auto v:metadata) if(v!=5.0f) return 5;
 return 0;
}
"""
    for key, value in {
        "PIXELS": pixels,
        "CHANNELS": channels,
        "SRC_STRIDE": src_stride,
        "SRC_OFFSET": src_offset,
        "DST_STRIDE": dst_stride,
        "DST_OFFSET": dst_offset,
        "CORES": cores,
    }.items():
        oracle = oracle.replace(key, str(value))
    cpp = tmp_path / "oracle.cpp"
    cpp.write_text("#include <cstdint>\n#include <cstring>\n#include <vector>\n" + source + "\n" + oracle)
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-D__CPU_SIM",
            "-D__DAV_VEC__",
            f"-I{isa_include}",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run(
        [str(binary)],
        check=True,
        env={**os.environ, "PTO_CPU_SIM_NUM_CORES": str(cores), "PTO_CPU_SIM_TRACE_ENABLE": "0"},
    )


def test_rejects_non_copy():
    with pytest.raises(ValueError, match="requires a COPY plan"):
        copy_kernel(VectorPlan(VectorOp.ADD, 16, 16))


def test_public_copy_plan_rejects_unsupported_geometry():
    with pytest.raises(ValueError, match="whole 16-pixel tiles"):
        PublicCopyPlan(VectorOp.COPY, 5, 5)
    with pytest.raises(ValueError, match="at most 512 channels"):
        PublicCopyPlan(VectorOp.COPY, 16, 16, channels=528, source_stride=528, destination_stride=528)
    with pytest.raises(ValueError, match="requires a COPY plan"):
        PublicCopyPlan(VectorOp.ADD, 16, 16)


def test_public_copy_dispatch_uses_one_logical_block():
    native = NativeYOLO26.__new__(NativeYOLO26)
    native.experimental_public_copy = True
    source, output, dummy = object(), object(), object()
    calls = []
    native._dummy = lambda: dummy
    native._emit = lambda plan, inputs, output, label, blocks=8: calls.append(
        (plan, inputs, output, label, blocks)
    )
    plan = VectorPlan(VectorOp.COPY, 20, 20, channels=32, source_stride=256, destination_stride=32)

    assert native._vector(plan, source, out=output) is output
    actual, inputs, actual_output, label, blocks = calls[0]
    assert isinstance(actual, PublicCopyPlan)
    assert actual.metadata().equal(plan.metadata())
    assert (inputs, actual_output, label, blocks) == ((source, dummy), output, "vector:COPY", 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
