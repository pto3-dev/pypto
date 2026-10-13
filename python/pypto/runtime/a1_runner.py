# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronous execution of an A1 compilation containing one static kernel."""

import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from .pto_isa import ensure_pto_isa_root


def _load_tools() -> tuple[Any, Any]:
    from simpler_setup import (  # pyright: ignore[reportMissingImports]  # noqa: PLC0415
        A1SingleLaunch,
        KernelCompiler,
    )

    return KernelCompiler, A1SingleLaunch


def execute_single_launch(work_dir: Path, args: list[Any], device_id: int) -> None:
    """Execute a single InCore entry using CPU FP16/FP32 tensor carriers.

    Graph orchestration and device-resident arguments are outside this path.
    The whole object uses the runtime's pinned ISA and its kernel tensor ABI.
    All carriers must match the persisted static signature before native
    compilation or device allocation begins.
    """
    if (work_dir / "kernel_config.py").exists() or (work_dir / "orchestration").exists():
        raise ValueError("Ascend310P graph execution is not yet supported by the single-launch runtime")
    kernels = sorted((work_dir / "kernels").glob("*/*.cpp"))
    if len(kernels) != 1:
        raise ValueError("Ascend310P single launch requires exactly one InCore kernel")
    metadata = json.loads((work_dir / "compiled_meta.json").read_text())
    if metadata.get("platform") != "a1" or metadata.get("backend_type") != "Ascend310P":
        raise ValueError("single launch requires an Ascend310P compilation")
    params = metadata["params"]
    if len(args) != len(params):
        raise ValueError("argument count does not match the A1 kernel signature")
    for arg, param in zip(args, params, strict=True):
        if not isinstance(arg, torch.Tensor):
            raise TypeError("Ascend310P single launch requires torch tensors")
        if arg.device.type != "cpu" or not arg.is_contiguous():
            raise ValueError("Ascend310P single launch requires contiguous CPU tensors")
        dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(param["dtype"])
        if dtype is None or arg.dtype != dtype or list(arg.shape) != param["shape"]:
            raise ValueError("tensor carrier does not match the static A1 kernel signature")

    compiler_type, launcher_type = _load_tools()
    compiler = compiler_type("a1")
    isa_root = ensure_pto_isa_root()
    with tempfile.TemporaryDirectory(prefix="pypto_a1_entry_") as directory:
        entry = Path(directory) / "entry.cpp"
        source = kernels[0].read_text()
        source += (
            '\nextern "C" __global__ AICORE void PyptoA1TensorEntry(__gm__ int64_t* args) {\n'
            "    kernel_entry(args);\n}\n"
        )
        entry.write_text(source)
        image = compiler.compile_single_launch(
            str(entry), isa_root, extra_include_dirs=[str(kernels[0].parent)]
        )
    with launcher_type(image, device_id=device_id) as runtime:
        runtime(*args)
