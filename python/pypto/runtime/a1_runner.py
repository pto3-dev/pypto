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
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypto._external_source import external_source_digest

if TYPE_CHECKING:
    from pypto.ir import CompiledProgram

import torch

from .pto_isa import ensure_pto_isa_root


def _load_tools() -> tuple[Any, Any]:
    from simpler_setup import (  # pyright: ignore[reportMissingImports]  # noqa: PLC0415
        A1SingleLaunch,
        KernelCompiler,
    )

    return KernelCompiler, A1SingleLaunch


def _external_include_dirs(work_dir: Path) -> list[str]:
    manifest = work_dir / "a1_external_kernel.json"
    if not manifest.exists():
        return []
    metadata = json.loads(manifest.read_text())
    directories = metadata.get("include_dirs")
    if (
        metadata.get("schema") != 1
        or not isinstance(directories, list)
        or not all(isinstance(x, str) for x in directories)
    ):
        raise ValueError("invalid A1 external include manifest")
    return directories


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

    _, launcher_type = _load_tools()
    image = _compile_image(work_dir, kernels[0], ensure_pto_isa_root())
    with launcher_type(image, device_id=device_id) as runtime:
        runtime(*args)


def _compile_image(work_dir: Path, kernel: Path, isa_root: str) -> bytes:
    compiler_type, _ = _load_tools()
    compiler = compiler_type("a1")
    with tempfile.TemporaryDirectory(prefix="pypto_a1_entry_") as directory:
        entry = Path(directory) / "entry.cpp"
        source = kernel.read_text()
        source += (
            '\nextern "C" __global__ AICORE void PyptoA1TensorEntry(__gm__ int64_t* args) {\n'
            "    kernel_entry(args);\n}\n"
        )
        entry.write_text(source)
        return compiler.compile_single_launch(
            str(entry), isa_root, extra_include_dirs=[str(kernel.parent), *_external_include_dirs(work_dir)]
        )


def _resident_artifact(compiled: "CompiledProgram") -> tuple[Path, Path, list[dict[str, Any]], str]:
    work_dir = Path(compiled.output_dir)
    if (work_dir / "kernel_config.py").exists() or (work_dir / "orchestration").exists():
        raise ValueError("A1 resident preparation requires one standalone kernel")
    kernels = sorted((work_dir / "kernels").glob("*/*.cpp"))
    metadata = json.loads((work_dir / "compiled_meta.json").read_text())
    if len(kernels) != 1 or metadata.get("platform") != "a1" or metadata.get("backend_type") != "Ascend310P":
        raise ValueError("A1 resident preparation requires one Ascend310P kernel")
    params = metadata.get("params")
    if not isinstance(params, list) or not params:
        raise ValueError("invalid resident kernel signature")
    for param in params:
        shape = param.get("shape") if isinstance(param, dict) else None
        if (
            not isinstance(shape, list)
            or not 1 <= len(shape) <= 5
            or any(type(x) is not int or x <= 0 for x in shape)
            or param.get("dtype") not in ("fp16", "fp32")
        ):
            raise ValueError("resident kernel signature requires static FP16/FP32 tensors")
    key = external_source_digest(
        [kernels[0]], include_dirs=[kernels[0].parent, *_external_include_dirs(work_dir)]
    )
    return work_dir, kernels[0], params, key


class A1ResidentKernel:
    """One static compiled signature bound to a resident session image."""

    def __init__(self, session: "A1ResidentSession", kernel_id: int, params: list[dict[str, Any]]):
        self._session = session
        self._kernel_id = kernel_id
        self._params = params

    def __call__(self, *tensors: Any, blocks: int = 1) -> None:
        """Execute on matching resident tensors without implicit transfers."""
        with self._session._lock:
            self._session._require_open()
            if len(tensors) != len(self._params):
                raise ValueError("resident argument count does not match the static signature")
            for tensor, param in zip(tensors, self._params, strict=True):
                dtype = {"fp16": torch.float16, "fp32": torch.float32}[param["dtype"]]
                if (
                    list(getattr(tensor, "shape", ())) != param["shape"]
                    or getattr(tensor, "dtype", None) != dtype
                ):
                    raise ValueError("resident tensor does not match the static signature")
            self._session._runtime.run_device(*tensors, kernel_id=self._kernel_id, blocks=blocks)


class A1ResidentSession:
    """Prepare static PTO3 kernels sharing session-owned device tensors.

    CPU inputs/outputs require explicit upload/download. Images are cached only
    inside this session by source/include content. The selected ISA/toolchain
    is a session snapshot and must remain unchanged until close. Calls are
    synchronous; this API does not perform graph capture or device scheduling.
    """

    def __init__(self, compiled: "CompiledProgram", device_id: int = 0):
        work_dir, kernel, _, key = _resident_artifact(compiled)
        self._lock = threading.RLock()
        self._closed = False
        self._isa_root = ensure_pto_isa_root()
        _, launcher_type = _load_tools()
        self._runtime = launcher_type(_compile_image(work_dir, kernel, self._isa_root), device_id=device_id)
        self._images = {key: 0}

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("A1 resident session is closed")

    def prepare(self, compiled: "CompiledProgram") -> A1ResidentKernel:
        """Bind a compiled static signature, registering its image if needed."""
        with self._lock:
            self._require_open()
            work_dir, kernel, params, key = _resident_artifact(compiled)
            if key not in self._images:
                self._images[key] = self._runtime.register_kernel(
                    _compile_image(work_dir, kernel, self._isa_root)
                )
            return A1ResidentKernel(self, self._images[key], params)

    def upload(self, tensor: torch.Tensor) -> Any:
        """Upload one contiguous CPU tensor once."""
        with self._lock:
            self._require_open()
            return self._runtime.upload(tensor)

    def empty(self, shape: tuple[int, ...], dtype: torch.dtype) -> Any:
        """Allocate a dense resident tensor without initialization or transfer."""
        with self._lock:
            self._require_open()
            return self._runtime.empty(shape, dtype)

    def close(self) -> None:
        """Drain and release all images/tensors; failures remain retryable."""
        with self._lock:
            if not self._closed:
                self._runtime.close()
                self._closed = True

    def __enter__(self) -> "A1ResidentSession":
        self._require_open()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
