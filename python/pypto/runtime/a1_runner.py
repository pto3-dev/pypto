# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronous execution of an A1 compilation containing one static kernel."""

import importlib
import json
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pypto._external_source import external_source_digest

if TYPE_CHECKING:
    from pypto.ir import CompiledProgram

import torch

from .pto_isa import ensure_pto_isa_root

_Session = TypeVar("_Session", bound="A1ResidentSession")

_TENSOR_DTYPES = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int32": torch.int32,
    "uint32": torch.uint32,
}


def _load_tools() -> tuple[Any, Any]:
    tools = importlib.import_module("simpler_setup")
    return tools.KernelCompiler, tools.A1SingleLaunch


def _external_include_dirs(work_dir: Path) -> list[str]:
    manifest = work_dir / "a1_external_kernel.json"
    if not manifest.exists():
        return []
    metadata = json.loads(manifest.read_text())
    directories = metadata.get("include_dirs")
    if metadata.get("schema") != 1 or not isinstance(directories, list):
        raise ValueError("invalid A1 external include manifest")
    values = cast(list[object], directories)
    if not all(isinstance(value, str) for value in values):
        raise ValueError("invalid A1 external include manifest")
    return cast(list[str], values)


def execute_single_launch(work_dir: Path, args: list[Any], device_id: int) -> None:
    """Execute a single InCore entry using static CPU tensor carriers.

    Graph orchestration and device-resident arguments are outside this path.
    The whole object uses the runtime's pinned ISA and its kernel tensor ABI.
    All carriers must match the persisted static signature before native
    compilation or device allocation begins.
    """
    spmd_blocks = _spmd_blocks(work_dir)
    if (
        (work_dir / "kernel_config.py").exists() or (work_dir / "orchestration").exists()
    ) and spmd_blocks is None:
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
        dtype = _TENSOR_DTYPES.get(param["dtype"])
        if dtype is None or arg.dtype != dtype or list(arg.shape) != param["shape"]:
            raise ValueError("tensor carrier does not match the static A1 kernel signature")

    if spmd_blocks is not None:
        from pypto.ir import CompiledProgram  # noqa: PLC0415

        compiled = CompiledProgram.from_dir(work_dir)
        with A1L2Worker(compiled, device_id) as worker:
            resident = tuple(worker.upload(arg) for arg in args)
            worker.prepare_spmd(compiled, *resident)()
            for arg, tensor, param in zip(args, resident, params, strict=True):
                if param["direction"] in ("Out", "InOut"):
                    arg.copy_(tensor.to_cpu())
        return

    _, launcher_type = _load_tools()
    image = _compile_image(work_dir, kernels[0], ensure_pto_isa_root())
    with launcher_type(image, device_id=device_id) as runtime:
        runtime(*args)


def _compile_image(
    work_dir: Path,
    kernel: Path,
    isa_root: str,
    *,
    independent_instances: int = 1,
    argument_count: int = 0,
) -> bytes:
    compiler_type, _ = _load_tools()
    compiler = compiler_type("a1")
    with tempfile.TemporaryDirectory(prefix="pypto_a1_entry_") as directory:
        entry = Path(directory) / "entry.cpp"
        source = kernel.read_text()
        if independent_instances > 1:
            # Load instruction/ABI headers before virtualizing the grid: builtin
            # declarations must retain their real names. A one-core kernel body
            # sees grid (0, 1); the global entry selects an independent signature
            # using the physical core. No tensor layout or descriptor ABI changes.
            source = (
                '#include <pto/pto-inst.hpp>\n#include "tensor.h"\n'
                'extern "C" AICORE inline __attribute__((always_inline)) '
                "void kernel_entry(__gm__ int64_t*);\n"
                "#define get_block_idx() 0\n#define get_block_num() 1\n"
                + source
                + "\n#undef get_block_idx\n#undef get_block_num\n"
            )
            call = f"kernel_entry(args + get_block_idx() * {argument_count})"
        else:
            call = "kernel_entry(args)"
        source += (
            '\nextern "C" __global__ AICORE void PyptoA1TensorEntry(__gm__ int64_t* args) {\n'
            f"    {call};\n}}\n"
        )
        entry.write_text(source)
        return compiler.compile_single_launch(
            str(entry), isa_root, extra_include_dirs=[str(kernel.parent), *_external_include_dirs(work_dir)]
        )


def _spmd_blocks(work_dir: Path) -> int | None:
    """Read the compiler's static single-grid contract without executing config."""
    path = work_dir / "a1_spmd.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    blocks = record.get("blocks")
    kernel = record.get("kernel")
    if (
        record.get("schema") != 1
        or record.get("task_count") != 1
        or type(blocks) is not int
        or not 1 <= blocks <= 8
        or not isinstance(kernel, str)
        or len(Path(kernel).parts) != 3
        or Path(kernel).parts[0] != "kernels"
        or Path(kernel).parts[1] not in ("aic", "aiv")
        or Path(kernel).suffix != ".cpp"
    ):
        raise ValueError("invalid A1 static SPMD manifest")
    kernels = sorted((work_dir / "kernels").glob("*/*.cpp"))
    metadata = json.loads((work_dir / "compiled_meta.json").read_text())
    count = len(metadata.get("params", []))
    order = record.get("argument_order")
    if not isinstance(order, list):
        raise ValueError("A1 SPMD manifest does not match its kernel signature")
    values = cast(list[object], order)
    if (
        kernels != [work_dir / kernel]
        or record.get("argument_count") != count
        or any(type(index) is not int for index in values)
        or sorted(cast(list[int], values)) != list(range(count))
    ):
        raise ValueError("A1 SPMD manifest does not match its kernel signature")
    return blocks


def _resident_artifact(compiled: "CompiledProgram") -> tuple[Path, Path, list[dict[str, Any]], str]:
    work_dir = Path(compiled.output_dir)
    spmd_blocks = _spmd_blocks(work_dir)
    if (
        (work_dir / "kernel_config.py").exists() or (work_dir / "orchestration").exists()
    ) and spmd_blocks is None:
        raise ValueError("A1 resident preparation requires one standalone kernel or static SPMD grid")
    kernels = sorted((work_dir / "kernels").glob("*/*.cpp"))
    metadata = json.loads((work_dir / "compiled_meta.json").read_text())
    if len(kernels) != 1 or metadata.get("platform") != "a1" or metadata.get("backend_type") != "Ascend310P":
        raise ValueError("A1 resident preparation requires one Ascend310P kernel")
    raw_params = metadata.get("params")
    if not isinstance(raw_params, list) or not raw_params:
        raise ValueError("invalid resident kernel signature")
    records = cast(list[object], raw_params)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("resident kernel signature requires static FP16/FP32/INT32/UINT32 tensors")
        param = cast(dict[str, Any], record)
        shape = param.get("shape")
        if not isinstance(shape, list):
            raise ValueError("resident kernel signature requires static FP16/FP32/INT32/UINT32 tensors")
        extents = cast(list[object], shape)
        if (
            not 1 <= len(extents) <= 5
            or any(type(extent) is not int or extent <= 0 for extent in extents)
            or not isinstance(param.get("dtype"), str)
            or param["dtype"] not in _TENSOR_DTYPES
        ):
            raise ValueError("resident kernel signature requires static FP16/FP32/INT32/UINT32 tensors")
    params = cast(list[dict[str, Any]], records)
    key = external_source_digest(
        [kernels[0]], include_dirs=[kernels[0].parent, *_external_include_dirs(work_dir)]
    )
    return work_dir, kernels[0], params, key


class A1ResidentKernel:
    """One static compiled signature bound to a resident session image."""

    def __init__(
        self,
        session: "A1ResidentSession",
        kernel_id: int,
        params: list[dict[str, Any]],
        *,
        required_blocks: int | None = None,
        argument_order: list[int] | None = None,
    ):
        self._argument_order = argument_order
        self._session = session
        self._kernel_id = kernel_id
        self._params = params
        self._required_blocks = required_blocks

    def _arguments(self, tensors: tuple[Any, ...]) -> tuple[Any, ...]:
        # SPMD outlining may reorder live-ins, especially loop-carried outputs.
        # Keep the public signature stable and permute only native descriptors.
        if self._argument_order is None:
            return tensors
        return tuple(tensors[index] for index in self._argument_order)

    def _validate(self, tensors: tuple[Any, ...], blocks: int) -> None:
        if self._required_blocks is not None and (type(blocks) is not int or blocks != self._required_blocks):
            raise ValueError(f"independent-instance kernel requires blocks={self._required_blocks}")
        if len(tensors) != len(self._params):
            raise ValueError("resident argument count does not match the static signature")
        for tensor, param in zip(tensors, self._params, strict=True):
            dtype = _TENSOR_DTYPES.get(param["dtype"])
            if (
                dtype is None
                or list(getattr(tensor, "shape", ())) != param["shape"]
                or getattr(tensor, "dtype", None) != dtype
            ):
                raise ValueError("resident tensor does not match the static signature")

    def __call__(self, *tensors: Any, blocks: int = 1) -> None:
        """Execute on matching resident tensors without implicit transfers."""
        with self._session._lock:
            self._session._require_open()
            self._validate(tensors, blocks)
            self._session._runtime.run_device(
                *self._arguments(tensors), kernel_id=self._kernel_id, blocks=blocks
            )


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
            blocks = _spmd_blocks(work_dir)
            order = json.loads((work_dir / "a1_spmd.json").read_text())["argument_order"] if blocks else None
            return A1ResidentKernel(
                self, self._images[key], params, required_blocks=blocks, argument_order=order
            )

    def _call_entries(self, calls: Sequence[tuple[object, tuple[Any, ...], int]]) -> list[Any]:
        entries: list[Any] = []
        for kernel, tensors, blocks in calls:
            if not isinstance(kernel, A1ResidentKernel) or kernel._session is not self:
                raise ValueError("sequence kernel must belong to this resident session")
            kernel._validate(tensors, blocks)
            entries.append((kernel._kernel_id, blocks, kernel._arguments(tensors)))
        return entries

    def prepare_sequence(self, calls: list[tuple[A1ResidentKernel, tuple[Any, ...], int]]) -> Any:
        """Bind ordered native calls and immutable device arguments for repeated replay."""
        with self._lock:
            self._require_open()
            return self._runtime.prepare_sequence(self._call_entries(calls))

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

    def __enter__(self: _Session) -> _Session:
        self._require_open()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class A1TaskGraph:
    """Immutable registered static graph owned by one A1 L2 worker.

    Input values may change at the bound device addresses between synchronous
    runs. Shape, dtype, task order and address changes require a new graph.
    """

    def __init__(self, worker: "A1L2Worker", prepared: Any, task_count: int):
        self._worker = worker
        self._prepared = prepared
        self.task_count = task_count

    def __call__(self) -> None:
        self._worker.run(self)

    def rebuild(self) -> None:
        """Refresh the CANN model's profiling state without changing bindings.

        Call after profiler start/stop, outside measured replay windows.
        The static task order, tensor addresses and this graph remain fixed.
        """
        with self._worker._lock:
            self._worker._require_open()
            self._prepared.rebuild()


class A1L2Worker(A1ResidentSession):
    """Single-chip owner of static A1 task graphs and resident tensors/images.

    This backend uses CANN device-model scheduling for coupled 310P cores.
    Task dependencies follow the registered stream order. One model is submitted
    per graph run, with a final wait and no per-task host kernel launches.
    It does not implement dynamic A2/A3 TensorMap/AICPU orchestration or the
    generic ChipWorker ABI. Use prepare() for kernels and register_graph() for
    an ordered list of prepared kernels and session-owned resident arguments.
    """

    level = 2

    def prepare_independent(self, compiled: "CompiledProgram", instances: int) -> A1ResidentKernel:
        """Bind independent single-core invocations to physical 310P cores.

        Each instance receives one complete static signature, flattened in
        instance-major order. Its body runs with logical grid (0, 1). Pass
        ``blocks=instances`` when calling it or registering a graph task.
        Inputs may share read-only storage; outputs and any scratch tensors
        must be independent. Kernels must be correct on one core and must not
        communicate with another instance or depend on physical core identity
        inside a precompiled/header helper. This is explicit batch scheduling,
        not a change to the kernel's arithmetic or tensor layout.

        Args:
            compiled: A standalone Ascend310P artifact, including extern kernels.
            instances: Independent signatures/physical cores, from one to eight.

        Returns:
            A session-owned kernel accepting ``instances`` repeated signatures.

        Raises:
            ValueError: The instance count or flattened signature is unsupported.
            RuntimeError: The worker has been closed.
        """
        if type(instances) is not int or not 1 <= instances <= 8:
            raise ValueError(f"independent instances must be an integer from one to eight, got {instances!r}")
        with self._lock:
            self._require_open()
            work_dir, kernel, params, key = _resident_artifact(compiled)
            if _spmd_blocks(work_dir) is not None:
                raise ValueError("SPMD grids cannot be virtualized as independent single-core instances")
            if len(params) * instances > 64:
                raise ValueError("independent instances require at most 64 flattened tensor arguments")
            if instances == 1:
                return self.prepare(compiled)
            # Include arity: the physical-core argument offset depends on it,
            # even when two static signatures share identical source/includes.
            batch_key = f"independent:{instances}:{len(params)}:{key}"
            if batch_key not in self._images:
                self._images[batch_key] = self._runtime.register_kernel(
                    _compile_image(
                        work_dir,
                        kernel,
                        self._isa_root,
                        independent_instances=instances,
                        argument_count=len(params),
                    )
                )
            return A1ResidentKernel(
                self, self._images[batch_key], params * instances, required_blocks=instances
            )

    def prepare_spmd(self, compiled: "CompiledProgram", *tensors: Any) -> A1TaskGraph:
        """Bind a public ``pl.spmd`` single-grid artifact as one device task.

        Tensors remain resident and explicit. Compose dependent grids with
        prepare() and register_graph(); stream ordering supplies the edge.
        """
        blocks = _spmd_blocks(Path(compiled.output_dir))
        if blocks is None:
            raise ValueError("prepare_spmd requires a compiled static pl.spmd grid")
        return self.register_graph([(self.prepare(compiled), tensors, blocks)])

    def register_graph(self, calls: list[tuple[A1ResidentKernel, tuple[Any, ...], int]]) -> A1TaskGraph:
        """Prepare one device model; retain its stable arguments until close."""
        with self._lock:
            self._require_open()
            prepared = self._runtime.prepare_graph(self._call_entries(calls))
            return A1TaskGraph(self, prepared, len(calls))

    def run(self, graph: object) -> None:
        """Execute a graph belonging to this worker without implicit transfers."""
        with self._lock:
            self._require_open()
            if not isinstance(graph, A1TaskGraph) or graph._worker is not self:
                raise ValueError("graph must belong to this A1 L2 worker")
            graph._prepared()
