# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collect 310P native chip observations for a strictly checked YOLO26n replay.

This is an instrumented sequential diagnostic, separate from the default
device-model backend. Its envelopes include observer and synchronization costs.
"""

import argparse
import ctypes
import hashlib
import importlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pypto import ir
from pypto.backend import BackendType

from .chip_profile_buffers import Buffers
from .executor import NativeYOLO26, RejectHostOperators
from .precision import prepare_fp16_model
from .profile import _hardware, _validate


class Flag(ctypes.Structure):
    _fields_ = [("value", ctypes.c_uint64), ("padding", ctypes.c_uint64 * 15)]


class Header(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("tasks", ctypes.c_uint32),
        ("cores", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("frequency", ctypes.c_uint64),
        ("status", ctypes.c_uint64),
        ("timeout_us", ctypes.c_uint64),
        ("padding", ctypes.c_uint64 * 11),
    ]


class Record(ctypes.Structure):
    _fields_ = [
        ("dispatch_begin", ctypes.c_uint64),
        ("dispatch_end", ctypes.c_uint64),
        ("released", ctypes.c_uint64),
        ("received", ctypes.c_uint64 * 8),
        ("finished", ctypes.c_uint64 * 8),
    ]


def _wire_type(tasks: int) -> Any:
    class Wire(ctypes.Structure):
        _fields_ = [
            ("header", Header),
            ("commands", Flag * tasks),
            ("ready", Flag * tasks),
            ("states", Flag * 8),
            ("hello", Flag * 8),
            ("records", Record * tasks),
        ]

    return Wire


def _compile_support(output: Path) -> Any:
    env_manager = importlib.import_module("simpler").env_manager
    PROJECT_ROOT = importlib.import_module("simpler_setup.environment").PROJECT_ROOT
    GxxToolchain = importlib.import_module("simpler_setup.toolchain").GxxToolchain

    if platform.machine() != "aarch64":
        raise RuntimeError("the standalone AICPU diagnostic requires an aarch64 host")
    env_manager.ensure("ASCEND_HOME_PATH")
    toolchain = GxxToolchain()
    if toolchain.ascend_home_path is None:
        raise RuntimeError("source the CANN environment before collecting")
    sdk = Path(toolchain.ascend_home_path)
    common = PROJECT_ROOT / "src/common"
    include = PROJECT_ROOT / "src/a1/platform/include"
    flags = [toolchain.cxx_path, "-std=c++17", "-shared", "-fPIC", "-O2", "-Wl,--build-id"]
    includes = [
        common,
        common / "worker",
        common / "task_interface",
        common / "log/include",
        include,
        sdk / "include",
        sdk / "pkg_inc",
        sdk / "pkg_inc/runtime",
        sdk / "pkg_inc/runtime/runtime",
    ]
    sources = [
        Path(__file__).with_name("chip_profile_host.cpp"),
        common / "aicpu_loader/host/load_aicpu_op.cpp",
        common / "log/host_log.cpp",
        common / "log/unified_log_host.cpp",
        common / "task_interface/assert_compat.cpp",
    ]
    commands = [
        [
            *flags,
            *(f"-I{path}" for path in includes),
            *(str(path) for path in sources),
            f"-L{sdk / 'lib64'}",
            f"-Wl,-rpath,{sdk / 'lib64'}",
            "-lascendcl",
            "-lruntime",
            "-ldl",
            "-pthread",
            "-o",
            str(output / "host.so"),
        ],
        [
            *flags,
            f"-I{include}",
            str(PROJECT_ROOT / "src/a1/platform/onboard/aicpu/chip_profile.cpp"),
            "-o",
            str(output / "observer.so"),
        ],
        [
            *flags,
            f"-I{common}",
            str(common / "aicpu_loader/device/aicpu_dispatcher.cpp"),
            "-o",
            str(output / "dispatcher.so"),
        ],
    ]
    for command in commands:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"diagnostic support compilation failed:\n{error.stderr}") from error
    bridge = ctypes.CDLL(str(output / "host.so"))
    bridge.ProbeError.restype = ctypes.c_char_p
    bridge.ProbeCreate.argtypes = [ctypes.c_char_p] * 3
    bridge.ProbeCreate.restype = ctypes.c_void_p
    bridge.ProbeRun.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    bridge.ProbeRun.restype = ctypes.c_int
    bridge.ProbeClose.argtypes = [ctypes.c_void_p]
    bridge.Descriptor.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    return bridge


def _combined_source(sources: list[Path]) -> str:
    chunks = [
        "#include <cstdint>\n#include <pto/pto-inst.hpp>\n"
        '#include "tensor.h"\n#include "chip_profile_core.h"\n'
    ]
    # Native convolutions still share these model-local helpers. Include them
    # once outside per-entry namespaces; generated public kernels need none.
    for name in ("conv_bias.hpp", "conv_input.hpp", "fp16_activation.hpp"):
        chunks.append(f'#include "{Path(__file__).with_name(name)}"\n')
    for index, source in enumerate(sources):
        body = "\n".join(line for line in source.read_text().splitlines() if not line.startswith("#include"))
        declarations = (
            'extern "C" AICORE void kernel_entry(',
            'extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(',
        )
        if sum(body.count(declaration) for declaration in declarations) != 1:
            raise ValueError(f"expected one native or generated operator entry: {source.name}")
        for declaration in declarations:
            body = body.replace(declaration, f"AICORE void op_{index}(")
        chunks.append(f"namespace op{index} {{\n{body}\n}}\n")
    cases = "\n".join(f"case {index}: op{index}::op_{index}(args); break;" for index in range(len(sources)))
    entry = (
        Path(__file__).with_name("chip_profile_entry.cpp.in").read_text().replace("// GENERATED_CASES", cases)
    )
    return "\n".join([*chunks, entry])


def _validate_observer_trace(task_count: int, blocks: set[int]) -> None:
    # The wire and AICPU observer already encode a runtime task count. Their
    # two-state-per-task counters bound it to UINT32_MAX/2, not one historical
    # model's task total. Retain the qualified one/eight-core dispatch profile.
    if not 0 < task_count <= 0x7FFFFFFF or not blocks or not blocks <= {1, 8}:
        raise ValueError("the diagnostic requires a bounded nonempty graph with one/eight-core tasks")


class DiagnosticYOLO26(NativeYOLO26):
    """Use the existing graph builder with native or generated operator bodies."""

    def __init__(self, model: Any, output: Path, bridge: Any, device: int):
        self.bridge = bridge
        self.handle = None
        self.capture: dict[str, Any] = {}
        super().__init__(
            model,
            128,
            output,
            device,
            four_way_accumulation=True,
            linear_query_key=True,
            precise_small_softmax=True,
        )

    def __enter__(self) -> "DiagnosticYOLO26":
        return self

    def _prepare(self, jit: Any, key: str | None = None) -> Any:
        if key is None:
            raise ValueError("diagnostic kernels require an explicit cache key")
        if self.session is None:
            self.session = Buffers(self.bridge, self.device_id)
        if key not in self._prepared:
            sources = jit._external_source_paths()
            if sources:
                if len(sources) != 1:
                    raise ValueError("the diagnostic requires one operator body per task")
                source = Path(sources[0])
                if source.parent != Path(__file__).parent:
                    raise ValueError("the diagnostic accepts only the local external operator bodies")
            else:
                # Compile public PyPTO callees through the same backend as the
                # default executor. Profiling must not retain an obsolete C++
                # implementation after the model's public migration.
                directory = self.output_dir / f"kernel_{self._compiled_count:04d}"
                self._compiled_count += 1
                compiled = ir.compile(
                    jit.specialize(),
                    backend_type=BackendType.Ascend310P,
                    output_dir=str(directory),
                    dump_passes=False,
                )
                generated = sorted((Path(compiled.output_dir) / "kernels").rglob("*.cpp"))
                if len(generated) != 1:
                    raise ValueError("the diagnostic requires one generated kernel per task")
                source = generated[0]
            self._prepared[key] = source
            self.session._images[str(key)] = 1
        return self._prepared[key]

    def prepare_observer(self) -> None:
        from pypto.runtime.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

        KernelCompiler = importlib.import_module("simpler_setup.kernel_compiler").KernelCompiler

        _validate_observer_trace(len(self.trace), {task.blocks for task in self.trace})
        sources = sorted(set(task.kernel for task in self.trace))
        source = self.output_dir / "combined.cpp"
        source.write_text(_combined_source(sources))
        image = KernelCompiler("a1").compile_single_launch(str(source), ensure_pto_isa_root())
        image_path = self.output_dir / "combined.o"
        image_path.write_bytes(image)
        calls = [
            [
                sources.index(task.kernel),
                task.blocks,
                self.session.upload_array(
                    np.array([tensor.descriptor() for tensor in task.arguments], dtype=np.uint64)
                ),
            ]
            for task in self.trace
        ]
        self.calls = self.session.upload_array(np.array(calls, dtype=np.uint64))
        self.wire_type = _wire_type(len(self.trace))
        self.wire_address = self.session.allocate(ctypes.sizeof(self.wire_type))
        self.handle = self.bridge.ProbeCreate(
            str(self.output_dir.parent / "dispatcher.so").encode(),
            str(self.output_dir.parent / "observer.so").encode(),
            str(image_path).encode(),
        )
        if not self.handle:
            raise RuntimeError(self.bridge.ProbeError().decode())
        self.image_sha256 = hashlib.sha256(image).hexdigest()
        self.sources = {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources}

    def replay(self) -> None:
        start = time.perf_counter()
        wire = self.wire_type()
        wire.header.version, wire.header.tasks, wire.header.cores, wire.header.timeout_us = (
            1,
            len(self.trace),
            8,
            500000,
        )
        self.session.upload_array(np.frombuffer(wire, dtype=np.uint8), self.wire_address)
        guard = RejectHostOperators()
        with guard:
            status = self.bridge.ProbeRun(self.handle, self.wire_address, self.calls)
        elapsed = (time.perf_counter() - start) * 1000
        if status:
            raise RuntimeError(self.bridge.ProbeError().decode())
        wire = self.wire_type.from_buffer_copy(
            self.session.download(self.wire_address, (ctypes.sizeof(wire),), np.uint8)
        )
        self.capture = {
            "schema_version": 1,
            "timestamp_kind": "aicpu_observed_envelopes",
            "status": wire.header.status,
            "frequency": wire.header.frequency,
            "host_ms_including_wire_upload_and_observation": elapsed,
            "host_aten_operations_during_replay": guard.calls,
            "raw_core_ids": [line.value - 1 for line in wire.hello],
            "labels": [task.label for task in self.trace],
            "task_blocks": [task.blocks for task in self.trace],
            "tasks": [
                {
                    name: list(getattr(record, name))
                    if name in ("received", "finished")
                    else getattr(record, name)
                    for name, _ in Record._fields_
                }
                for record in wire.records
            ],
            "operator_sources": self.sources,
            "aicore_image_sha256": self.image_sha256,
        }
        if wire.header.status != 1:
            raise RuntimeError(f"native observation did not complete: {wire.header.status}")

    def close(self) -> None:
        if self.handle:
            self.bridge.ProbeClose(self.handle)
            self.handle = None
        super().close()


def _arrays(native: DiagnosticYOLO26) -> dict[str, np.ndarray]:
    values = {
        "detections": native.output.to_cpu().reshape(1, 300, 6).numpy(),
        "dense": native.dense.to_cpu().reshape(336, 96).numpy(),
    }
    values.update({f"feature{index}": tensor.to_cpu().numpy() for index, tensor in native.features.items()})
    values.update({f"box{index}": tensor.to_cpu().numpy() for index, tensor in enumerate(native.raw_boxes)})
    values.update({f"cls{index}": tensor.to_cpu().numpy() for index, tensor in enumerate(native.raw_scores)})
    return values


def _require_reference_array(value: np.ndarray, expected: np.ndarray, name: str) -> None:
    if (
        value.shape != expected.shape
        or value.dtype != expected.dtype
        or value.tobytes() != expected.tobytes()
    ):
        raise AssertionError(f"diagnostic output differs from strict reference: {name}")


def main() -> None:
    convert = importlib.import_module("simpler_setup.tools.a1_chip_swimlane").convert
    YOLO = importlib.import_module("ultralytics").YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "input", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data = np.load(args.input)
    if data.shape != (1, 3, 128, 128) or data.dtype != np.float32:
        raise ValueError("input must be a batch-one FP32 NCHW 128x128 array")
    bridge = _compile_support(args.output)
    model = YOLO(str(args.weights)).model.eval()
    model.model[-1].end2end = True
    model = prepare_fp16_model(model)
    with DiagnosticYOLO26(model, args.output / "compiled", bridge, args.device) as native:
        native.prepare_observer()
        hardware = _hardware(args.device)
        root = Path(__file__).resolve().parents[3]
        provenance = {
            "pypto_commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "runtime_commit": subprocess.check_output(
                ["git", "-C", str(root / "runtime"), "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        native.input.copy_from(torch.from_numpy(data).reshape(-1))
        native.replay()
        for iteration in range(args.iterations):
            native.replay()
            capture = {
                **native.capture,
                **provenance,
                "hardware": hardware,
                "input_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            }
            (args.output / f"capture{iteration}.json").write_text(json.dumps(capture, indent=2) + "\n")
            trace, report = convert(capture)
            (args.output / f"swimlane{iteration}.json").write_text(json.dumps(trace) + "\n")
            (args.output / f"critical{iteration}.json").write_text(json.dumps(report, indent=2) + "\n")
            values = _arrays(native)
            if len(values) != 31:
                raise AssertionError("incomplete full-model readback")
            folder = args.output / f"output{iteration}"
            folder.mkdir()
            for name, value in values.items():
                np.save(folder / f"{name}.npy", value)
                expected = np.load(args.reference / f"{name}.npy")
                _require_reference_array(value, expected, name)
            print(
                f"Round {iteration}: {report['observed_span_ms']:.6f} ms observed, 31 arrays byte-equal",
                flush=True,
            )
        validation = {
            "reference_arrays_per_round": 31,
            "tolerance_changes": False,
            "strict_current_input": _validate(model, native, torch.from_numpy(data)),
        }
        for seed in (0, 1, 2, 17, 317, 2026):
            value = torch.rand((1, 3, 128, 128), generator=torch.Generator().manual_seed(seed))
            validation[f"seed{seed}"] = _validate(model, native, value)
        (args.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    main()
