# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A1 cache ABI selection and footer forwarding before any device operation."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pypto._a1_cache_policy import A1_CACHE_POLICY_FILE, a1_cache_argument_count, a1_cache_policy_files
from pypto.runtime import a1_runner


@pytest.fixture
def artifact(tmp_path):
    kernel = tmp_path / "kernels/aiv/copy.cpp"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("void kernel_entry(void* args, unsigned long offset);\n")
    params = [
        {"name": name, "direction": direction, "shape": [2, 16], "dtype": "fp32"}
        for name, direction in (("x", "In"), ("out", "Out"))
    ]
    (tmp_path / "compiled_meta.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "supported_execution_modes": ["program"],
                "platform": "a1",
                "backend_type": "Ascend310P",
                "params": params,
                "num_return_types": 1,
            }
        )
    )
    for name, value in a1_cache_policy_files({"kernels/aiv/copy.cpp": 2}).items():
        (tmp_path / name).write_text(value)
    return tmp_path


@pytest.fixture
def tools(monkeypatch):
    events = []

    class Compiler:
        def __init__(self, platform):
            assert platform == "a1"

        def compile_single_launch(self, source, _isa, **_kwargs):
            events.append(("compile", Path(source).read_text()))
            return b"image"

    class Runtime:
        max_argument_abi_version = 2
        l2_cache_info = (1 << 40, 0)

        def __init__(self, _image, device_id, argument_abi_version=1):
            events.append(("create", device_id, argument_abi_version))

        def register_kernel(self, _image):
            events.append(("register",))
            return 1

        def prepare_graph(self, calls):
            events.append(("graph", calls))
            return lambda: events.append(("run",))

        def close(self):
            events.append(("close",))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def __call__(self, *args):
            events.append(("call", args))

    monkeypatch.setattr(a1_runner, "_load_tools", lambda: (Compiler, Runtime))
    monkeypatch.setattr(a1_runner, "ensure_pto_isa_root", lambda: "isa")
    return events, Compiler, Runtime


@pytest.mark.parametrize("instances,footer", [(1, 2), (2, 4), (8, 16)])
def test_footer_follows_the_entire_call_not_each_image(artifact, tools, instances, footer):
    events, _, _ = tools
    kernel = artifact / "kernels/aiv/copy.cpp"
    a1_runner._compile_image(artifact, kernel, "isa", independent_instances=instances, argument_count=2)
    source = events[-1][1]
    assert f"static_cast<uint64_t>(args[{footer}])" in source
    call = "kernel_entry(args" if instances == 1 else "kernel_entry(args + get_block_idx() * 2"
    assert call + f", static_cast<uint64_t>(args[{footer}]))" in source
    if instances > 1:
        assert "void kernel_entry(__gm__ int64_t*, uint64_t);" in source


@pytest.mark.parametrize("count", [0, 1, 3])
def test_independent_bad_arity_rejected_before_tools(monkeypatch, artifact, count):
    monkeypatch.setattr(a1_runner, "_load_tools", lambda: pytest.fail("invalid arity must not load tools"))
    with pytest.raises(ValueError, match="independent signature"):
        a1_runner._compile_image(
            artifact, artifact / "kernels/aiv/copy.cpp", "isa", independent_instances=8, argument_count=count
        )


def test_standalone_selects_abi2_without_changing_tensor_identity(artifact, tools):
    events, _, _ = tools
    args = [torch.empty((2, 16)) for _ in range(2)]
    a1_runner.execute_single_launch(artifact, args, 3)
    assert ("create", 3, 2) in events
    actual = next(event[1] for event in events if event[0] == "call")
    assert all(x is y for x, y in zip(actual, args, strict=True))


def test_old_runtime_rejected_before_native_compile_or_creation(monkeypatch, artifact, tools):
    events, compiler, _ = tools
    monkeypatch.setattr(a1_runner, "_load_tools", lambda: (compiler, SimpleNamespace()))
    with pytest.raises(RuntimeError, match="runtime argument ABI 2"):
        a1_runner.execute_single_launch(artifact, [torch.empty((2, 16)) for _ in range(2)], 0)
    assert not events


def test_resident_abi2_accepts_initial_and_independent_bypass_entries(artifact, tools):
    events, _, _ = tools
    compiled: Any = SimpleNamespace(output_dir=artifact)
    with a1_runner.A1L2Worker(compiled) as worker:
        assert worker.l2_cache_info == (1 << 40, 0)
        kernel = worker.prepare_independent(compiled, 8)
        assert len(kernel._params) == 16
        assert worker.prepare_independent(compiled, 8)._kernel_id == kernel._kernel_id
        tensors = (SimpleNamespace(shape=(2, 16), dtype=torch.float32),) * 16
        graph = worker.register_graph([(kernel, tensors, 8)])
        assert graph.task_count == 1
        graph()
    assert sum(event[0] == "create" for event in events) == 1
    assert sum(event[0] == "register" for event in events) == 1
    assert events[-2:] == [("run",), ("close",)]


def test_abi1_rejects_later_bypass_before_registration(artifact, tools):
    events, _, _ = tools
    sidecar = artifact / A1_CACHE_POLICY_FILE
    saved = sidecar.read_text()
    sidecar.unlink()
    source = artifact / "kernels/aiv/copy.cpp"
    bypass_source = source.read_text()
    source.write_text("void kernel_entry(void* args);\n")
    compiled: Any = SimpleNamespace(output_dir=artifact)
    with a1_runner.A1L2Worker(compiled) as worker:
        assert ("create", 0, 1) in events
        sidecar.write_text(saved)
        source.write_text(bypass_source)
        before = list(events)
        with pytest.raises(ValueError, match="session created with argument_abi_version=2"):
            worker.prepare(compiled)
        with pytest.raises(ValueError, match="session created with argument_abi_version=2"):
            worker.prepare_independent(compiled, 8)
        assert events == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("schema", 2),
        ("argument_abi_version", True),
        ("argument_abi_version", 1),
        ("kernels", []),
        ("kernels", {}),
    ],
)
def test_invalid_contract_fails_before_loading_tools(monkeypatch, artifact, field, value):
    path = artifact / A1_CACHE_POLICY_FILE
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    monkeypatch.setattr(a1_runner, "_load_tools", lambda: pytest.fail("invalid artifact must not load tools"))
    compiled: Any = SimpleNamespace(output_dir=artifact)
    with pytest.raises(ValueError, match="argument contract"):
        a1_runner.A1ResidentSession(compiled)


@pytest.mark.parametrize(
    "name,count",
    [
        ("../copy.cpp", 2),
        ("kernels/aiv/copy.cpp", True),
        ("kernels/aiv/copy.cpp", 0),
        ("kernels/aiv/copy.cpp", 65),
        ("kernels/aiv/copy.cpp", 3),
        ("kernels/aiv/missing.cpp", 2),
    ],
)
def test_bad_arity_and_paths_rejected(artifact, name, count):
    for filename, text in a1_cache_policy_files({name: count}).items():
        (artifact / filename).write_text(text)
    with pytest.raises(ValueError, match="signature"):
        a1_cache_argument_count(artifact, artifact / "kernels/aiv/copy.cpp")


def test_chip_observer_rejects_untransported_offset_before_combining(artifact):
    from examples.models.yolo26_native.chip_profile import _combined_source  # noqa: PLC0415

    with pytest.raises(ValueError, match="chip observer does not yet support"):
        _combined_source([artifact / "kernels/aiv/copy.cpp"])


@pytest.mark.parametrize("metadata", [None, [], {"params": True}, {"params": []}])
def test_cache_contract_rejects_invalid_compiled_signature(artifact, metadata):
    # The sidecar cannot authorize a footer using malformed or mismatched
    # compiled metadata, even when its own schema and kernel path are valid.
    (artifact / "compiled_meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="arity does not match the tensor signature"):
        a1_cache_argument_count(artifact, artifact / "kernels/aiv/copy.cpp")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
