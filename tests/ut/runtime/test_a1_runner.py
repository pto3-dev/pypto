# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""310P compiler selection, independent capabilities, and runtime boundaries."""

import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pypto import ir
from pypto.runtime import RunConfig, a1_runner
from pypto.runtime.runner import DfxOptions, _execute_compiled


@pytest.fixture
def artifact(tmp_path):
    kernel = tmp_path / "kernels" / "aiv" / "add.cpp"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("void kernel_entry(void* args);\n")
    params = [
        {"name": name, "direction": "In", "shape": [16, 128], "dtype": "fp32"}
        for name in ("lhs", "rhs", "out")
    ]
    (tmp_path / "compiled_meta.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "supported_execution_modes": ["program"],
                "platform": "a1",
                "backend_type": "Ascend310P",
                "params": params,
                "num_return_types": 0,
            }
        )
    )
    return tmp_path


@pytest.mark.parametrize(
    ("dtype_name", "dtype"),
    [("fp32", torch.float32), ("fp16", torch.float16), ("int32", torch.int32), ("uint32", torch.uint32)],
)
def test_single_launch_uses_a1_and_preserves_argument_identity(monkeypatch, artifact, dtype_name, dtype):
    metadata = json.loads((artifact / "compiled_meta.json").read_text())
    for param in metadata["params"]:
        param["dtype"] = dtype_name
    (artifact / "compiled_meta.json").write_text(json.dumps(metadata))
    calls = {}

    class Compiler:
        def __init__(self, platform):
            calls["platform"] = platform

        def compile_single_launch(self, source, isa_root, **_kwargs):
            from pathlib import Path  # noqa: PLC0415

            calls["source"] = Path(source).read_text()
            calls["isa_root"] = isa_root
            return b"ELF object"

    class Runtime:
        def __init__(self, image, device_id):
            calls.update(image=image, device=device_id)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls["closed"] = True

        def __call__(self, *args):
            calls["args"] = args

    monkeypatch.setattr(a1_runner, "_load_tools", lambda: (Compiler, Runtime))
    monkeypatch.setattr(a1_runner, "ensure_pto_isa_root", lambda: "pinned-isa")
    args: list[Any] = [torch.empty((16, 128), dtype=dtype) for _ in range(3)]
    a1_runner.execute_single_launch(artifact, args, 3)
    assert calls["platform"] == "a1"
    assert calls["isa_root"] == "pinned-isa"
    assert calls["device"] == 3
    assert all(actual is expected for actual, expected in zip(calls["args"], args, strict=True))
    assert "kernel_entry(args)" in calls["source"]
    assert calls["closed"]


@pytest.mark.parametrize(
    "bad",
    [torch.empty((16, 64)), torch.empty((16, 128), dtype=torch.bfloat16), torch.empty((128, 16)).t(), 1],
)
def test_invalid_carriers_do_not_load_native_runtime(monkeypatch, artifact, bad):
    def unexpected_load():
        pytest.fail("invalid carrier must fail before loading simpler")

    monkeypatch.setattr(a1_runner, "_load_tools", unexpected_load)
    with pytest.raises((ValueError, TypeError)):
        a1_runner.execute_single_launch(artifact, [bad, torch.empty((16, 128)), torch.empty((16, 128))], 0)


def test_graph_and_multiple_kernels_are_rejected(artifact):
    marker = artifact / "kernel_config.py"
    marker.touch()
    args: list[Any] = [torch.empty((16, 128)) for _ in range(3)]
    with pytest.raises(ValueError, match="graph execution"):
        a1_runner.execute_single_launch(artifact, args, 0)
    marker.unlink()
    (artifact / "kernels" / "aiv" / "other.cpp").touch()
    with pytest.raises(ValueError, match="exactly one InCore"):
        a1_runner.execute_single_launch(artifact, args, 0)


def test_public_dispatch_selects_a1_without_legacy_runner(monkeypatch, artifact):
    seen = []
    monkeypatch.setattr(a1_runner, "execute_single_launch", lambda *args: seen.append(args))
    args: list[Any] = [torch.empty((16, 128)) for _ in range(3)]
    _execute_compiled(artifact, args, platform="a1", device_id=2)
    assert seen == [(artifact, args, 2)]
    with pytest.raises(ValueError, match="ring configuration"):
        _execute_compiled(
            artifact, args, platform="a1", device_id=2, config=RunConfig(platform="a1", ring_heap=1024)
        )
    with pytest.raises(ValueError, match="graph or DFX"):
        _execute_compiled(artifact, args, platform="a1", device_id=2, dfx=DfxOptions(enable_pmu=True))


def test_reloaded_public_callable_keeps_a1_dispatch(monkeypatch, artifact):
    from pypto.ir import CompiledProgram  # noqa: PLC0415

    seen = []
    monkeypatch.setattr(a1_runner, "execute_single_launch", lambda *args: seen.append(args))
    compiled = CompiledProgram.from_dir(artifact)
    args: list[Any] = [torch.empty((16, 128)) for _ in range(3)]
    assert compiled(*args) is None
    assert seen == [(artifact, args, 0)]
    with pytest.raises(ValueError, match="require platform"):
        compiled(*args, config=RunConfig(platform="a2a3sim"))
    assert len(seen) == 1
    with pytest.raises(ValueError, match="require platform"):
        CompiledProgram.from_dir(artifact, platform="a2a3sim")


def test_external_include_manifest_is_validated(artifact):
    manifest = artifact / "a1_external_kernel.json"
    manifest.write_text(json.dumps({"schema": 1, "include_dirs": ["includes"]}))
    assert a1_runner._external_include_dirs(artifact) == ["includes"]
    manifest.write_text(json.dumps({"schema": 1, "include_dirs": "not-a-list"}))
    with pytest.raises(ValueError, match="manifest"):
        a1_runner._external_include_dirs(artifact)


@pytest.mark.parametrize(
    ("dtype_name", "dtype"),
    [("fp32", torch.float32), ("int32", torch.int32), ("uint32", torch.uint32)],
)
def test_resident_session_caches_images_and_checks_static_signatures(
    monkeypatch, artifact, dtype_name, dtype
):
    metadata = json.loads((artifact / "compiled_meta.json").read_text())
    for param in metadata["params"]:
        param["dtype"] = dtype_name
    (artifact / "compiled_meta.json").write_text(json.dumps(metadata))
    calls = {"compile": 0, "register": 0, "run": 0, "close": 0}

    class Compiler:
        def __init__(self, _platform):
            pass

        def compile_single_launch(self, *_args, **_kwargs):
            calls["compile"] += 1
            return b"native-image"

    class Runtime:
        def __init__(self, *_args, **_kwargs):
            pass

        def register_kernel(self, _image):
            calls["register"] += 1
            return calls["register"]

        def run_device(self, *_args, **_kwargs):
            calls["run"] += 1

        def close(self):
            calls["close"] += 1
            if calls["close"] == 1:
                raise RuntimeError("drain failed")

    monkeypatch.setattr(a1_runner, "_load_tools", lambda: (Compiler, Runtime))
    monkeypatch.setattr(a1_runner, "ensure_pto_isa_root", lambda: "isa")
    compiled: Any = SimpleNamespace(output_dir=artifact)
    session = a1_runner.A1ResidentSession(compiled)
    kernel = session.prepare(compiled)
    session.prepare(compiled)
    assert calls["compile"] == 1 and calls["register"] == 0
    tensor = SimpleNamespace(shape=(16, 128), dtype=dtype)
    kernel(tensor, tensor, tensor)
    assert calls["run"] == 1
    with pytest.raises(ValueError, match="count"):
        kernel(tensor)
    with pytest.raises(ValueError, match="signature"):
        kernel(SimpleNamespace(shape=(128, 16), dtype=dtype), tensor, tensor)
    source = artifact / "kernels/aiv/add.cpp"
    source.write_text(source.read_text() + "// changed source\n")
    session.prepare(compiled)
    assert calls["compile"] == 2 and calls["register"] == 1
    with pytest.raises(RuntimeError, match="drain"):
        session.close()
    assert not session._closed
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        kernel(tensor, tensor, tensor)


def test_resident_sequence_checks_ownership_and_signature_before_binding():
    captured = []
    session = a1_runner.A1ResidentSession.__new__(a1_runner.A1ResidentSession)
    session._lock = threading.RLock()
    session._closed = False
    session._runtime = SimpleNamespace(prepare_sequence=lambda calls: captured.append(calls))
    kernel = a1_runner.A1ResidentKernel(session, 3, [{"shape": [16], "dtype": "fp16"}])
    tensor = SimpleNamespace(shape=(16,), dtype=torch.float16)
    with pytest.raises(ValueError, match="signature"):
        session.prepare_sequence([(kernel, (SimpleNamespace(shape=(32,), dtype=torch.float16),), 8)])
    foreign = a1_runner.A1ResidentKernel(
        a1_runner.A1ResidentSession.__new__(a1_runner.A1ResidentSession),
        3,
        [{"shape": [16], "dtype": "fp16"}],
    )
    with pytest.raises(ValueError, match="belong"):
        session.prepare_sequence([(foreign, (tensor,), 8)])
    assert not captured
    session.prepare_sequence([(kernel, (tensor,), 8)])
    assert captured == [[(3, 8, (tensor,))]]
    session._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        session.prepare_sequence([(kernel, (tensor,), 8)])


def test_a1_l2_graph_uses_model_backend_and_checks_ownership():
    captured = []
    runs = []
    worker = a1_runner.A1L2Worker.__new__(a1_runner.A1L2Worker)
    worker._lock = threading.RLock()
    worker._closed = False
    worker._runtime = SimpleNamespace(
        prepare_graph=lambda calls: captured.append(calls) or (lambda: runs.append(1))
    )
    params = [{"shape": [32], "dtype": "fp16"}]
    kernel = a1_runner.A1ResidentKernel(worker, 3, params)
    tensor = SimpleNamespace(shape=(32,), dtype=torch.float16)
    for count in [2, 32, 216]:
        graph = worker.register_graph([(kernel, (tensor,), 8)] * count)
        assert graph.task_count == count
        worker.run(graph)
        graph()
        assert captured[-1] == [(3, 8, (tensor,))] * count
    assert worker.level == 2 and len(runs) == 6
    foreign = a1_runner.A1L2Worker.__new__(a1_runner.A1L2Worker)
    foreign._lock = threading.RLock()
    foreign._closed = False
    with pytest.raises(ValueError, match="belong"):
        foreign.run(graph)
    with pytest.raises(ValueError, match="signature"):
        worker.register_graph([(kernel, (SimpleNamespace(shape=(16,), dtype=torch.float16),), 8)])
    assert len(captured) == 3
    worker._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        graph()
    with pytest.raises(RuntimeError, match="closed"):
        worker.register_graph([(kernel, (tensor,), 8)])


def test_a1_graph_rebuild_preserves_identity_without_executing_tasks():
    seen = []
    worker = a1_runner.A1L2Worker.__new__(a1_runner.A1L2Worker)
    worker._lock = threading.RLock()
    worker._closed = False
    prepared = SimpleNamespace(rebuild=lambda: seen.append("rebuild"))
    graph = a1_runner.A1TaskGraph(worker, prepared, 216)
    graph.rebuild()
    assert seen == ["rebuild"]
    assert graph.task_count == 216 and graph._prepared is prepared
    worker._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        graph.rebuild()
    assert seen == ["rebuild"]


def test_independent_binding_virtualizes_grid_and_caches_by_count(monkeypatch, artifact):
    sources = []
    registered = []

    class Compiler:
        def __init__(self, platform):
            assert platform == "a1"

        def compile_single_launch(self, source, _isa_root, **_kwargs):
            from pathlib import Path  # noqa: PLC0415

            sources.append(Path(source).read_text())
            return b"image"

    class Runtime:
        def __init__(self, *_args, **_kwargs):
            pass

        def register_kernel(self, image):
            registered.append(image)
            return len(registered)

        def close(self):
            pass

    monkeypatch.setattr(a1_runner, "_load_tools", lambda: (Compiler, Runtime))
    monkeypatch.setattr(a1_runner, "ensure_pto_isa_root", lambda: "isa")
    compiled: Any = SimpleNamespace(output_dir=artifact)
    with a1_runner.A1L2Worker(compiled) as worker:
        original = worker.prepare_independent(compiled, 1)
        two = worker.prepare_independent(compiled, 2)
        cached = worker.prepare_independent(compiled, 2)
        three = worker.prepare_independent(compiled, 3)
        assert len(sources) == 3 and len(registered) == 2
        assert original._kernel_id == 0 and two._kernel_id == cached._kernel_id
        assert two._kernel_id != three._kernel_id
        assert len(two._params) == 6 and len(three._params) == 9
        assert "#define get_block_idx() 0" not in sources[0]
        assert "#define get_block_idx() 0" in sources[1]
        assert "#define get_block_num() 1" in sources[1]
        assert sources[1].index("#undef get_block_idx") < sources[1].index("PyptoA1TensorEntry")
        assert "kernel_entry(args + get_block_idx() * 3)" in sources[1]
        assert "always_inline" in sources[1]
    with pytest.raises(RuntimeError, match="closed"):
        worker.prepare_independent(compiled, 2)


@pytest.mark.parametrize("instances", [0, -1, 9, True, 2.0, "2"])
def test_independent_binding_rejects_invalid_count_before_loading(instances):
    worker = a1_runner.A1L2Worker.__new__(a1_runner.A1L2Worker)
    compiled: Any = None  # Invalid counts must fail before inspecting the artifact.
    with pytest.raises(ValueError, match="one to eight"):
        worker.prepare_independent(compiled, instances)


def test_independent_binding_checks_blocks_and_argument_budget(artifact):
    worker = a1_runner.A1L2Worker.__new__(a1_runner.A1L2Worker)
    worker._lock = threading.RLock()
    worker._closed = False
    worker._runtime = SimpleNamespace(
        prepare_graph=lambda _calls: pytest.fail("invalid bindings must not reach native graph setup")
    )
    params = [{"shape": [32], "dtype": "fp16"}] * 8
    kernel = a1_runner.A1ResidentKernel(worker, 7, params, required_blocks=8)
    tensors = (SimpleNamespace(shape=(32,), dtype=torch.float16),) * 8
    for blocks in (1, 7, 9):
        with pytest.raises(ValueError, match="requires blocks=8"):
            worker.register_graph([(kernel, tensors, blocks)])
        with pytest.raises(ValueError, match="requires blocks=8"):
            kernel(*tensors, blocks=blocks)
    with pytest.raises(ValueError, match="argument count"):
        kernel(*tensors[:-1], blocks=8)
    metadata = json.loads((artifact / "compiled_meta.json").read_text())
    metadata["params"] *= 3  # Nine tensors x eight instances exceeds the runtime's 64 slots.
    (artifact / "compiled_meta.json").write_text(json.dumps(metadata))
    compiled: Any = SimpleNamespace(output_dir=artifact)
    with pytest.raises(ValueError, match="64 flattened"):
        worker.prepare_independent(compiled, 8)


@pytest.mark.parametrize("dtype_name", [None, ["int32"], {"dtype": "int32"}, "int64", "bf16"])
def test_unsupported_storage_metadata_rejected_before_native_load(monkeypatch, artifact, dtype_name):
    metadata = json.loads((artifact / "compiled_meta.json").read_text())
    metadata["params"][0]["dtype"] = dtype_name
    (artifact / "compiled_meta.json").write_text(json.dumps(metadata))

    def unexpected_load():
        pytest.fail("unsupported storage metadata must not load the native runtime")

    monkeypatch.setattr(a1_runner, "_load_tools", unexpected_load)
    compiled: Any = SimpleNamespace(output_dir=artifact)
    with pytest.raises(ValueError, match="resident kernel signature"):
        a1_runner._resident_artifact(compiled)


@pytest.mark.parametrize("dtype_name,actual", [("int32", torch.uint32), ("uint32", torch.int32)])
def test_integer_signedness_mismatch_rejected_before_native_load(monkeypatch, artifact, dtype_name, actual):
    metadata = json.loads((artifact / "compiled_meta.json").read_text())
    for param in metadata["params"]:
        param["dtype"] = dtype_name
    (artifact / "compiled_meta.json").write_text(json.dumps(metadata))

    def unexpected_load():
        pytest.fail("mismatched integer carriers must not load the native runtime")

    monkeypatch.setattr(a1_runner, "_load_tools", unexpected_load)
    with pytest.raises(ValueError, match="static A1 kernel signature"):
        a1_runner.execute_single_launch(artifact, [torch.empty((16, 128), dtype=actual)] * 3, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("blocks", [0, 9, True, "8"])
def test_invalid_static_spmd_manifest_is_rejected_before_runtime(artifact, blocks):
    (artifact / "a1_spmd.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "task_count": 1,
                "blocks": blocks,
                "kernel": "kernels/aiv/add.cpp",
                "argument_count": 3,
            }
        )
    )
    with pytest.raises(ValueError, match="SPMD manifest"):
        a1_runner._resident_artifact(ir.CompiledProgram.from_dir(artifact))


@pytest.mark.parametrize("order", [[0, 0, 2], [2, 1], [False, 1, 2], [-1, 1, 2], [3, 1, 2]])
def test_static_spmd_argument_permutation_must_be_bijective(artifact, order):
    (artifact / "a1_spmd.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "task_count": 1,
                "blocks": 8,
                "kernel": "kernels/aiv/add.cpp",
                "argument_count": 3,
                "argument_order": order,
            }
        )
    )
    with pytest.raises(ValueError, match="kernel signature"):
        a1_runner._resident_artifact(ir.CompiledProgram.from_dir(artifact))
