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
from typing import Any

import pytest
import torch
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
                "schema": 1,
                "platform": "a1",
                "backend_type": "Ascend310P",
                "params": params,
                "num_return_types": 0,
            }
        )
    )
    return tmp_path


def test_single_launch_uses_a1_and_preserves_argument_identity(monkeypatch, artifact):
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
    args: list[Any] = [torch.empty((16, 128)) for _ in range(3)]
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
