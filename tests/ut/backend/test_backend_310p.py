# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""310P target identification and fail-fast selection without A3 fallback."""

import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


@pytest.fixture(autouse=True)
def restore_backend_selection():
    """Preserve the caller's global backend selection across rejection tests."""
    previous = backend.get_backend_type() if backend.is_backend_configured() else None
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()
    if previous is not None:
        backend.set_backend_type(previous)


def test_310p_is_a_distinct_target():
    assert BackendType.__members__["Ascend310P"] == BackendType.Ascend310P
    assert BackendType.Ascend310P != BackendType.Ascend910B
    assert BackendType.Ascend310P != BackendType.Ascend950
    assert BackendType.Ascend910B.value == 0
    assert BackendType.Ascend950.value == 1


def test_310p_does_not_return_an_a3_backend():
    with pytest.raises(ValueError, match="Ascend310P.*experimental.*A1 lowering"):
        backend.get_backend_instance(BackendType.Ascend310P)


def test_rejected_selection_does_not_poison_configuration():
    with pytest.raises(ValueError, match="Ascend310P.*experimental"):
        backend.set_backend_type(BackendType.Ascend310P)
    assert not backend.is_backend_configured()
    backend.set_backend_type(BackendType.Ascend910B)
    assert backend.get_backend_instance(backend.get_backend_type()).get_type_name() == "910B"


def test_rejected_selection_preserves_an_existing_backend():
    backend.set_backend_type(BackendType.Ascend950)
    with pytest.raises(ValueError, match="cannot change to Ascend310P"):
        backend.set_backend_type(BackendType.Ascend310P)
    assert backend.get_backend_type() == BackendType.Ascend950


@pytest.mark.parametrize("execution_mode", [ExecutionMode.ONBOARD, ExecutionMode.SIM])
def test_runtime_rejects_310p_before_platform_dispatch(execution_mode):
    with pytest.raises(ValueError, match="Ascend310P.*experimental.*A1 lowering"):
        RunConfig(arch=BackendType.Ascend310P, execution_mode=execution_mode)


@pytest.mark.parametrize("platform", [None, "a2a3", "a2a3sim", "a5", "a5sim"])
def test_compile_rejects_310p_before_creating_artifacts(tmp_path, platform):
    output_dir = tmp_path / "must_not_exist"
    program = ir.Program([], "unsupported_310p", ir.Span.unknown())
    with pytest.raises(ValueError, match="Ascend310P.*experimental.*A1 lowering"):
        ir.compile(
            program,
            backend_type=BackendType.Ascend310P,
            platform=platform,
            output_dir=str(output_dir),
            skip_ptoas=True,
        )
    assert not output_dir.exists()
    assert not backend.is_backend_configured()


@pytest.mark.parametrize("platform", ["a2a3", "a2a3sim", "a5", "a5sim"])
@pytest.mark.parametrize("target_keyword", ["arch", "backend_type"])
def test_legacy_runtime_platform_cannot_hide_310p(platform, target_keyword):
    with pytest.raises(ValueError, match="Ascend310P.*experimental.*A1 lowering"):
        RunConfig(platform=platform, **{target_keyword: BackendType.Ascend310P})


@pytest.mark.parametrize(
    ("backend_type", "available"),
    [(BackendType.Ascend910B, True), (BackendType.Ascend950, True), (BackendType.Ascend310P, False)],
)
def test_availability_query_does_not_configure_a_backend(backend_type, available):
    assert backend.is_backend_available(backend_type) is available
    assert not backend.is_backend_configured()
