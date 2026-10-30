# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public 310P QK model ABI and opt-in selection."""

from typing import Any, cast

import pytest
from pypto import ir
from pypto.backend import BackendType

from examples.models.yolo26_native.matrix import MatrixPlan


def test_public_qk_model_codegen(tmp_path):
    plan = MatrixPlan(400, 400, 32, True, experimental_public_qk=True)
    assert plan.metadata().tolist()[:4] == [400, 400, 32, 1]
    ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
    )
    code = "\n".join(path.read_text() for path in tmp_path.rglob("*.pto"))
    assert "pto.tmatmul" in code and "pto.tmatmul.acc" in code
    assert "pto.get_block_idx" in code and "pto.get_block_num" in code
    before_column, inside_column = code.split("scf.for %column", 1)
    assert before_column.count("pto.tload ") == 2
    assert inside_column.count("pto.tload ") == 2
    assert code.count("!pto.ptr<f16>") >= 3
    assert "!pto.ptr<f32>" in code


@pytest.mark.parametrize(
    "args",
    [(400, 64, 400, False, False), (16, 16, 32, True, False), (400, 400, 32, True, True)],
)
def test_public_qk_rejects_other_products(args):
    with pytest.raises(ValueError, match="requires 400x400x32"):
        MatrixPlan(*args, experimental_public_qk=True)


def test_public_qk_flag_requires_boolean():
    with pytest.raises(TypeError, match="must be a boolean"):
        MatrixPlan(400, 400, 32, True, experimental_public_qk=cast(Any, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
