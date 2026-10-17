# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Placement attributes survive direct calls for both pure engine kinds."""

import pypto.language as pl
from _orchestration_codegen_common import _generate_orch_code
from pypto import backend
from pypto.backend import BackendType


def test_static_mapping_is_opt_in_and_preserves_dynamic_tasks():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def fixed_aic(
            self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pl.func_attr({"static_block_mapping": True})
            return out

        @pl.function(type=pl.FunctionType.AIC)
        def dynamic_aic(
            self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def fixed_aiv(
            self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pl.func_attr({"static_block_mapping": True})
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def dynamic_aiv(
            self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            a: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            b: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a = self.fixed_aic(x, a)
            b = self.dynamic_aic(a, b)
            a = self.fixed_aiv(b, a)
            b = self.dynamic_aiv(a, b)
            return b

    code = _generate_orch_code(Program)
    assert code.count("launch_spec.set_static_block_mapping(true)") == 2
    for task in (0, 2):
        assert f"params_t{task}.launch_spec.set_static_block_mapping(true);" in code
    for task in (1, 3):
        assert f"params_t{task}.launch_spec.set_static_block_mapping" not in code
    assert code.index("params_t0.launch_spec.set_static_block_mapping(true)") < code.index(
        "rt_submit_aic_task(0"
    )
    assert code.index("params_t2.launch_spec.set_static_block_mapping(true)") < code.index(
        "rt_submit_aiv_task(2"
    )
