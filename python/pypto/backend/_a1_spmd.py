# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Describe the fail-closed single-grid subset of A1 SPMD orchestration."""

import json
from typing import Any

from pypto.pypto_core import ir


def _single_call(body: ir.Stmt) -> ir.Call | None:
    calls: list[ir.Call] = []
    valid = True

    def visit(stmt: ir.Stmt) -> None:
        nonlocal valid
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                visit(child)
        elif isinstance(stmt, ir.RuntimeScopeStmt):
            visit(stmt.body)
        elif isinstance(stmt, ir.EvalStmt):
            # A side-effect-only grid call has no assignment/result value.
            # Other evaluated expressions still cannot be collapsed away.
            if isinstance(stmt.expr, ir.Call) and isinstance(stmt.expr.op, ir.GlobalVar):
                calls.append(stmt.expr)
            else:
                valid = False
        elif isinstance(stmt, ir.AssignStmt):
            if isinstance(stmt.value, ir.Call) and isinstance(stmt.value.op, ir.GlobalVar):
                calls.append(stmt.value)
            elif isinstance(stmt.value, ir.TupleGetItemExpr) and isinstance(stmt.value.tuple, ir.Var):
                # NormalizeReturnOrder materializes pure tuple projections for
                # multiple outputs; they do not add another device dispatch.
                pass
            elif not isinstance(stmt.value, ir.Var):
                valid = False
        elif not isinstance(stmt, ir.ReturnStmt):
            # A host loop, predicate, allocation or Submit must never silently
            # collapse to one unconditional device launch.
            valid = False

    visit(body)
    return calls[0] if valid and len(calls) == 1 else None


def single_spmd_files(program: ir.Program, orchestration: ir.Function | None) -> dict[str, str]:
    """Persist one static grid with an unchanged tensor argument ABI.

    General dynamic orchestration is deliberately not advertised by this
    bridge. Several such grids can be composed through A1L2Worker graphs.
    """
    if orchestration is None:
        return {}
    call = _single_call(orchestration.body)
    if call is None:
        return {}
    wrapper = program.get_function(call.op.name)
    blocks = call.attrs.get("core_num")
    if (
        wrapper is None
        or wrapper.func_type != ir.FunctionType.Spmd
        or not isinstance(blocks, ir.ConstInt)
        or not 1 <= blocks.value <= 8
    ):
        return {}
    inner = _single_call(wrapper.body)
    if inner is None:
        return {}
    kernel = program.get_function(inner.op.name)
    if kernel is None or kernel.func_type not in (ir.FunctionType.AIC, ir.FunctionType.AIV):
        return {}
    order = list(range(len(orchestration.params)))
    for caller, invocation in ((orchestration, call), (wrapper, inner)):
        positions = {param.unique_id: index for index, param in enumerate(caller.params)}
        if len(invocation.args) != len(caller.params) or any(
            not isinstance(arg, ir.Var) or arg.unique_id not in positions for arg in invocation.args
        ):
            return {}
        permutation = [positions[arg.unique_id] for arg in invocation.args if isinstance(arg, ir.Var)]
        if sorted(permutation) != list(range(len(caller.params))):
            return {}
        order = [order[index] for index in permutation]
    if len(kernel.params) != len(orchestration.params) or any(
        not isinstance(param.type, ir.TensorType) for param in orchestration.params
    ):
        return {}
    core = "aiv" if kernel.func_type == ir.FunctionType.AIV else "aic"
    record: dict[str, Any] = {
        "schema": 1,
        "blocks": blocks.value,
        "kernel": f"kernels/{core}/{kernel.name}.cpp",
        "argument_count": len(kernel.params),
        "argument_order": order,
        "task_count": 1,
    }
    return {"a1_spmd.json": json.dumps(record, indent=2)}
