# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared hardware gating and model-plan loading for focused YOLO26 tests."""

import ctypes
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def require_310p() -> None:
    """Skip when native coupled-core Ascend310P hardware is unavailable."""
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    name = acl.aclrtGetSocName()
    if not name or not name.startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")


def load_plan(name: str) -> Any:
    """Load the user-facing model operator from the current checkout."""
    path = Path(__file__).resolve().parents[3] / f"examples/models/yolo26_native/{name}.py"
    spec = importlib.util.spec_from_file_location(f"yolo26_native_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
