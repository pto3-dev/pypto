# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Versioned A1 cache-policy artifact contract, independent of device tools."""

import json
from pathlib import Path, PurePosixPath
from typing import cast

A1_CACHE_POLICY_FILE = "a1_cache_policy.json"
A1_CACHE_ARGUMENT_ABI = 2


def a1_cache_policy_files(kernels: dict[str, int]) -> dict[str, str]:
    """Describe only entries that require the runtime-owned alias offset."""
    if not kernels:
        return {}
    return {
        A1_CACHE_POLICY_FILE: json.dumps(
            {"schema": 1, "argument_abi_version": A1_CACHE_ARGUMENT_ABI, "kernels": kernels}, indent=2
        )
    }


def a1_cache_argument_count(work_dir: Path, kernel: Path) -> int | None:
    """Return a bypass entry's tensor arity after validating its sidecar."""
    path = work_dir / A1_CACHE_POLICY_FILE
    if not path.exists():
        return None
    decoded: object = json.loads(path.read_text())
    if not isinstance(decoded, dict):
        raise ValueError("invalid A1 cache-policy argument contract")
    # JSON object keys are strings; values stay untrusted until checked below.
    record = cast(dict[str, object], decoded)
    raw_kernels = record.get("kernels")
    if (
        type(record.get("schema")) is not int
        or record["schema"] != 1
        or type(record.get("argument_abi_version")) is not int
        or record["argument_abi_version"] != A1_CACHE_ARGUMENT_ABI
        or not isinstance(raw_kernels, dict)
        or not raw_kernels
    ):
        raise ValueError("invalid A1 cache-policy argument contract")
    kernels = cast(dict[str, object], raw_kernels)
    validated: dict[str, int] = {}
    for name, count in kernels.items():
        parts = PurePosixPath(name).parts
        if (
            PurePosixPath(name).as_posix() != name
            or len(parts) != 3
            or parts[0] != "kernels"
            or parts[1] not in ("aic", "aiv")
            or not parts[2].endswith(".cpp")
            or ".." in parts
            or type(count) is not int
            or not 1 <= count <= 64
            or not (work_dir / name).is_file()
        ):
            raise ValueError("invalid A1 cache-policy kernel signature")
        validated[name] = count
    relative = kernel.relative_to(work_dir).as_posix()
    count = validated.get(relative)
    if count is not None:
        metadata: object = json.loads((work_dir / "compiled_meta.json").read_text())
        if not isinstance(metadata, dict):
            raise ValueError("A1 cache-policy arity does not match the tensor signature")
        params = cast(dict[str, object], metadata).get("params")
        if not isinstance(params, list) or len(cast(list[object], params)) != count:
            raise ValueError("A1 cache-policy arity does not match the tensor signature")
    return count
