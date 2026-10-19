# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private resident buffers for the standalone A1 YOLO diagnostic."""

import ctypes
import math
from typing import Any

import numpy as np
import torch


class Buffers:
    """Own one diagnostic context and its explicit FP16/FP32 allocations."""

    def __init__(self, bridge: Any, device: int):
        self.bridge = bridge
        self.acl = ctypes.CDLL("libascendcl.so")
        signatures = {
            "aclInit": [ctypes.c_void_p],
            "aclrtCreateContext": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
            "aclrtDestroyContext": [ctypes.c_void_p],
            "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_int],
            "aclrtFree": [ctypes.c_void_p],
            "aclrtMemcpy": [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int],
        }
        for name, arguments in signatures.items():
            function = getattr(self.acl, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        status = self.acl.aclInit(None)
        if status not in (0, 100002):
            self.check(status)
        self.context = ctypes.c_void_p()
        self.check(self.acl.aclrtCreateContext(ctypes.byref(self.context), device))
        self.allocations: list[int] = []
        self._images: dict[str, int] = {}

    @staticmethod
    def check(status: int) -> None:
        if status:
            raise RuntimeError(f"A1 diagnostic ACL operation failed: {status}")

    def allocate(self, size: int) -> int:
        address = ctypes.c_void_p()
        self.check(self.acl.aclrtMalloc(ctypes.byref(address), size, 2))
        assert address.value is not None
        self.allocations.append(address.value)
        return address.value

    def upload_array(self, value: np.ndarray, address: int | None = None) -> int:
        value = np.ascontiguousarray(value)
        destination = self.allocate(value.nbytes) if address is None else address
        self.check(self.acl.aclrtMemcpy(destination, value.nbytes, value.ctypes.data, value.nbytes, 1))
        return destination

    def download(self, address: int, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        result = np.empty(shape, dtype=dtype)
        self.check(self.acl.aclrtMemcpy(result.ctypes.data, result.nbytes, address, result.nbytes, 2))
        return result

    def empty(self, shape: tuple[int, ...], dtype: torch.dtype) -> "DeviceTensor":
        return DeviceTensor(self, shape, dtype)

    def upload(self, value: torch.Tensor) -> "DeviceTensor":
        result = self.empty(tuple(value.shape), value.dtype)
        result.copy_from(value.contiguous())
        return result

    def register_graph(self, _calls: Any) -> None:
        return None

    def close(self) -> None:
        for address in reversed(self.allocations):
            self.check(self.acl.aclrtFree(address))
        self.allocations.clear()
        if self.context.value is not None:
            self.check(self.acl.aclrtDestroyContext(self.context))
            self.context = ctypes.c_void_p()


class DeviceTensor:
    """Private contiguous view; addresses remain valid until Buffers.close()."""

    def __init__(
        self, owner: Buffers, shape: tuple[int, ...], dtype: torch.dtype, address: int | None = None
    ):
        if dtype not in (torch.float16, torch.float32):
            raise ValueError("diagnostic buffers require FP16 or FP32")
        self.owner, self.shape, self.dtype = owner, tuple(shape), dtype
        self.numpy_dtype = np.float16 if dtype == torch.float16 else np.float32
        self.bytes = math.prod(shape) * np.dtype(self.numpy_dtype).itemsize
        self.address = owner.allocate(self.bytes) if address is None else address

    def copy_from(self, value: torch.Tensor) -> None:
        if tuple(value.shape) != self.shape or value.dtype != self.dtype or not value.is_contiguous():
            raise ValueError("diagnostic input shape, dtype and contiguous storage must match")
        self.owner.upload_array(value.numpy(), self.address)

    def view(self, shape: tuple[int, ...], offset: int = 0) -> "DeviceTensor":
        if offset < 0 or math.prod(shape) + offset > math.prod(self.shape):
            raise ValueError("diagnostic view exceeds resident allocation")
        return DeviceTensor(
            self.owner, shape, self.dtype, self.address + offset * np.dtype(self.numpy_dtype).itemsize
        )

    def to_cpu(self) -> torch.Tensor:
        return torch.from_numpy(self.owner.download(self.address, self.shape, self.numpy_dtype))

    def descriptor(self) -> int:
        shape = (ctypes.c_uint32 * len(self.shape))(*self.shape)
        data = ctypes.create_string_buffer(128)
        self.owner.bridge.Descriptor(
            self.address, self.bytes, shape, len(self.shape), int(self.dtype == torch.float16), data
        )
        return self.owner.upload_array(np.frombuffer(data.raw, dtype=np.uint8))
