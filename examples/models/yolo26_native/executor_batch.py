# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Production-shape YOLO26 with one independent image per coupled 310P core."""

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .executor import Launch, NativeYOLO26


class _ConstantUploads:
    """Share immutable setup tensors while keeping activation allocations distinct.

    This setup-only proxy never changes the worker's upload method. Device
    ownership, graph preparation and execution remain with the actual worker.
    NativeYOLO26 only uploads metadata/weights/zero constants while recording;
    its input and intermediate activation tensors use separate empty storage.
    """

    def __init__(self, worker: Any):
        self.worker = worker
        self.constants: dict[tuple[str, tuple[int, ...], bytes], Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.worker, name)

    def upload(self, tensor: torch.Tensor) -> Any:
        key = (
            str(tensor.dtype),
            tuple(tensor.shape),
            hashlib.sha256(tensor.contiguous().numpy().tobytes()).digest(),
        )
        if key not in self.constants:
            self.constants[key] = self.worker.upload(tensor)
        return self.constants[key]


class _SharedImage(NativeYOLO26):
    def __init__(
        self,
        *args: Any,
        shared: "_SharedImage | None" = None,
        experimental_public_add: bool = False,
        **kwargs: Any,
    ):
        self._public_add = experimental_public_add
        self._shared_image = shared
        super().__init__(*args, **kwargs)

    def _prepare(self, jit: Any, key: str | None = None) -> Any:
        if self._shared_image is not None:
            self.session = self._shared_image.session
            self._prepared = self._shared_image._prepared
            self._artifacts = self._shared_image._artifacts
        kernel = super()._prepare(jit, key)
        if not isinstance(self.session, _ConstantUploads):
            self.session = _ConstantUploads(self.session)
        return kernel


class NativeYOLO26Batch:
    """Record a 640x640 batch-eight graph on one physical Ascend310P chip.

    Each layer binds eight independent signatures to eight coupled cores.
    Convolution profile selection, tensor layouts and exact TopK semantics
    follow NativeYOLO26. Constant weights and compiled
    images are shared; each image owns separate inputs and intermediate data.
    Replay submits one static device model without host tensor arithmetic.
    Other shapes/batches remain with NativeYOLO26 until separately qualified.
    """

    def __init__(
        self,
        model: Any,
        size: int,
        output_dir: Path,
        device_id: int = 0,
        *,
        batch_size: int = 8,
        weight_l2_bypass: bool = False,
        spmd_attention_pack: bool = False,
        experimental_public_add: bool = False,
    ):
        if type(size) is not int or type(batch_size) is not int or (size, batch_size) != (640, 8):
            raise ValueError("native batch YOLO26 requires size=640 and batch_size=8")
        if type(spmd_attention_pack) is not bool:
            raise TypeError("SPMD attention packing must be a boolean")
        if type(experimental_public_add) is not bool:
            raise TypeError("experimental public ADD must be a boolean")
        self.spmd_attention_pack = spmd_attention_pack
        self.size, self.batch_size = size, batch_size
        self.images: list[_SharedImage] = []
        self.trace: list[Launch] = []
        self.session: Any = None
        try:
            for image in range(batch_size):
                native = _SharedImage(
                    model,
                    size,
                    output_dir / f"image-{image}",
                    device_id,
                    shared=self.images[0] if self.images else None,
                    four_way_accumulation=True,
                    linear_query_key=False,
                    precise_small_softmax=True,
                    weight_l2_bypass=weight_l2_bypass,
                    experimental_public_add=experimental_public_add,
                )
                self.images.append(native)
                self.session = native.session.worker
            prepared = {}
            # Layer-major ordering retains every per-image dependency. A graph
            # task finishes all eight independent images before its consumer.
            # Only the core-to-signature mapping changes; each kernel sees the
            # same logical single-core grid used by the independent regressions.
            for siblings in zip(*(image.trace for image in self.images), strict=True):
                first = siblings[0]
                if any(t.label != first.label or t.kernel._params != first.kernel._params for t in siblings):
                    raise ValueError("batch image traces must have matching operation signatures")
                if first.kernel not in prepared:
                    artifact = self.images[0]._artifacts[first.kernel]
                    prepared[first.kernel] = self.session.prepare_independent(artifact, batch_size)
                arguments = tuple(argument for task in siblings for argument in task.arguments)
                self.trace.append(Launch(prepared[first.kernel], arguments, batch_size, first.label))
            if spmd_attention_pack:
                from .spmd_qkv import pack_batch_attention  # noqa: PLC0415

                self.trace = pack_batch_attention(self)
            self._sequence = self.session.register_graph(
                [(task.kernel, task.arguments, task.blocks) for task in self.trace]
            )
        except Exception:
            self.close()
            raise

    def replay(self) -> None:
        """Execute all eight resident images with one synchronous model call."""
        self._sequence()

    def rebuild_graph(self) -> None:
        """Refresh profiling state outside measured replay windows."""
        self._sequence.rebuild()

    def copy_input(self, tensor: torch.Tensor) -> None:
        """Explicitly upload FP32 NCHW input outside device replay timing."""
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != (self.batch_size, 3, self.size, self.size)
        ):
            raise ValueError("native batch input must be CPU FP32 [8,3,640,640]")
        for image, native in enumerate(self.images):
            native.input.copy_from(tensor[image].contiguous().reshape(-1))

    def detections_cpu(self) -> torch.Tensor:
        """Explicitly download final detections, outside device replay timing."""
        return torch.stack([image.output.to_cpu().reshape(300, 6) for image in self.images])

    def coverage(self) -> dict[str, Any]:
        """Distinguish independent images, physical cores, tasks and submissions."""
        return {
            "batch_size": self.batch_size,
            "spmd_attention_pack": self.spmd_attention_pack,
            "physical_chip_count": 1,
            "coupled_cores_per_task": self.batch_size,
            "device_compute_tasks": len(self.trace),
            "worker_level": 2,
            "model_execute_calls_per_replay": 1,
            "runtime_kernel_launches_per_replay": 0,
            "operations": dict(Counter(task.label for task in self.trace)),
            "shared_constants": len(self.images[0].session.constants),
        }

    def close(self) -> None:
        """Drain the shared worker once; image views do not own separate devices."""
        if self.session is not None:
            self.session.close()

    def __enter__(self) -> "NativeYOLO26Batch":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
