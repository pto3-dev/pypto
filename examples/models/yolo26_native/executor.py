# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static YOLO26-nano execution: all activation arithmetic stays on 310P.

CPU work is limited to loading/fusing/packing weights, creating static geometry,
recording host dispatch and explicit transfers. This is synchronous replay,
not hardware graph capture. Unsupported modules fail rather than falling back.
"""

import ctypes
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1L2Worker
from torch.utils._python_dispatch import TorchDispatchMode

from .depthwise import DepthwisePlan
from .detection import DecodePlan, TopKPlan
from .layout import ConcatPlan, VectorOp, VectorPlan
from .matrix import MatrixPlan
from .operators import ConvPlan
from .softmax import Softmax16Plan


class RejectHostOperators(TorchDispatchMode):
    """Reject any ATen arithmetic during the native launch trace."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple = (), kwargs: dict[str, Any] | None = None
    ) -> Any:
        self.calls += 1
        raise RuntimeError(f"native YOLO26 replay attempted a host ATen operation: {func}")


@dataclass(frozen=True)
class Feature:
    """NHWC feature map, optionally a channel slice of a wider resident row."""

    data: Any
    height: int
    width: int
    channels: int
    row_stride: int
    offset: int = 0

    def channel_slice(self, start: int, channels: int) -> "Feature":
        """Create a metadata-only view; native routing materializes when needed."""
        if start < 0 or channels <= 0 or start + channels > self.channels:
            raise ValueError("feature channel slice exceeds its logical channels")
        return Feature(self.data, self.height, self.width, channels, self.row_stride, self.offset + start)

    def to_cpu(self) -> torch.Tensor:
        """Explicit debug/readback, never used by the execution trace."""
        value = self.data.to_cpu().reshape(self.height, self.width, self.row_stride)
        return (
            value[:, :, self.offset : self.offset + self.channels].permute(2, 0, 1).unsqueeze(0).contiguous()
        )


@dataclass(frozen=True)
class Launch:
    """One prepared native kernel invocation with fixed resident arguments."""

    kernel: Any
    arguments: tuple[Any, ...]
    blocks: int
    label: str

    def run(self) -> None:
        self.kernel(*self.arguments, blocks=self.blocks)


def _softmax_kernel(cols: int) -> Any:
    @pl.jit.incore
    def softmax(x: pl.Tensor[[16, cols], pl.FP32], output: pl.Out[pl.Tensor[[16, cols], pl.FP32]]):
        a = pl.load(x, [0, 0], [16, cols])
        max_scratch = pl.tile.create([16, cols], dtype=pl.FP32)
        maximum = pl.tile.row_max(a, max_scratch)
        shifted = pl.sub(a, pl.tile.row_expand(a, maximum))
        exponent = pl.exp(shifted)
        sum_scratch = pl.tile.create([16, cols], dtype=pl.FP32)
        total = pl.tile.row_sum(exponent, sum_scratch)
        value = pl.div(exponent, pl.tile.row_expand(exponent, total))
        pl.store(value, [0, 0], output)
        return output

    return softmax


def _runtime_l2_cache_offset(device_id: int) -> int:
    """Query the initialized session's visible runtime device, never a physical ordinal."""
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    acl.aclrtGetDevice.restype = ctypes.c_int
    current = ctypes.c_int32()
    if acl.aclrtGetDevice(ctypes.byref(current)) or current.value != device_id:
        raise RuntimeError("weight L2 bypass requires the native session's current runtime device")
    runtime = ctypes.CDLL("libruntime.so")
    runtime.rtGetL2CacheOffset.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
    runtime.rtGetL2CacheOffset.restype = ctypes.c_int
    value = ctypes.c_uint64()
    status = runtime.rtGetL2CacheOffset(device_id, ctypes.byref(value))
    if status or not value.value:
        raise RuntimeError(
            f"weight L2 bypass offset unavailable on runtime device {device_id}: status {status}"
        )
    return value.value


class NativeYOLO26:
    """Record and replay a fixed batch-one 128/640 YOLO26-nano graph."""

    def __init__(
        self,
        model: Any,
        size: int,
        output_dir: Path,
        device_id: int = 0,
        *,
        four_way_accumulation: bool = False,
        linear_query_key: bool = False,
        precise_small_softmax: bool = False,
        weight_l2_bypass: bool = False,
    ):
        if size not in (128, 640):
            raise ValueError("native YOLO26 accepts batch-one input sizes 128 and 640")
        head = model.model[-1]
        if not head.end2end or head.reg_max != 1 or head.nc != 80 or head.max_det != 300 or head.agnostic_nms:
            raise ValueError("native YOLO26 requires the 80-class NMS-free one-to-one head with max_det=300")
        if type(weight_l2_bypass) is not bool:
            raise TypeError("weight L2 bypass must be a boolean")
        if weight_l2_bypass and not four_way_accumulation:
            raise ValueError("weight L2 bypass requires four-way convolution")
        self.weight_l2_bypass = weight_l2_bypass
        self.weight_l2_cache_offset = 0
        self.four_way_accumulation = four_way_accumulation
        self.linear_query_key = linear_query_key
        self.precise_small_softmax = precise_small_softmax
        self.size = size
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        self.session: Any = None
        self.trace: list[Launch] = []
        self.features: dict[int, Feature] = {}
        self._compiled_count = 0
        self._prepared: dict[str, Any] = {}
        self._zero: Any = None
        try:
            self._build(model)
            self._sequence = self.session.register_graph(
                [(launch.kernel, launch.arguments, launch.blocks) for launch in self.trace]
            )
        except Exception:
            self.close()
            raise

    def _prepare(self, jit: Any, key: str | None = None) -> Any:
        if key is not None and key in self._prepared:
            return self._prepared[key]
        directory = self.output_dir / f"kernel_{self._compiled_count:04d}"
        self._compiled_count += 1
        compiled = ir.compile(
            jit.specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(directory),
            dump_passes=False,
        )
        if not isinstance(compiled, ir.CompiledProgram):
            raise ValueError("native YOLO26 requires a single-chip compiled artifact")
        if self.session is None:
            self.session = A1L2Worker(compiled, self.device_id)
        kernel = self.session.prepare(compiled)
        if key is not None:
            self._prepared[key] = kernel
        return kernel

    def _emit(self, plan: Any, inputs: tuple[Any, ...], output: Any, label: str, blocks: int = 8) -> None:
        kernel = self._prepare(plan.jit_kernel(), repr(plan))
        metadata = self.session.upload(plan.metadata())
        self.trace.append(Launch(kernel, (metadata, *inputs, output), blocks, label))

    def _empty(self, count: int, dtype: torch.dtype = torch.float16) -> Any:
        return self.session.empty((count,), dtype)

    def _dummy(self) -> Any:
        if self._zero is None:
            self._zero = self.session.upload(torch.zeros(16, dtype=torch.float16))
        return self._zero

    def _vector(self, plan: VectorPlan, x: Any, b: Any = None, out: Any = None) -> Any:
        if out is None:
            out = self._empty(
                plan.output_size, torch.float32 if plan.operation == VectorOp.HALF_TO_FLOAT else torch.float16
            )
        self._emit(plan, (x, self._dummy() if b is None else b), out, f"vector:{plan.operation.name}")
        return out

    def _copy(self, x: Feature, out: Feature) -> None:
        if (x.height, x.width, x.channels) != (out.height, out.width, out.channels):
            raise ValueError("native channel routing requires equal logical geometry")
        plan = VectorPlan(
            VectorOp.COPY, x.height, x.width, x.channels, x.row_stride, out.row_stride, x.offset, out.offset
        )
        self._vector(plan, x.data, out=out.data)

    def _compact(self, x: Feature) -> Feature:
        stride = (x.channels + 15) // 16 * 16
        if x.row_stride == stride and x.offset == 0:
            return x
        out = Feature(self._empty(x.height * x.width * stride), x.height, x.width, x.channels, stride)
        self._copy(x, out)
        return out

    def _concat(self, values: list[Feature]) -> Feature:
        height, width = values[0].height, values[0].width
        channels = sum(v.channels for v in values)
        if channels % 16:
            raise ValueError("native concatenation requires aligned logical channels")
        if any((v.height, v.width) != (height, width) for v in values):
            raise ValueError("native concatenation requires matching spatial geometry")
        result = Feature(self._empty(height * width * channels), height, width, channels, channels)
        if (
            2 <= len(values) <= 4
            and channels <= 4096
            and height * width <= 2**24
            and all((v.row_stride - v.channels) // 16 <= 65535 for v in values)
        ):
            plan = ConcatPlan(height, width, tuple((v.channels, v.row_stride, v.offset) for v in values))
            inputs = tuple(v.data for v in values) + (self._dummy(),) * (4 - len(values))
            self._emit(plan, inputs, result.data, "concat")
            return result
        offset = 0
        for value in values:
            self._copy(value, result.channel_slice(offset, value.channels))
            offset += value.channels
        return result

    def _add(self, a: Feature, b: Feature) -> Feature:
        if (a.height, a.width, a.channels) != (b.height, b.width, b.channels):
            raise ValueError("native residual addition requires identical logical geometry")
        result = Feature(
            self._empty(a.height * a.width * a.channels), a.height, a.width, a.channels, a.channels
        )
        plan = VectorPlan(
            VectorOp.ADD,
            a.height,
            a.width,
            a.channels,
            a.row_stride,
            result.row_stride,
            a.offset,
            0,
            second_stride=b.row_stride,
            second_offset=b.offset,
        )
        self._vector(plan, a.data, b.data, result.data)
        return result

    def _convolution(self, module: Any, x: Feature, silu: bool) -> Feature:
        if module.in_channels != x.channels or module.dilation != (1, 1) or module.padding_mode != "zeros":
            raise ValueError("unsupported native convolution input/dilation/padding mode")
        if (
            module.kernel_size[0] != module.kernel_size[1]
            or module.stride[0] != module.stride[1]
            or module.padding[0] != module.padding[1]
        ):
            raise ValueError("native convolution requires square symmetric geometry")
        bias = (
            module.bias.detach().float().contiguous()
            if module.bias is not None
            else torch.zeros(module.out_channels)
        )
        if module.groups == 1:
            plan = ConvPlan(
                x.height,
                x.width,
                x.channels,
                module.out_channels,
                module.kernel_size[0],
                module.stride[0],
                module.padding[0],
                silu,
                four_way_accumulation=self.four_way_accumulation,
                source_stride=x.row_stride,
                source_offset=x.offset,
            )
            packed_bias = torch.zeros(plan.co, dtype=torch.float32)
            packed_bias[: module.out_channels] = bias
            weight = self.session.upload(plan.pack_weight(module.weight.detach()))
            dbias = self.session.upload(packed_bias)
            output = self._empty(plan.output_size)
            kernel = self._prepare(plan.jit_kernel(), repr(plan))
            if self.weight_l2_bypass and not self.weight_l2_cache_offset:
                self.weight_l2_cache_offset = _runtime_l2_cache_offset(self.device_id)
            meta = self.session.upload(plan.metadata(weight_l2_offset=self.weight_l2_cache_offset))
            self.trace.append(
                Launch(kernel, (meta, x.data, weight, dbias, output, self._dummy()), 8, "convolution")
            )
            return Feature(output, plan.oh, plan.ow, module.out_channels, plan.co)
        if (
            module.groups != x.channels
            or module.out_channels != x.channels
            or module.kernel_size != (3, 3)
            or module.padding != (1, 1)
        ):
            raise ValueError("native grouped convolution requires 3x3 depthwise with padding one")
        x = self._compact(x)
        plan = DepthwisePlan(
            x.height,
            x.width,
            x.channels,
            module.stride[0],
            silu,
            four_way_accumulation=self.four_way_accumulation,
        )
        weight = self.session.upload(plan.pack_weight(module.weight.detach()))
        dbias = self.session.upload(bias)
        output = self._empty(plan.output_size)
        self._emit(plan, (x.data, weight, dbias), output, "depthwise")
        return Feature(output, plan.oh, plan.ow, x.channels, x.channels)

    def _pool(self, x: Feature) -> Feature:
        result = Feature(
            self._empty(x.height * x.width * x.channels), x.height, x.width, x.channels, x.channels
        )
        self._vector(
            VectorPlan(VectorOp.MAXPOOL, x.height, x.width, x.channels, x.row_stride, x.channels, x.offset),
            x.data,
            out=result.data,
        )
        return result

    def _upsample(self, x: Feature) -> Feature:
        result = Feature(
            self._empty(x.height * x.width * 4 * x.channels),
            x.height * 2,
            x.width * 2,
            x.channels,
            x.channels,
        )
        plan = VectorPlan(
            VectorOp.UPSAMPLE,
            result.height,
            result.width,
            x.channels,
            x.row_stride,
            x.channels,
            x.offset,
            input_height=x.height,
            input_width=x.width,
        )
        self._vector(plan, x.data, out=result.data)
        return result

    def _matmul(self, a: Any, b: Any, m: int, n: int, k: int, transpose_b: bool = False) -> Any:
        plan = MatrixPlan(m, n, k, transpose_b, linear_accumulation=self.linear_query_key and transpose_b)
        result = self._empty(m * n)
        self._emit(plan, (a, b), result, "matrix")
        return result

    def _softmax(self, scores: Any, n: int) -> Any:
        if self.precise_small_softmax and n == 16:
            result = self._empty(n * n)
            self._emit(Softmax16Plan(), (scores,), result, "softmax:pair")
            return result
        floats = self._vector(VectorPlan(VectorOp.HALF_TO_FLOAT, n * n), scores)
        result = self._empty(n * n, torch.float32)
        kernel = self._prepare(_softmax_kernel(n), f"softmax:{n}")
        for row in range(0, n, 16):
            self.trace.append(
                Launch(kernel, (floats.view((16, n), row * n), result.view((16, n), row * n)), 1, "softmax")
            )
        return self._vector(VectorPlan(VectorOp.FLOAT_TO_HALF, n * n), result)

    def _attention(self, module: Any, x: Feature) -> Feature:
        qkv = self._module(module.qkv, x)
        n = x.height * x.width
        if module.num_heads != 2 or module.key_dim != 32 or module.head_dim != 64:
            raise ValueError("native attention currently validates two heads with key/head dimensions 32/64")
        attended = []
        values = []
        for head in range(module.num_heads):
            base = head * (2 * module.key_dim + module.head_dim)
            q = self._compact(qkv.channel_slice(base, module.key_dim))
            k = self._compact(qkv.channel_slice(base + module.key_dim, module.key_dim))
            v = self._compact(qkv.channel_slice(base + 2 * module.key_dim, module.head_dim))
            scaled = self._vector(
                VectorPlan(VectorOp.SCALE_HALF, n * module.key_dim, scale=module.scale), q.data
            )
            scores = self._matmul(scaled, k.data, n, n, module.key_dim, True)
            probabilities = self._softmax(scores, n)
            output = self._matmul(probabilities, v.data, n, module.head_dim, n)
            attended.append(Feature(output, x.height, x.width, module.head_dim, module.head_dim))
            values.append(v)
        encoded = self._module(module.pe, self._concat(values))
        return self._module(module.proj, self._add(self._concat(attended), encoded))

    def _split_block(self, module: Any, x: Feature) -> Feature:
        value = self._module(module.cv1, x)
        pieces = [value.channel_slice(0, module.c), value.channel_slice(module.c, module.c)]
        for child in module.m:
            pieces.append(self._module(child, pieces[-1]))
        return self._module(module.cv2, self._concat(pieces))

    def _c3(self, module: Any, x: Feature) -> Feature:
        a = self._module(module.m, self._module(module.cv1, x))
        return self._module(module.cv3, self._concat([a, self._module(module.cv2, x)]))

    def _bottleneck(self, module: Any, x: Feature) -> Feature:
        result = self._module(module.cv2, self._module(module.cv1, x))
        return self._add(x, result) if module.add else result

    def _c2psa(self, module: Any, x: Feature) -> Feature:
        value = self._module(module.cv1, x)
        a = value.channel_slice(0, module.c)
        b = self._module(module.m, value.channel_slice(module.c, module.c))
        return self._module(module.cv2, self._concat([a, b]))

    def _psa(self, module: Any, x: Feature) -> Feature:
        attention = self._attention(module.attn, x)
        value = self._add(x, attention) if module.add else attention
        result = self._module(module.ffn, value)
        return self._add(value, result) if module.add else result

    def _sppf(self, module: Any, x: Feature) -> Feature:
        if module.m.kernel_size != 5:
            raise ValueError("native SPPF requires 5x5 max pooling")
        parts = [self._module(module.cv1, x)]
        for _ in range(module.n):
            parts.append(self._pool(parts[-1]))
        result = self._module(module.cv2, self._concat(parts))
        return self._add(x, result) if module.add else result

    def _upsample_module(self, module: Any, x: Feature) -> Feature:
        if module.mode != "nearest" or module.scale_factor != 2.0:
            raise ValueError("unsupported native upsampling")
        return self._upsample(x)

    def _concat_module(self, module: Any, x: list[Feature]) -> Feature:
        if module.d != 1:
            raise ValueError("native YOLO26 concatenation requires channel dimension one")
        return self._concat(x)

    def _module(self, module: Any, x: Any) -> Any:
        name = type(module).__name__
        if isinstance(module, torch.nn.Sequential):
            for child in module:
                x = self._module(child, x)
            return x
        if isinstance(module, torch.nn.Conv2d):
            return self._convolution(module, x, False)
        if name in ("Conv", "DWConv"):
            if not isinstance(module.act, (torch.nn.SiLU, torch.nn.Identity)):
                raise ValueError(f"unsupported convolution activation {type(module.act).__name__}")
            return self._convolution(module.conv, x, isinstance(module.act, torch.nn.SiLU))
        handlers = {
            "C3k2": self._split_block,
            "C2f": self._split_block,
            "C3k": self._c3,
            "C3": self._c3,
            "Bottleneck": self._bottleneck,
            "C2PSA": self._c2psa,
            "PSABlock": self._psa,
            "SPPF": self._sppf,
            "Upsample": self._upsample_module,
            "Concat": self._concat_module,
            "Detect": self._detect,
            "Attention": self._attention,
        }
        handler = handlers.get(name)
        if handler is None:
            raise ValueError(f"unsupported native YOLO26 module {name}")
        return handler(module, x)

    def _detect(self, module: Any, features: list[Feature]) -> Any:
        total = sum(f.height * f.width for f in features)
        self.dense = self._empty(total * 96, torch.float32)
        self.raw_boxes = []
        self.raw_scores = []
        offset = 0
        for level, value in enumerate(features):
            boxes = self._module(module.one2one_cv2[level], value)
            scores = self._module(module.one2one_cv3[level], value)
            self.raw_boxes.append(boxes)
            self.raw_scores.append(scores)
            plan = DecodePlan(value.height, value.width, int(module.stride[level]), offset, total)
            self._emit(plan, (boxes.data, scores.data), self.dense, "decode")
            offset += value.height * value.width
        result = self._empty(1800, torch.float32)
        self._emit(TopKPlan(total), (self.dense,), result, "topk", blocks=1)
        return result

    def _build(self, model: Any) -> None:
        pack = VectorPlan(VectorOp.PACK, self.size, self.size, channels=3)
        self._prepare(pack.jit_kernel(), repr(pack))
        self.input = self._empty(3 * self.size * self.size, torch.float32)
        packed = self._vector(pack, self.input)
        x = Feature(packed, self.size, self.size, 3, 16)
        for module in model.model:
            if module.f == -1:
                source = x
            elif isinstance(module.f, int):
                source = self.features[module.f]
            else:
                source = [x if index == -1 else self.features[index] for index in module.f]
            x = self._module(module, source)
            if isinstance(x, Feature):
                self.features[module.i] = x
        self.output = x

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Explicit input transfer, pure native trace replay, explicit readback."""
        if (
            input_tensor.device.type != "cpu"
            or input_tensor.dtype != torch.float32
            or tuple(input_tensor.shape) != (1, 3, self.size, self.size)
        ):
            raise ValueError("native YOLO26 input requires CPU FP32 [1,3,size,size]")
        self.input.copy_from(input_tensor.contiguous().reshape(-1))
        self.replay()
        return self.output.to_cpu().reshape(1, 300, 6)

    def replay(self) -> None:
        """Dispatch only prepared native kernels; no activation host arithmetic."""
        sequence = getattr(self, "_sequence", None)
        if sequence is not None:
            sequence()
        else:
            for launch in self.trace:
                launch.run()

    def rebuild_graph(self) -> None:
        """Refresh the registered graph after profiler start or stop.

        This prepares CANN tasks without executing model arithmetic. Exclude
        rebuilding and subsequent warmup from measured replay windows.
        """
        sequence = getattr(self, "_sequence", None)
        if sequence is None:
            raise RuntimeError("native device graph is not prepared")
        sequence.rebuild()

    def coverage(self) -> dict[str, Any]:
        """Summarize emitted native stages and resident image counts."""
        return {
            "launches": len(self.trace),
            "prepared_stream_sequence": False,
            "prepared_device_graph": True,
            "worker_level": 2,
            "scheduler_backend": "a1_static_device_model",
            "device_compute_tasks": len(self.trace),
            "runtime_kernel_launches_per_replay": 0,
            "model_execute_calls_per_replay": 1,
            "host_submissions_per_replay": 1,
            "operations": dict(Counter(v.label for v in self.trace)),
            "compiled_signatures": self._compiled_count,
            "registered_images": len(self.session._images),
        }

    def close(self) -> None:
        if self.session is not None:
            self.session.close()

    def __enter__(self) -> "NativeYOLO26":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
