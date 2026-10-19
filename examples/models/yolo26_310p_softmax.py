# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hybrid YOLO26 validation: CPU model with native PTO3/310P attention softmax.

This example measures synchronous host calls including allocation and copies.
It does not measure device-only kernel latency or full-PTO3 model performance.
Install Ultralytics separately and supply a local YOLO26-nano checkpoint.
"""

import argparse
import hashlib
import importlib
import json
import statistics
import time
import types
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime.pto_isa import ensure_pto_isa_root


def _softmax_kernel(cols: int) -> Any:
    @pl.jit.incore
    def softmax(
        x: pl.Tensor[[16, cols], pl.FP32],
        output: pl.Tensor[[16, cols + 32], pl.FP32],
    ):
        a = pl.load(x, [0, 0], [16, cols])
        max_scratch = pl.tile.create([16, cols], dtype=pl.FP32)
        maximum = pl.tile.row_max(a, max_scratch)
        shifted = pl.sub(a, pl.tile.row_expand(a, maximum))
        exponential = pl.exp(shifted)
        sum_scratch = pl.tile.create([16, cols], dtype=pl.FP32)
        total = pl.tile.row_sum(exponential, sum_scratch)
        result = pl.div(exponential, pl.tile.row_expand(exponential, total))
        pl.store(result, [0, 16], output)

    return softmax


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("At least one timing sample is required")
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": float(torch.tensor(values, dtype=torch.float64).quantile(0.95)),
        "min_ms": min(values),
    }


class SoftmaxBridge:
    """Keep one compiled image/session and tile real attention score matrices."""

    def __init__(self, cols: int, output_dir: Path, device_id: int, stack: ExitStack):
        if cols not in (16, 400):
            raise ValueError("Validated YOLO26 profiles require attention width 16 or 400")
        self.cols = cols
        self.host_call_ms: list[float] = []
        self.module_calls: dict[str, int] = {}
        self.check_accuracy = True
        self.max_softmax_abs_error = 0.0
        started = time.perf_counter()
        ir.compile(
            _softmax_kernel(cols).specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(output_dir),
            dump_passes=False,
        )
        kernels = list((output_dir / "kernels").glob("*/*.cpp"))
        if len(kernels) != 1:
            raise RuntimeError("Expected one PTO3-generated softmax kernel")
        entry = output_dir / "softmax_entry.cpp"
        entry.write_text(
            kernels[0].read_text()
            + '\nextern "C" __global__ AICORE void YoloSoftmaxEntry(__gm__ int64_t* args) {\n'
            + "    kernel_entry(args);\n}\n"
        )
        tools = importlib.import_module("simpler_setup")
        image = tools.KernelCompiler("a1").compile_single_launch(
            str(entry), ensure_pto_isa_root(), extra_include_dirs=[str(kernels[0].parent)]
        )
        self.runner = stack.enter_context(tools.A1SingleLaunch(image, device_id=device_id))
        self.compile_and_session_ms = (time.perf_counter() - started) * 1000
        self.image_sha256 = hashlib.sha256(image).hexdigest()

    def __call__(self, scores: torch.Tensor, module_name: str) -> torch.Tensor:
        """Run FP32 softmax on each aligned group of sixteen score rows."""
        if scores.device.type != "cpu" or scores.dtype != torch.float32 or scores.shape[-1] != self.cols:
            raise ValueError("Bridge requires CPU FP32 scores with the compiled attention width")
        flat = scores.contiguous().reshape(-1, self.cols)
        if flat.shape[0] % 16:
            raise ValueError("Attention row count must be divisible by sixteen")
        result = torch.empty_like(flat)
        for offset in range(0, flat.shape[0], 16):
            source = flat[offset : offset + 16].clone()
            output = torch.full((16, self.cols + 32), 7.0)
            started = time.perf_counter()
            self.runner(source, output)
            self.host_call_ms.append((time.perf_counter() - started) * 1000)
            if self.check_accuracy:
                assert torch.equal(output[:, :16], torch.full_like(output[:, :16], 7.0))
                assert torch.equal(
                    output[:, self.cols + 16 :], torch.full_like(output[:, self.cols + 16 :], 7.0)
                )
            result[offset : offset + 16] = output[:, 16 : self.cols + 16]
        result = result.reshape(scores.shape)
        self.module_calls[module_name] = self.module_calls.get(module_name, 0) + 1
        if self.check_accuracy:
            expected = torch.softmax(scores, dim=-1)
            torch.testing.assert_close(result, expected, rtol=0.0003, atol=0.000003)
            self.max_softmax_abs_error = max(
                self.max_softmax_abs_error, float((result - expected).abs().max())
            )
        return result


def _attention_forward(bridge: SoftmaxBridge, name: str) -> Any:
    def forward(module: Any, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = module.qkv(x)
        q, k, v = qkv.view(batch, module.num_heads, module.key_dim * 2 + module.head_dim, tokens).split(
            [module.key_dim, module.key_dim, module.head_dim], dim=2
        )
        scores = (q * module.scale).transpose(-2, -1) @ k
        probabilities = bridge(scores, name)
        attended = (v @ probabilities.transpose(-2, -1)).reshape(batch, channels, height, width)
        attended = attended + module.pe(v.reshape(batch, channels, height, width))
        return module.proj(attended)

    return forward


@contextmanager
def _native_softmax(model: Any, attention_type: type, bridge: SoftmaxBridge) -> Iterator[list[str]]:
    originals = []
    names = []
    try:
        for name, module in model.named_modules():
            if isinstance(module, attention_type):
                originals.append((module, module.forward))
                names.append(name)
                module.forward = types.MethodType(_attention_forward(bridge, name), module)
        if len(names) != 2:
            raise ValueError(f"Expected two YOLO26-nano Attention modules, found {len(names)}")
        yield names
    finally:
        for module, forward in originals:
            module.forward = forward


def _tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for key in sorted(value) for tensor in _tensors(value[key])]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensors(item)]
    return []


def run_profile(
    weights: Path, size: int, output_dir: Path, iterations: int, device_id: int
) -> dict[str, Any]:
    """Compare all model tensor outputs, then measure warmed hybrid host latency."""
    ultralytics = importlib.import_module("ultralytics")
    attention_type = importlib.import_module("ultralytics.nn.modules.block").Attention
    model = ultralytics.YOLO(str(weights)).model.cpu().float().eval()
    convs = [
        {
            "name": name,
            "weight_shape": list(module.weight.shape),
            "stride": list(module.stride),
            "groups": module.groups,
        }
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    generator = torch.Generator().manual_seed(211)
    inputs = [torch.rand((1, 3, size, size), generator=generator) for _ in range(2)]
    with torch.inference_mode():
        expected = [_tensors(model(x)) for x in inputs]
        with ExitStack() as stack:
            bridge = SoftmaxBridge((size // 32) ** 2, output_dir, device_id, stack)
            errors = []
            with _native_softmax(model, attention_type, bridge) as names:
                for x, reference in zip(inputs, expected, strict=True):
                    actual = _tensors(model(x))
                    assert len(actual) == len(reference) and actual
                    tensor_errors = []
                    for result, golden in zip(actual, reference, strict=True):
                        torch.testing.assert_close(result, golden, rtol=0.0001, atol=0.002)
                        tensor_errors.append(float((result - golden).abs().max()))
                    errors.append(tensor_errors)
                bridge.check_accuracy = False
                for _ in range(2):
                    model(inputs[0])
                bridge.host_call_ms.clear()
                bridge.module_calls.clear()
                model_ms = []
                for index in range(iterations):
                    started = time.perf_counter()
                    model(inputs[index % len(inputs)])
                    model_ms.append((time.perf_counter() - started) * 1000)
            return {
                "execution_scope": "hybrid_cpu_model_with_pto3_310p_softmax",
                "full_pto3_model": False,
                "full_pto3_latency_ms": None,
                "full_pto3_fps": None,
                "input_shape": [1, 3, size, size],
                "precision": "CPU FP32 model, PTO3 FP32 softmax on 310P",
                "attention_modules": names,
                "attention_width": bridge.cols,
                "correctness_inputs": len(inputs),
                "output_shapes": [list(t.shape) for t in expected[0]],
                "output_max_abs_errors": errors,
                "max_softmax_abs_error": bridge.max_softmax_abs_error,
                "compile_and_session_ms": bridge.compile_and_session_ms,
                "kernel_image_sha256": bridge.image_sha256,
                "warmup_model_iterations": 2,
                "measured_model_iterations": iterations,
                "hybrid_model_host_latency": _summary(model_ms),
                "softmax_block_host_latency": _summary(bridge.host_call_ms),
                "softmax_block_host_calls": len(bridge.host_call_ms),
                "measured_attention_module_calls": bridge.module_calls,
                "timing_scope": (
                    "Synchronous host wall time; native block calls include allocation, H2D, "
                    "execution, D2H and release. Compilation/session creation excluded from warmed samples."
                ),
                "device_kernel_latency_ms": None,
                "cpu_conv2d_count": len(convs),
                "cpu_conv2d_inventory": convs,
                "remaining_native_work": [
                    "General convolution, depthwise convolution and fused bias/activation",
                    "Pooling, upsample, concatenation and general reshape/transpose layouts",
                    "Larger GEMM/attention products beyond the validated narrow panel profile",
                    "Device-resident buffers and graph scheduling/capture/replay",
                ],
            }


def main() -> None:
    """Run one explicit local-weight profile and save its labelled evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=[128, 640], default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not args.weights.is_file() or args.iterations < 1 or args.device < 0:
        parser.error("Require a local checkpoint, positive iterations and nonnegative device")
    torch.set_num_threads(2)
    report = run_profile(args.weights, args.size, args.output_dir, args.iterations, args.device)
    report["weight_sha256"] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    report_path = args.output_dir / "hybrid_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "cpu_conv2d_inventory"}, indent=2)
    )


if __name__ == "__main__":
    main()
