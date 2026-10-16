# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate and time complete native YOLO26-nano execution on Ascend310P.

Run as a module with local Ultralytics installed and --weights supplied.
Timings include synchronous host dispatch and explicit input/output transfer;
they are not device-only kernel timings or hardware graph capture.
"""

import argparse
import ctypes
import hashlib
import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from .executor import NativeYOLO26, RejectHostOperators
from .precision import prepare_fp16_model


def _hardware(user_device_id: int) -> dict[str, Any]:
    """Report the current logical device and its physical CANN mapping."""
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    logical = ctypes.c_int32()
    physical = ctypes.c_int32()
    acl.aclrtGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    acl.aclrtGetDevice.restype = ctypes.c_int
    mapping = acl.aclrtGetPhyDevIdByLogicDevId
    mapping.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    mapping.restype = ctypes.c_int
    if acl.aclrtGetDevice(ctypes.byref(logical)) or mapping(logical.value, ctypes.byref(physical)):
        raise RuntimeError("CANN device identity query failed")
    if logical.value != user_device_id:
        raise RuntimeError("CANN current device does not match the native session")
    return {
        "soc": acl.aclrtGetSocName().decode(),
        "logical_device_id": logical.value,
        "physical_device_id": physical.value,
    }


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}


def _selection_is_exact(dense: torch.Tensor, actual: torch.Tensor) -> None:
    """Check full TopK semantics, allowing arbitrary ordering of equal scores."""
    values = dense[:, 4:84].reshape(-1).topk(300).values
    torch.testing.assert_close(actual[:, 4], values, rtol=0, atol=0)
    used = set()
    for row in actual:
        label = int(row[5])
        assert float(label) == float(row[5]) and 0 <= label < 80
        matches = (
            torch.nonzero((dense[:, :4] == row[:4]).all(1) & (dense[:, 4 + label] == row[4]))
            .reshape(-1)
            .tolist()
        )
        available = [anchor for anchor in matches if (anchor, label) not in used]
        assert available, "native detection has no unique matching dense anchor/class"
        used.add((available[0], label))


def _cpu_reference(model: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    features = {}
    handles = []
    for module in model.model[:-1]:

        def capture(_module: Any, _args: Any, output: torch.Tensor, index: int = module.i) -> None:
            features[index] = output.detach().float().clone()

        handles.append(module.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            final, raw = model(x.half())
            dense = model.model[-1]._inference(raw["one2one"]).transpose(1, 2)[0].float()
        return final[0].float(), dense, features
    finally:
        for handle in handles:
            handle.remove()


def _validate(model: Any, native: NativeYOLO26, x: torch.Tensor) -> dict[str, Any]:
    expected, dense, features = _cpu_reference(model, x)
    native.input.copy_from(x.contiguous().reshape(-1))
    guard = RejectHostOperators()
    with guard:
        native.replay()
    assert guard.calls == 0
    actual = native.output.to_cpu().reshape(300, 6)
    dense_actual = native.dense.to_cpu().reshape(-1, 96)[:, :84]
    actual_features = {index: native.features[index].to_cpu().float() for index in features}
    feature_errors = {
        str(index): _error(actual_features[index], target) for index, target in features.items()
    }
    _selection_is_exact(dense_actual, actual)
    probability_delta = (dense_actual[:, 4:] - dense[:, 4:]).abs()
    score_delta = (actual[:, 4] - expected[:, 4]).abs()
    result = {
        "host_aten_operations_during_replay": guard.calls,
        "feature_errors": feature_errors,
        "decoded_box_error": _error(dense_actual[:, :4], dense[:, :4]),
        "class_probability_error": _error(dense_actual[:, 4:], dense[:, 4:]),
        "final_sorted_score_error": _error(actual[:, 4], expected[:, 4]),
        "strict_dense_probability_mismatches": int(
            (probability_delta > 1e-6 + 0.03 * dense[:, 4:].abs()).sum()
        ),
        "strict_final_score_mismatches": int((score_delta > 1e-6 + 0.03 * expected[:, 4].abs()).sum()),
        "strict_probability_tolerance": {"rtol": 0.03, "atol": 1e-6},
        "topk_semantics_exact_against_native_dense": True,
        "output_shape": [1, 300, 6],
    }
    input_digest = hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16]
    diagnostic = native.output_dir.parent / f"validation_{input_digest}.json"
    diagnostic.write_text(json.dumps(result, indent=2) + "\n")
    for index, target in features.items():
        torch.testing.assert_close(
            actual_features[index], target, rtol=0.02, atol=0.02, msg=f"native YOLO26 feature {index}"
        )
    torch.testing.assert_close(dense_actual[:, :4], dense[:, :4], rtol=0.005, atol=0.5)
    torch.testing.assert_close(dense_actual[:, 4:], dense[:, 4:], rtol=0.03, atol=0.000001)
    torch.testing.assert_close(actual[:, 4], expected[:, 4], rtol=0.03, atol=0.000001)
    return result


def _timings(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "p95_ms": ordered[min(len(values) - 1, int(0.95 * len(values)))],
        "fps_from_median": 1000 / statistics.median(values),
    }


def run_profile(
    weights: Path,
    size: int,
    output_dir: Path,
    iterations: int,
    device_id: int = 0,
    *,
    precision_policy: str = "auto",
    weight_l2_bypass: bool = False,
) -> dict[str, Any]:
    """Validate changing inputs and report warmed full-native host latency."""
    if not weights.is_file() or iterations < 3:
        raise ValueError("supply local weights and at least three iterations")
    if precision_policy not in ("auto", "cube", "pto2"):
        raise ValueError(f"expected precision policy auto/cube/pto2, got {precision_policy!r}")
    selected_policy = "pto2" if precision_policy == "auto" and size == 128 else precision_policy
    if selected_policy == "auto":
        selected_policy = "cube"
    if selected_policy == "pto2" and size != 128:
        raise ValueError("the PTO2 precision policy is validated for 128x128 only")
    precise = selected_policy == "pto2"
    torch.set_num_threads(1)
    ultralytics = importlib.import_module("ultralytics")
    model = ultralytics.YOLO(str(weights)).model.eval()
    model.model[-1].end2end = True
    model = prepare_fp16_model(model)
    started = time.perf_counter()
    with NativeYOLO26(
        model,
        size,
        output_dir / "compiled",
        device_id,
        four_way_accumulation=precise,
        linear_query_key=precise,
        precise_small_softmax=precise,
        weight_l2_bypass=weight_l2_bypass,
    ) as native:
        prepare_ms = (time.perf_counter() - started) * 1000
        coverage = native.coverage()
        hardware = _hardware(device_id)
        checks = []
        for seed in (313, 317):
            x = torch.rand(
                (1, 3, size, size), generator=torch.Generator().manual_seed(seed), dtype=torch.float32
            )
            checks.append(_validate(model, native, x))
        for _ in range(3):
            native.forward(x)
        full = []
        replay = []
        for _ in range(iterations):
            started = time.perf_counter()
            native.forward(x)
            full.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            native.replay()
            replay.append((time.perf_counter() - started) * 1000)
        result = {
            "model": "YOLO26-nano",
            "hardware": hardware,
            "ultralytics_version": ultralytics.__version__,
            "native_sources_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.iterdir())
                if path.suffix in (".py", ".cpp")
            },
            "size": size,
            "precision_policy": {
                "requested": precision_policy,
                "selected": selected_policy,
                "convolution": "four_way_fp32" if precise else "cube_fp32",
                "depthwise": "four_way_fp32" if precise else "linear_fp32",
                "query_key": "linear_fp32" if precise else "cube_fp32",
                "small_softmax": "fp32_compensated" if precise else "dsl_fp32",
            },
            "weight_l2_cache": {
                "policy": "bypass" if native.weight_l2_bypass else "normal",
                "runtime_offset": native.weight_l2_cache_offset,
                "scope": "read-only ordinary four-way convolution weights",
            },
            "batch": 1,
            "inputs": {
                "kind": "synthetic_uniform_0_1",
                "layout": "NCHW",
                "dtype": "FP32",
                "validation_seeds": [313, 317],
                "timing_seed": 317,
            },
            "dtype": "FP16 with FP32 accumulation/softmax",
            "fp16_reference": "PTO2 BatchNorm folding order and separate FP16 sigmoid/multiply",
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "execution": (
                "PTO3 public external and DSL kernels; all model activation arithmetic on Ascend310P"
            ),
            "cpu_model_fallback": False,
            "hardware_graph_capture": False,
            "queue_isolated": False,
            "preparation_ms": prepare_ms,
            "preparation_scope": (
                "Native trace/JIT construction, constant packing/upload and device allocation; "
                "checkpoint loading and CPU fusion excluded."
            ),
            "coverage": coverage,
            "validation": checks,
            "iterations": iterations,
            "samples_ms": {"full_forward": full, "resident_replay": replay},
            "full_forward_with_transfers": _timings(full),
            "resident_replay_host_dispatch": _timings(replay),
            "timing_scope": (
                "Synchronous host wall time; replay includes host dispatch/synchronization, "
                "excludes input/output copies; neither measurement is device-only."
            ),
            "topk_ties": (
                "Equal-score ties use deterministic candidate traversal order; "
                "semantic membership and sorted scores are checked rather than arbitrary Torch tie ordering."
            ),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(128, 640), default=640)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--precision-policy",
        choices=("auto", "cube", "pto2"),
        default="auto",
        help=(
            "auto selects PTO2 precision for 128x128 and Cube for 640x640; "
            "PTO2 policy is validated at 128 only"
        ),
    )
    parser.add_argument(
        "--weight-l2-bypass",
        action="store_true",
        help="bypass L2 for read-only ordinary four-way convolution weights; requires PTO2 precision policy",
    )
    args = parser.parse_args()
    result = run_profile(
        args.weights,
        args.size,
        args.output_dir,
        args.iterations,
        args.device_id,
        precision_policy=args.precision_policy,
        weight_l2_bypass=args.weight_l2_bypass,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
