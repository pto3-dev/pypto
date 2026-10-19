# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO2-compatible constant preparation and CPU FP16 reference boundaries.

The native convolution kernels implement the same separate FP16 sigmoid and
multiply boundary. CPU model evaluation is used only by validation.
"""

from typing import Any

import torch


class FP16SiLU(torch.nn.SiLU):
    """Round sigmoid to the input dtype before multiplying, as in PTO2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


@torch.no_grad()
def prepare_fp16_model(model: Any) -> Any:
    """Fuse CPU constants using PTO2's operation order and select FP16 SiLU.

    Call on the unfused evaluation model after selecting the one-to-one head.
    No model input is evaluated during this preparation.
    """
    folds = []
    for module in model.modules():
        conv, bn = getattr(module, "conv", None), getattr(module, "bn", None)
        if not isinstance(conv, torch.nn.Conv2d) or not isinstance(bn, torch.nn.BatchNorm2d):
            continue
        gamma, beta, mean, variance = bn.weight, bn.bias, bn.running_mean, bn.running_var
        if gamma is None or beta is None or mean is None or variance is None:
            raise ValueError("FP16 preparation requires affine BatchNorm with tracked running statistics")
        scale = gamma.float() / torch.sqrt(variance.float() + bn.eps)
        bias = conv.bias.float() if conv.bias is not None else torch.zeros_like(beta)
        weight = conv.weight.float() * scale.reshape(-1, 1, 1, 1)
        folded_bias = (bias - mean.float()) * scale + beta.float()
        folds.append((module, weight, folded_bias))
    model.fuse(verbose=False)
    for module, weight, bias in folds:
        # Fusion can remove unused one-to-many head modules before visiting them.
        if not hasattr(module, "bn"):
            module.conv.weight.copy_(weight)
            module.conv.bias.copy_(bias)
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.SiLU):
                setattr(module, name, FP16SiLU(inplace=child.inplace))
    return model.half()
