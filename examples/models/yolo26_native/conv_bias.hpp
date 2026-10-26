/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#ifndef EXAMPLES_MODELS_YOLO26_NATIVE_CONV_BIAS_HPP_
#define EXAMPLES_MODELS_YOLO26_NATIVE_CONV_BIAS_HPP_

#include <pto/pto-inst.hpp>

// Keep UB in adjacent Nx16 panels, matching L0C's NZ column blocks. A zero
// bias repeat stride reuses C16 across M rows instead of reissuing scalar
// loop control and masks per row. Each lane retains the same FP32 addition;
// the caller orders bias DMA before this operation and narrowing after it.
template <unsigned Rows, unsigned Columns = 16>
AICORE inline void add_conv_bias(__ubuf__ float* value, __ubuf__ float* bias) {
  static_assert(Rows > 0 && Rows <= 255, "A1 bias repeats must fit the instruction repeat field");
  static_assert(Columns > 0 && Columns % 16 == 0, "A1 bias panels must contain full C16 columns");
  set_mask_norm();
  set_vector_mask(0, 0xffff);
  for (unsigned column = 0; column < Columns / 16; ++column)
    vadd(value + column * Rows * 16, value + column * Rows * 16, bias + column * 16, Rows, 1, 1, 1, 2, 2, 0);
}
#endif  // EXAMPLES_MODELS_YOLO26_NATIVE_CONV_BIAS_HPP_
