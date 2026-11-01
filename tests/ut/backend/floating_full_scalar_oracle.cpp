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

#include <cstdint>
#include <cstring>
#include <vector>

int main() {
  using Element = FS_TYPE;
  constexpr int kGuard = 16;
  constexpr int kElements = FS_ROWS * 16;
  std::vector<Element> output(kElements + 2 * kGuard, Element(7));
  auto expected = output;
  for (int i = 0; i < kElements; ++i) expected[kGuard + i] = Element(FS_VALUE);
  const uint32_t shape[2] = {FS_ROWS, 16};
  Tensor tensor;
  tensor.init_external(output.data(), output.size() * sizeof(Element), shape, 2, DataType::FS_DTYPE, 0);
  tensor.start_offset = kGuard;
  int64_t args[1] = {reinterpret_cast<int64_t>(&tensor)};
  pto::cpu_sim::ScopedExecutionContext context(0, 0);
  kernel_entry(args);
  return std::memcmp(output.data(), expected.data(), output.size() * sizeof(Element)) ? 1 : 0;
}
