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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
constexpr int kElements = CW_ROWS * 16, kGuard = 16;
constexpr int kOutput = (CW_ROWS + 32) * CW_COLUMNS;
template <typename T>
bool read_data(const char* path, std::vector<T>& values) {
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
  return stream.good() && stream.peek() == std::char_traits<char>::eof();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) return 10;
  std::vector<float> all_source(3 * kElements), all_bias(3 * 16);
  std::vector<half> all_expected(3 * kElements);
  if (!read_data(argv[1], all_source) || !read_data(argv[2], all_bias) || !read_data(argv[3], all_expected))
    return 11;
  for (int dataset = 0; dataset < 3; ++dataset) {
    pto::cpu_sim::ScopedExecutionContext context(0, 0);
    std::vector<float> source(kElements + 2 * kGuard, 7), bias(16 + 2 * kGuard, 11);
    std::vector<half> output(kOutput + 2 * kGuard, half(13)), expected = output;
    std::copy_n(all_source.data() + dataset * kElements, kElements, source.data() + kGuard);
    std::copy_n(all_bias.data() + dataset * 16, 16, bias.data() + kGuard);
    const auto source_before = source, bias_before = bias;
    for (int row = 0; row < CW_ROWS; ++row)
      std::copy_n(all_expected.data() + dataset * kElements + row * 16, 16,
                  expected.data() + kGuard + (row + 16) * CW_COLUMNS + CW_COLUMN);
    const uint32_t xshape[2] = {CW_ROWS, 16}, bshape[2] = {1, 16};
    const uint32_t yshape[2] = {CW_ROWS + 32, CW_COLUMNS};
    Tensor x, b, y;
    x.init_external(source.data(), source.size() * sizeof(float), xshape, 2, DataType::FLOAT32, 0);
    b.init_external(bias.data(), bias.size() * sizeof(float), bshape, 2, DataType::FLOAT32, 0);
    y.init_external(output.data(), output.size() * sizeof(half), yshape, 2, DataType::FLOAT16, 0);
    x.start_offset = b.start_offset = y.start_offset = kGuard;
    int64_t args[3] = {reinterpret_cast<int64_t>(&x), reinterpret_cast<int64_t>(&b),
                       reinterpret_cast<int64_t>(&y)};
    kernel_entry(args);
    for (int i = 0; i < kOutput; ++i) {
      if (std::memcmp(output.data() + kGuard + i, expected.data() + kGuard + i, sizeof(half))) {
        std::fprintf(stderr, "dataset=%d index=%d actual=%.9g expected=%.9g\n", dataset, i,
                     static_cast<double>(output[kGuard + i]), static_cast<double>(expected[kGuard + i]));
        return 1;
      }
    }
    if (std::memcmp(output.data(), expected.data(), output.size() * sizeof(half))) return 2;
    if (std::memcmp(source.data(), source_before.data(), source.size() * sizeof(float))) return 3;
    if (std::memcmp(bias.data(), bias_before.data(), bias.size() * sizeof(float))) return 4;
  }
  return 0;
}
