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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
constexpr int kElements = SILU_ELEMENTS, kGuard = 16;

bool read_data(const char* path, std::vector<half>& values) {
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(half));
  return stream.good() && stream.peek() == std::char_traits<char>::eof();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 10;
  std::vector<half> all_input(2 * kElements), all_expected(2 * kElements);
  if (!read_data(argv[1], all_input) || !read_data(argv[2], all_expected)) return 11;
  const int cores = std::atoi(std::getenv("PTO_CPU_SIM_NUM_CORES"));
  for (int dataset = 0; dataset < 2; ++dataset) {
    std::vector<half> input(kElements + 2 * kGuard, half(7));
    std::vector<half> output(kElements + 2 * kGuard, half(13)), expected = output;
    std::copy_n(all_input.data() + dataset * kElements, kElements, input.data() + kGuard);
    std::copy_n(all_expected.data() + dataset * kElements, kElements, expected.data() + kGuard);
    const auto input_before = input;
    const uint32_t shape[2] = {kElements / 16, 16};
    Tensor x, y;
    x.init_external(input.data(), input.size() * sizeof(half), shape, 2, DataType::FLOAT16, 0);
    y.init_external(output.data(), output.size() * sizeof(half), shape, 2, DataType::FLOAT16, 0);
    x.start_offset = y.start_offset = kGuard;
    int64_t args[2] = {reinterpret_cast<int64_t>(&x), reinterpret_cast<int64_t>(&y)};
    for (int core = 0; core < cores; ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args);
    }
    for (int i = 0; i < kElements; ++i) {
      if (std::memcmp(output.data() + kGuard + i, expected.data() + kGuard + i, sizeof(half))) {
        std::fprintf(stderr, "dataset=%d index=%d input=%.9g actual=%.9g expected=%.9g\n", dataset, i,
                     static_cast<double>(input[kGuard + i]), static_cast<double>(output[kGuard + i]),
                     static_cast<double>(expected[kGuard + i]));
        return 1;
      }
    }
    if (std::memcmp(output.data(), expected.data(), output.size() * sizeof(half))) return 2;
    if (std::memcmp(input.data(), input_before.data(), input.size() * sizeof(half))) return 3;
  }
  return 0;
}
