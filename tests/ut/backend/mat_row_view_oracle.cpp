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
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
  constexpr int guard = 16, source_rows = (MR_BLOCKS + 1) * MR_M;
  constexpr int ni = source_rows * MR_STRIDE, no = (source_rows + 32) * MR_CHANNELS;
  for (int dataset = 0; dataset < 3; ++dataset) {
    pto::cpu_sim::ScopedExecutionContext initial(0, 0);
    std::vector<half> source(ni + 2 * guard, half(7)), weights(256 + 2 * guard, half(11));
    std::vector<float> output(no + 2 * guard, 13), expected = output;
    for (int i = 0; i < ni; ++i) {
      if (dataset == 0) {
        source[guard + i] = half(((i * 13) % 101 - 50) / 16.0f);
      } else {
        // Enumerate both signs of finite FP16 encodings, excluding NaN/Inf.
        // The last dataset scales them into small and subnormal values.
        unsigned bits = (i * 73 + dataset * 17) % 63488;
        bits = bits < 31744 ? bits : (bits - 31744) | 0x8000;
        const half value = std::bit_cast<half>(static_cast<uint16_t>(bits));
        source[guard + i] = dataset == 1 ? value : half(static_cast<float>(value) / 65536);
      }
    }
    // Identity weights isolate alias/addressing behavior from Cube reduction
    // trees. The scalar FMA reference also preserves signed-zero semantics.
    for (int k = 0; k < 16; ++k) {
      for (int n = 0; n < 16; ++n) {
        weights[guard + k * 16 + n] = half(k == n ? 1 : 0);
      }
    }
    const auto old_source = source, old_weights = weights;
    for (int row = 0; row < MR_BLOCKS * MR_M; ++row) {
      for (int col = 0; col < MR_CHANNELS; ++col) {
        float value = 0;
        for (int k = 0; k < 16; ++k) {
          value =
              std::fma(static_cast<float>(source[guard + row * MR_STRIDE + MR_OFFSET + col / 16 * 16 + k]),
                       static_cast<float>(weights[guard + k * 16 + col % 16]), value);
        }
        expected[guard + (row + 16) * MR_CHANNELS + col] = value;
      }
    }
    std::array<Tensor, 3> tensors;
    const uint32_t shapes[3][2] = {{source_rows, MR_STRIDE}, {16, 16}, {source_rows + 32, MR_CHANNELS}};
    void* addresses[] = {source.data(), weights.data(), output.data()};
    const uint64_t sizes[] = {source.size() * sizeof(half), weights.size() * sizeof(half),
                              output.size() * sizeof(float)};
    for (int i = 0; i < 3; ++i) {
      tensors[i].init_external(addresses[i], sizes[i], shapes[i], 2,
                               i == 2 ? DataType::FLOAT32 : DataType::FLOAT16, 0);
      tensors[i].start_offset = guard;
    }
    constexpr int order[] = {MR_ARG0, MR_ARG1, MR_ARG2};
    std::array<int64_t, 3> args{};
    for (int i = 0; i < 3; ++i) {
      args[i] = reinterpret_cast<int64_t>(&tensors[order[i]]);
    }
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    if (std::memcmp(output.data(), expected.data(), output.size() * sizeof(float))) {
      for (int i = 0; i < static_cast<int>(output.size()); ++i) {
        if (std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(expected[i])) {
          std::cerr << "dataset=" << dataset << " index=" << i << " actual=" << output[i]
                    << " expected=" << expected[i] << '\n';
          break;
        }
      }
      return 1;
    }
    if (std::memcmp(source.data(), old_source.data(), source.size() * sizeof(half)) ||
        std::memcmp(weights.data(), old_weights.data(), weights.size() * sizeof(half))) {
      return 2;
    }
  }
  return 0;
}
