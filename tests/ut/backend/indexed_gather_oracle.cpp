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

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

int main() {
  using Data = GATHER_CTYPE;
  using Bits = std::conditional_t<sizeof(Data) == 2, uint16_t, uint32_t>;
  constexpr size_t rows = GATHER_ROWS, guard = 16;
  constexpr uint32_t sentinel = 0xa55ac33cU;
  constexpr std::array<uint32_t, 14> patterns = {
      0U,          0x80000000U, 0x7fc01234U, 0xffc05678U, 0x7f800001U, 0x7f800000U, 0xffffffffU,
      0x7fffffffU, 0x80000001U, 0x01020304U, 0x7e017c01U, 0xfc017e55U, 0x00008000U, 0x00003c00U};
  for (unsigned dataset = 0; dataset < 4; ++dataset) {
    std::vector<Data> input(8 * 112 + 2 * guard), output(rows * 128 + 2 * guard);
    std::vector<int32_t> indices(rows * 112 + 2 * guard, 7);
    for (size_t i = 0; i < input.size(); ++i) {
      const uint32_t bits = patterns[(i + dataset) % patterns.size()] ^
                            (dataset == 3 ? static_cast<uint32_t>(i) * 1664525U : 0);
      input[i] = std::bit_cast<Data>(static_cast<Bits>(bits));
    }
    for (auto& value : output) value = std::bit_cast<Data>(static_cast<Bits>(sentinel));
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < 96; ++col) {
        const size_t i = row * 96 + col;
        indices[guard + row * 112 + 8 + col] = dataset == 0   ? i % 640
                                               : dataset == 1 ? 639 - i % 640
                                               : dataset == 2 ? (i % 7 == 0 ? 0 : 639)
                                                              : (i * 37 + 19) % 640;
      }
    }
    const auto original = input;
    const auto originalIndices = indices;
    std::array<Tensor, 3> tensors;
    const uint32_t shapes[3] = {8 * 112, rows * 112, rows * 128};
    tensors[0].init_external(input.data(), input.size() * sizeof(Data), &shapes[0], 1, GATHER_DTYPE, 0);
    tensors[1].init_external(indices.data(), indices.size() * sizeof(int32_t), &shapes[1], 1, DataType::INT32,
                             0);
    tensors[2].init_external(output.data(), output.size() * sizeof(Data), &shapes[2], 1, GATHER_DTYPE, 0);
    const std::array<size_t, 3> order = {GATHER_ARG0, GATHER_ARG1, GATHER_ARG2};
    std::array<int64_t, 3> args{};
    for (size_t i = 0; i < tensors.size(); ++i) {
      tensors[i].start_offset = guard;
      args[i] = reinterpret_cast<int64_t>(&tensors[order[i]]);
    }
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    if (std::memcmp(input.data(), original.data(), input.size() * sizeof(Data)) || indices != originalIndices)
      return 1;
    for (size_t i = 0; i < output.size(); ++i) {
      Bits expected = static_cast<Bits>(sentinel);
      if (i >= guard && i < output.size() - guard) {
        const size_t row = (i - guard) / 128, col = (i - guard) % 128;
        if (col >= 16 && col < 112) {
          const size_t index = originalIndices[guard + row * 112 + 8 + col - 16];
          expected = std::bit_cast<Bits>(original[guard + (index / 80) * 112 + 16 + index % 80]);
        }
      }
      if (std::bit_cast<Bits>(output[i]) != expected) return 2;
    }
  }
  return 0;
}
