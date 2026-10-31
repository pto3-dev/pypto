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

// Appended to unmodified PyPTO-generated Tensor ABI code. No floating tolerance.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
struct Pair {
  float score;
  uint32_t index;
};

float Score(unsigned dataset, unsigned row, unsigned column) {
  constexpr std::array<uint32_t, 8> patterns = {0U,          0x80000000U, 0x7f800000U, 0xff800000U,
                                                0x3f800000U, 0xbf800000U, 1U,          0x80000001U};
  switch (dataset) {
    case 0:
      return static_cast<float>((column * 17 + row * 13) % 101) - 50.0f;
    case 1:
      return 0.5f;
    case 2:
      return std::bit_cast<float>(patterns[(column + row) % patterns.size()]);
    case 3:
      return static_cast<float>(column);
    case 4:
      return -static_cast<float>(column);
    default:
      return static_cast<float>((column / 3 + row) % 7);
  }
}

uint32_t Index(unsigned dataset, unsigned row, unsigned column) {
  constexpr std::array<uint32_t, 4> high = {0U, 0x80000000U, 0x7fff0000U, 0xffff0000U};
  if (dataset == 1) return 0xffffffffU - (column * 37 + row * 13) % SORT_COLS;
  if (dataset == 2) return high[column % 4] | column;
  return (0x80000000U | (row * 4099 + column * 257)) ^ (dataset << 24);
}
}  // namespace

int main() {
  constexpr unsigned rows = SORT_ROWS * SORT_PANELS;
  constexpr unsigned columns = SORT_COLS;
  constexpr unsigned input_stride = columns + 16, output_stride = 2 * columns + 16, guard = 16;
  constexpr uint32_t sentinel = 0xa55ac33cU;
  for (unsigned dataset = 0; dataset < 6; ++dataset) {
    std::vector<float> input(rows * input_stride + 2 * guard, std::bit_cast<float>(sentinel));
    std::vector<uint32_t> indices(input.size(), sentinel);
    std::vector<float> output(rows * output_stride + 2 * guard, std::bit_cast<float>(sentinel));
    auto expected = output;
    for (unsigned row = 0; row < rows; ++row) {
      for (unsigned block = 0; block < columns; block += 32) {
        std::array<Pair, 32> pairs;
        for (unsigned i = 0; i < pairs.size(); ++i) {
          pairs[i] = {Score(dataset, row, block + i), Index(dataset, row, block + i)};
          input[guard + row * input_stride + 8 + block + i] = pairs[i].score;
          indices[guard + row * input_stride + 8 + block + i] = pairs[i].index;
        }
        std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
          return a.score > b.score || (a.score == b.score && a.index < b.index);
        });
        for (unsigned i = 0; i < pairs.size(); ++i) {
          const auto at = guard + row * output_stride + 8 + 2 * (block + i);
          expected[at] = pairs[i].score;
          expected[at + 1] = std::bit_cast<float>(pairs[i].index);
        }
      }
    }
    const auto original = input;
    const auto original_indices = indices;
    std::array<Tensor, 3> tensors;
    const uint32_t shapes[3] = {rows * input_stride, rows * input_stride, rows * output_stride};
    tensors[0].init_external(input.data(), input.size() * sizeof(float), &shapes[0], 1, DataType::FLOAT32, 0);
    tensors[1].init_external(indices.data(), indices.size() * sizeof(uint32_t), &shapes[1], 1,
                             DataType::UINT32, 0);
    tensors[2].init_external(output.data(), output.size() * sizeof(float), &shapes[2], 1, DataType::FLOAT32,
                             0);
    const std::array<unsigned, 3> order = {SORT_ARG0, SORT_ARG1, SORT_ARG2};
    std::array<int64_t, 3> args{};
    for (unsigned i = 0; i < tensors.size(); ++i) {
      tensors[i].start_offset = guard;
      args[i] = reinterpret_cast<int64_t>(&tensors[order[i]]);
    }
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    if (std::memcmp(input.data(), original.data(), input.size() * sizeof(float)) ||
        indices != original_indices)
      return 1;
    for (unsigned i = 0; i < output.size(); ++i) {
      if (std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(expected[i])) {
        std::fprintf(stderr, "sort32 dataset %u element %u mismatch: %08x != %08x\n", dataset, i,
                     std::bit_cast<uint32_t>(output[i]), std::bit_cast<uint32_t>(expected[i]));
        return 2;
      }
    }
  }
  return 0;
}
