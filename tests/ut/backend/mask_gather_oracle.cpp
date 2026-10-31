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

// Appended to unmodified generated Tensor-ABI code; every output bit is checked.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using Data = MASK_TYPE;
using Index = MASK_INDEX_TYPE;
constexpr unsigned rows = MASK_ROWS, columns = MASK_COLS, panels = MASK_PANELS;
constexpr unsigned factor = MASK_A <= 2 ? 2 : MASK_A <= 6 ? 4 : 1;
constexpr unsigned outputColumns = columns / factor;
constexpr unsigned inputStride = columns + 32, outputStride = outputColumns + 32;
constexpr unsigned totalRows = panels * (rows + 2), guard = 16;
constexpr unsigned inputCount = totalRows * inputStride, outputCount = totalRows * outputStride;
constexpr uint32_t sentinel = 0xa55ac33cU;
unsigned Offset(unsigned pattern) { return pattern <= 2 ? pattern - 1 : pattern <= 6 ? pattern - 3 : 0; }
}  // namespace

int main() {
  constexpr std::array<uint32_t, 16> patterns = {
      0U,          0x80000000U, 0x7fc01234U, 0xffc05678U, 0x7f800001U, 0x7f800000U, 0xffffffffU, 0x80000001U,
      0x01020304U, 0x00008000U, 0x00007c00U, 0x0000fc00U, 0x00007e13U, 0x0000fe55U, 1U,          0x00007bffU};
  for (unsigned dataset = 0; dataset < 6; ++dataset) {
    std::vector<Data> input(inputCount + 2 * guard), values(outputCount + 2 * guard);
    std::vector<Index> indices(outputCount + 2 * guard);
    for (unsigned i = 0; i < input.size(); ++i) {
      const uint32_t bits = dataset < 2         ? patterns[(i + dataset) % patterns.size()]
                            : sizeof(Data) == 2 ? (i + dataset * inputCount) & 0xffffU
                                                : patterns[(i + dataset) % patterns.size()] ^ i * 1664525U;
      std::memcpy(&input[i], &bits, sizeof(Data));
    }
    for (auto& value : values) std::memcpy(&value, &sentinel, sizeof(Data));
    for (auto& value : indices) std::memcpy(&value, &sentinel, sizeof(Index));
    auto expectedValues = values;
    auto expectedIndices = indices;
    for (unsigned panel = 0; panel < panels; ++panel) {
      for (unsigned row = 0; row < rows; ++row) {
        const unsigned physicalRow = panel * (rows + 2) + row + 1;
        for (unsigned column = 0; column < outputColumns; ++column) {
          const auto at = guard + physicalRow * outputStride + 16 + column;
          const auto first = guard + physicalRow * inputStride + 16 + column * factor + Offset(MASK_A);
          const auto second = guard + physicalRow * inputStride + 16 + column * factor + Offset(MASK_B);
          std::memcpy(&expectedValues[at], &input[first], sizeof(Data));
          std::memcpy(&expectedIndices[at], &input[second], sizeof(Index));
        }
      }
    }
    const auto original = input;
    std::array<Tensor, 3> tensors;
    const uint32_t shapes[3] = {inputCount, outputCount, outputCount};
    tensors[0].init_external(input.data(), input.size() * sizeof(Data), &shapes[0], 1,
                             DataType::MASK_DATA_TYPE, 0);
    tensors[1].init_external(values.data(), values.size() * sizeof(Data), &shapes[1], 1,
                             DataType::MASK_DATA_TYPE, 0);
    tensors[2].init_external(indices.data(), indices.size() * sizeof(Index), &shapes[2], 1,
                             sizeof(Index) == 2 ? DataType::FLOAT16 : DataType::UINT32, 0);
    const std::array<unsigned, 3> order = {MASK_ARG_0, MASK_ARG_1, MASK_ARG_2};
    std::array<int64_t, 3> args{};
    for (auto& tensor : tensors) tensor.start_offset = guard;
    for (unsigned i = 0; i < args.size(); ++i) args[i] = reinterpret_cast<int64_t>(&tensors[order[i]]);
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    if (std::memcmp(input.data(), original.data(), input.size() * sizeof(Data))) return 1;
    if (std::memcmp(values.data(), expectedValues.data(), values.size() * sizeof(Data)) ||
        std::memcmp(indices.data(), expectedIndices.data(), indices.size() * sizeof(Index))) {
      std::fprintf(stderr, "mask gather dataset %u field/guard mismatch\n", dataset);
      return 2;
    }
  }
  return 0;
}
