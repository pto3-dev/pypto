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

// Independent staged ordering and raw field extraction, generated code unchanged.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
struct Candidate {
  float score;
  uint32_t index;
};
float Score(unsigned dataset, unsigned panel, unsigned column) {
  constexpr std::array<uint32_t, 8> special = {0U,          0x80000000U, 0x7f800000U, 0xff800000U,
                                               0x3f800000U, 0xbf800000U, 1U,          0x80000001U};
  if (dataset == 1) return 0.5f;
  if (dataset == 2) return std::bit_cast<float>(special[(column + panel) % special.size()]);
  if (dataset == 3) return static_cast<float>(column);
  if (dataset == 4) return -static_cast<float>(column);
  return static_cast<float>((column / (dataset == 5 ? 3 : 1) + panel * 11) % 31) - 15.0f;
}
uint32_t Index(unsigned dataset, unsigned panel, unsigned column) {
  return dataset == 1 ? UINT32_MAX - column * 31
                      : 0x80000000U | (panel * 4099U ^ column * 129U ^ dataset << 24);
}
}  // namespace

int main() {
  constexpr unsigned panels = 5, columns = SORT_COLS, pairs = EXTRACT_PAIRS, guard = 16;
  constexpr unsigned inputStride = columns + 16, outputStride = pairs + 16;
  constexpr unsigned inputCount = panels * inputStride, outputCount = panels * outputStride;
  constexpr uint32_t sentinel = 0xa55ac33cU;
  for (unsigned dataset = 0; dataset < 6; ++dataset) {
    std::vector<float> input(inputCount + 2 * guard, std::bit_cast<float>(sentinel));
    std::vector<uint32_t> inputIndices(input.size(), sentinel), positions(outputCount + 2 * guard, sentinel);
    std::vector<float> values(outputCount + 2 * guard, std::bit_cast<float>(sentinel));
    auto expectedValues = values;
    auto expectedPositions = positions;
    for (unsigned panel = 0; panel < panels; ++panel) {
      std::vector<Candidate> ordered;
      for (unsigned start = 0; start < columns; start += 32) {
        std::array<Candidate, 32> block;
        for (unsigned i = 0; i < block.size(); ++i) {
          block[i] = {Score(dataset, panel, start + i), Index(dataset, panel, start + i)};
          input[guard + panel * inputStride + 8 + start + i] = block[i].score;
          inputIndices[guard + panel * inputStride + 8 + start + i] = block[i].index;
        }
        std::stable_sort(block.begin(), block.end(), [](const Candidate& a, const Candidate& b) {
          return a.score > b.score || (a.score == b.score && a.index < b.index);
        });
        ordered.insert(ordered.end(), block.begin(), block.end());
      }
      // Each sort32 block uses its unsigned index tie key. Merge levels then
      // preserve earlier-run priority, rather than globally re-sorting payloads.
      std::stable_sort(ordered.begin(), ordered.end(),
                       [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
      for (unsigned i = 0; i < pairs; ++i) {
        const auto at = guard + panel * outputStride + 8 + i;
        expectedValues[at] = ordered[i].score;
        expectedPositions[at] = ordered[i].index;
      }
    }
    const auto original = input;
    const auto originalIndices = inputIndices;
    std::array<Tensor, 4> tensors;
    const uint32_t shapes[4] = {inputCount, inputCount, outputCount, outputCount};
    tensors[0].init_external(input.data(), input.size() * sizeof(float), &shapes[0], 1, DataType::FLOAT32, 0);
    tensors[1].init_external(inputIndices.data(), inputIndices.size() * sizeof(uint32_t), &shapes[1], 1,
                             DataType::UINT32, 0);
    tensors[2].init_external(values.data(), values.size() * sizeof(float), &shapes[2], 1, DataType::FLOAT32,
                             0);
    tensors[3].init_external(positions.data(), positions.size() * sizeof(uint32_t), &shapes[3], 1,
                             DataType::UINT32, 0);
    std::array<int64_t, 4> args{};
    for (unsigned i = 0; i < tensors.size(); ++i) {
      tensors[i].start_offset = guard;
      args[i] = reinterpret_cast<int64_t>(&tensors[i]);
    }
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    if (std::memcmp(input.data(), original.data(), input.size() * sizeof(float)) ||
        inputIndices != originalIndices)
      return 1;
    if (std::memcmp(values.data(), expectedValues.data(), values.size() * sizeof(float)) ||
        positions != expectedPositions) {
      std::fprintf(stderr, "sort/merge/extract dataset %u field or guard mismatch\n", dataset);
      return 2;
    }
  }
  return 0;
}
