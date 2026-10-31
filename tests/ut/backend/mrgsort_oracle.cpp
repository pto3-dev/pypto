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

// Appended to unmodified generated code; compare score/payload bits and all guards.
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
float Score(unsigned dataset, unsigned panel, unsigned source_number, unsigned item) {
  constexpr std::array<uint32_t, 8> special = {0U,          0x80000000U, 0x7f800000U, 0xff800000U,
                                               0x3f800000U, 0xbf800000U, 1U,          0x80000001U};
  switch (dataset) {
    case 0:
      return static_cast<float>((item * 17 + panel * 13 + source_number * 11) % 101) - 50.0f;
    case 1:
      return 0.5f;
    case 2:
      return std::bit_cast<float>(special[(item + panel + source_number) % special.size()]);
    case 3:
      return static_cast<float>(source_number * 1000 + item);
    case 4:
      return -static_cast<float>(item);
    default:
      return static_cast<float>((item / 3 + panel + source_number) % 7);
  }
}
}  // namespace

int main() {
  constexpr std::array<unsigned, 4> widths = {MERGE_WIDTH0, MERGE_WIDTH1, MERGE_WIDTH2, MERGE_WIDTH3};
  constexpr unsigned input_stride = *std::max_element(widths.begin(), widths.begin() + MERGE_WAYS) + 16;
  constexpr unsigned output_stride = MERGE_COLS + 16, guard = 16;
  constexpr unsigned input_count = MERGE_PANELS * 4 * input_stride;
  constexpr unsigned output_count = MERGE_PANELS * output_stride;
  constexpr uint32_t sentinel = 0xa55ac33cU;
  for (unsigned dataset = 0; dataset < 6; ++dataset) {
    std::vector<float> input(input_count + 2 * guard, std::bit_cast<float>(sentinel));
    std::vector<float> output(output_count + 2 * guard, std::bit_cast<float>(sentinel));
    auto expected = output;
    for (unsigned panel = 0; panel < MERGE_PANELS; ++panel) {
      std::vector<Pair> combined;
      for (unsigned source_number = 0; source_number < MERGE_WAYS; ++source_number) {
        std::vector<Pair> run;
        for (unsigned item = 0; item < widths[source_number] / 2; ++item) {
          const uint32_t payload = (0xffffffffU - item * 37U - panel * 4099U) ^ (source_number << 24);
          run.push_back({Score(dataset, panel, source_number, item), payload});
        }
        std::stable_sort(run.begin(), run.end(),
                         [](const Pair& a, const Pair& b) { return a.score > b.score; });
        for (unsigned item = 0; item < run.size(); ++item) {
          const auto at = guard + (panel * 4 + source_number) * input_stride + 8 + 2 * item;
          input[at] = run[item].score;
          input[at + 1] = std::bit_cast<float>(run[item].index);
        }
        combined.insert(combined.end(), run.begin(), run.end());
      }
      // Stable concatenation preserves earlier-list priority and each input's
      // relative order among equal scores; payload value is not a merge key.
      std::stable_sort(combined.begin(), combined.end(),
                       [](const Pair& a, const Pair& b) { return a.score > b.score; });
      for (unsigned item = 0; item < MERGE_COLS / 2; ++item) {
        const auto at = guard + panel * output_stride + 8 + 2 * item;
        expected[at] = combined[item].score;
        expected[at + 1] = std::bit_cast<float>(combined[item].index);
      }
    }
    const auto original = input;
    std::array<Tensor, 2> tensors;
    const uint32_t shapes[2] = {input_count, output_count};
    tensors[0].init_external(input.data(), input.size() * sizeof(float), &shapes[0], 1, DataType::FLOAT32, 0);
    tensors[1].init_external(output.data(), output.size() * sizeof(float), &shapes[1], 1, DataType::FLOAT32,
                             0);
    const std::array<unsigned, 2> order = {MERGE_ARG0, MERGE_ARG1};
    std::array<int64_t, 2> args{};
    for (unsigned i = 0; i < tensors.size(); ++i) {
      tensors[i].start_offset = guard;
    }
    for (unsigned i = 0; i < args.size(); ++i) {
      args[i] = reinterpret_cast<int64_t>(&tensors[order[i]]);
    }
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    if (std::memcmp(input.data(), original.data(), input.size() * sizeof(float))) {
      return 1;
    }
    for (unsigned i = 0; i < output.size(); ++i) {
      if (std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(expected[i])) {
        std::fprintf(stderr, "merge dataset %u element %u mismatch: %08x != %08x\n", dataset, i,
                     std::bit_cast<uint32_t>(output[i]), std::bit_cast<uint32_t>(expected[i]));
        return 2;
      }
    }
  }
  return 0;
}
