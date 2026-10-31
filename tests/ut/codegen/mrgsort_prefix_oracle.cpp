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
// Appended to unmodified generated code; the public kernel has two input lists.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr unsigned kGuard = 16;
constexpr uint32_t kSentinel = 0xa55ac33cU;
struct Pair {
  float score;
  uint32_t payload;
};

float Score(unsigned dataset, unsigned source, unsigned pair_index) {
  constexpr std::array<uint32_t, 8> patterns = {0U,          0x80000000U, 0x7f800000U, 0xff800000U,
                                                0x3f800000U, 0xbf800000U, 1U,          0x80000001U};
  switch (dataset) {
    case 0:
      return static_cast<float>((pair_index * 17 + source * 13) % 97) - 48.0f;
    case 1:
      return 0.5f;
    case 2:
      return std::bit_cast<float>(patterns[(pair_index + source) % patterns.size()]);
    case 3:
      return static_cast<float>(10000 * source + pair_index);
    case 4:
      return static_cast<float>(10000 * (2 - source) + pair_index);
    default:
      return static_cast<float>((pair_index / 3 + source) % 7);
  }
}

bool EqualBits(const std::vector<float>& actual, const std::vector<float>& expected) {
  for (unsigned i = 0; i < actual.size(); ++i) {
    if (std::bit_cast<uint32_t>(actual[i]) != std::bit_cast<uint32_t>(expected[i])) {
      std::fprintf(stderr, "merge prefix bit mismatch at element %u\n", i);
      return false;
    }
  }
  return true;
}

bool CheckDataset(unsigned dataset) {
  std::array<std::vector<float>, 2> inputs;
  std::vector<Pair> all;
  for (unsigned source = 0; source < 2; ++source) {
    const unsigned columns = source == 0 ? 64 : 128;
    inputs[source].assign(columns + 2 * kGuard, std::bit_cast<float>(kSentinel));
    std::vector<Pair> list;
    for (unsigned pair_index = 0; pair_index < columns / 2; ++pair_index) {
      list.push_back(
          {Score(dataset, source, pair_index), 0x7fc00000U | (source * 65537U + pair_index * 257U)});
    }
    std::stable_sort(list.begin(), list.end(),
                     [](const Pair& a, const Pair& b) { return a.score > b.score; });
    for (unsigned pair_index = 0; pair_index < list.size(); ++pair_index) {
      inputs[source][kGuard + 2 * pair_index] = list[pair_index].score;
      inputs[source][kGuard + 2 * pair_index + 1] = std::bit_cast<float>(list[pair_index].payload);
    }
    all.insert(all.end(), list.begin(), list.end());
  }
  const auto original = inputs;
  std::stable_sort(all.begin(), all.end(), [](const Pair& a, const Pair& b) { return a.score > b.score; });
  std::vector<float> output(128 + 2 * kGuard, std::bit_cast<float>(kSentinel));
  auto expected = output;
  for (unsigned pair_index = 0; pair_index < 64; ++pair_index) {
    expected[kGuard + 2 * pair_index] = all[pair_index].score;
    expected[kGuard + 2 * pair_index + 1] = std::bit_cast<float>(all[pair_index].payload);
  }
  kernel(inputs[0].data() + kGuard, inputs[1].data() + kGuard, output.data() + kGuard);
  return EqualBits(output, expected) && EqualBits(inputs[0], original[0]) &&
         EqualBits(inputs[1], original[1]);
}
}  // namespace

int main() {
  for (unsigned dataset = 0; dataset < 6; ++dataset) {
    if (!CheckDataset(dataset)) {
      return 1;
    }
  }
  std::puts("6 datasets: 384 merge pairs bitwise; inputs and guards unchanged");
  return 0;
}
