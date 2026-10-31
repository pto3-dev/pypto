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

// First-stage reference reads the original [anchor,96] records, never gather tables.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {
struct Pair {
  float score;
  uint32_t index;
};
float Probability(unsigned dataset, unsigned anchor, unsigned cls) {
  switch (dataset) {
    case 0:
      return static_cast<float>((anchor * 17 + cls * 31) % 4096) / 4096.0f;
    case 1:
      return 0.0f;
    case 2:
      return 0.5f;
    case 3:
      return static_cast<float>(anchor) / static_cast<float>(TOPK_ANCHORS);
    case 4:
      return 1.0f - static_cast<float>(anchor) / static_cast<float>(TOPK_ANCHORS);
    case 5:
      return cls == 79 ? 0.75f : 0.25f;
    case 6:
      return anchor >= TOPK_ANCHORS - 300 ? 1.0f : 0.5f;
    default:
      return cls == anchor % 80 ? static_cast<float>(anchor % 19) / 32.0f : 0.0f;
  }
}
bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
}
}  // namespace

int main(int argc, char** argv) {
  constexpr unsigned guard = 16, columns = 608;
  constexpr unsigned table_count = 2592 + ((TOPK_ANCHORS + 31) / 32) * 32;
  constexpr uint32_t sentinel = 0xa55ac33cU;
  if (argc != 2) return 10;
  std::vector<int32_t> metadata(table_count + 2 * guard, std::bit_cast<int32_t>(sentinel));
  std::ifstream table(argv[1], std::ios::binary);
  if (!table.read(reinterpret_cast<char*>(metadata.data() + guard), table_count * sizeof(int32_t))) return 11;
  const auto table_before = metadata;
  for (unsigned dataset = 0; dataset < 8; ++dataset) {
    std::vector<float> dense(TOPK_ANCHORS * 96 + 2 * guard, std::bit_cast<float>(sentinel));
    std::vector<float> output(columns + 2 * guard, std::bit_cast<float>(sentinel));
    auto expected = output;
    // Seed score/payload words match the documented internal -1 fill. Real
    // probabilities always dominate; only the 300-anchor case retains seeds.
    std::vector<Pair> pairs(304, Pair{-1.0f, std::bit_cast<uint32_t>(-1.0f)});
    for (unsigned anchor = 0; anchor < TOPK_ANCHORS; ++anchor) {
      float maximum = -1.0f;
      for (unsigned field = 0; field < 96; ++field) {
        float value = 1000.0f + static_cast<float>(field);
        if (field >= 4 && field < 84) {
          value = Probability(dataset, anchor, field - 4);
          maximum = std::max(maximum, value);
        }
        dense[guard + anchor * 96 + field] = value;
      }
      pairs.push_back({maximum, anchor});
    }
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const Pair& a, const Pair& b) { return a.score > b.score; });
    for (unsigned i = 0; i < 304; ++i) {
      expected[guard + 2 * i] = pairs[i].score;
      expected[guard + 2 * i + 1] = std::bit_cast<float>(pairs[i].index);
    }
    const auto dense_before = dense;
    std::array<Tensor, 3> tensors;
    const uint32_t shapes[] = {table_count, TOPK_ANCHORS * 96, columns};
    tensors[0].init_external(metadata.data(), metadata.size() * sizeof(int32_t), &shapes[0], 1,
                             DataType::INT32, 0);
    tensors[1].init_external(dense.data(), dense.size() * sizeof(float), &shapes[1], 1, DataType::FLOAT32, 0);
    tensors[2].init_external(output.data(), output.size() * sizeof(float), &shapes[2], 1, DataType::FLOAT32,
                             0);
    std::array<int64_t, 3> args{};
    for (unsigned i = 0; i < tensors.size(); ++i) {
      tensors[i].start_offset = guard;
      args[i] = reinterpret_cast<int64_t>(&tensors[i]);
    }
    {
      pto::cpu_sim::ScopedExecutionContext context(0, 0);
      kernel_entry(args.data());
    }
    if (metadata != table_before || !Same(dense, dense_before)) return 1;
    if (!Same(output, expected)) {
      for (unsigned i = 0; i < output.size(); ++i)
        if (std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(expected[i])) {
          std::fprintf(stderr, "anchor-prefix dataset %u word %u: %08x != %08x\n", dataset, i,
                       std::bit_cast<uint32_t>(output[i]), std::bit_cast<uint32_t>(expected[i]));
          break;
        }
      return 2;
    }
  }
  return 0;
}
