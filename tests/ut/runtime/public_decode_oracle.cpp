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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  constexpr int height = DECODE_HEIGHT, width = DECODE_WIDTH, stride = DECODE_STRIDE;
  constexpr int pixels = height * width, offset = DECODE_OFFSET, total = DECODE_TOTAL;
  constexpr int tableCount = DECODE_TABLE_COUNT, guard = 16;
  constexpr uint32_t sentinel = 0x4b123456;
  if (argc != 2) return 9;
  std::vector<int32_t> table(tableCount);
  std::ifstream file(argv[1], std::ios::binary);
  file.read(reinterpret_cast<char*>(table.data()), table.size() * sizeof(int32_t));
  if (!file || file.peek() != std::char_traits<char>::eof()) return 10;
  for (int dataset : {0, 1}) {
    std::vector<half> boxes(pixels * 16 + 2 * guard, static_cast<half>(7));
    std::vector<half> scores(pixels * 80 + 2 * guard, static_cast<half>(7));
    std::vector<int32_t> metadata(tableCount + 2 * guard, 17);
    std::memcpy(metadata.data() + guard, table.data(), table.size() * sizeof(int32_t));
    std::vector<float> output(total * 96 + 2 * guard, std::bit_cast<float>(sentinel));
    auto fill = [dataset](std::vector<half>& values, int count, unsigned salt) {
      for (int i = 0; i < count; ++i) {
        if (dataset == 0) {
          values[guard + i] =
              static_cast<half>((static_cast<int>((static_cast<unsigned>(i) * 7 + salt) % 127) - 63) / 16.0f);
        } else {
          const unsigned pattern = (static_cast<unsigned>(i) * 73 + salt) % 63488;
          const uint16_t bits = pattern < 31744 ? pattern : (pattern - 31744) | 0x8000;
          values[guard + i] = std::bit_cast<half>(bits);
        }
      }
    };
    fill(boxes, pixels * 16, 11);
    fill(scores, pixels * 80, 211);
    const auto originalBoxes = boxes, originalScores = scores;
    const auto originalMetadata = metadata;
    std::array<Tensor, 4> tensors;
    const uint32_t shapes[4] = {tableCount, pixels * 16, pixels * 80, total * 96};
    void* addresses[4] = {metadata.data(), boxes.data(), scores.data(), output.data()};
    const uint64_t sizes[4] = {metadata.size() * sizeof(int32_t), boxes.size() * sizeof(half),
                               scores.size() * sizeof(half), output.size() * sizeof(float)};
    const DataType types[4] = {DataType::INT32, DataType::FLOAT16, DataType::FLOAT16, DataType::FLOAT32};
    std::array<int64_t, 4> args{};
    for (int i = 0; i < 4; ++i) {
      tensors[i].init_external(addresses[i], sizes[i], &shapes[i], 1, types[i], 0);
      tensors[i].start_offset = guard;
      args[i] = reinterpret_cast<int64_t>(&tensors[i]);
    }
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    if (metadata != originalMetadata ||
        std::memcmp(boxes.data(), originalBoxes.data(), boxes.size() * sizeof(half)) ||
        std::memcmp(scores.data(), originalScores.data(), scores.size() * sizeof(half)))
      return 1;
    for (int i = 0; i < static_cast<int>(output.size()); ++i) {
      uint32_t expected = sentinel;
      if (i >= guard && i < guard + total * 96) {
        const int anchor = (i - guard) / 96, column = (i - guard) % 96;
        if (anchor >= offset && anchor < offset + pixels) {
          const int pixel = anchor - offset;
          float value = 0.0f;
          if (column < 4) {
            const half center = static_cast<half>(column % 2 ? pixel / width + 0.5f : pixel % width + 0.5f);
            const float distance = static_cast<float>(originalBoxes[guard + pixel * 16 + column]);
            const half coordinate = static_cast<half>(column < 2 ? static_cast<float>(center) - distance
                                                                 : static_cast<float>(center) + distance);
            value = static_cast<float>(static_cast<half>(static_cast<float>(coordinate) * stride));
          } else if (column < 84) {
            const float score = static_cast<float>(originalScores[guard + pixel * 80 + column - 4]);
            const float negative = 0.0f - score;
            const float exponential = std::exp(negative);
            const float denominator = exponential + 1.0f;
            value = static_cast<float>(static_cast<half>(1.0f / denominator));
          }
          expected = std::bit_cast<uint32_t>(value);
        }
      }
      if (std::bit_cast<uint32_t>(output[i]) != expected) {
        std::cerr << "dataset=" << dataset << " element=" << i - guard << " actual=" << std::hex
                  << std::bit_cast<uint32_t>(output[i]) << " expected=" << expected << '\n';
        return 2;
      }
    }
  }
  return 0;
}
