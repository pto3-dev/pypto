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
#include <iostream>
#include <vector>

int main() {
  constexpr int pixels = ADD_PIXELS, channels = ADD_CHANNELS;
  constexpr int ss = ADD_SOURCE_STRIDE, bs = ADD_SECOND_STRIDE, ds = ADD_DESTINATION_STRIDE;
  constexpr int so = ADD_SOURCE_OFFSET, bo = ADD_SECOND_OFFSET, dest = ADD_DESTINATION_OFFSET;
  for (int dataset : {0, 1}) {
    std::vector<half> left(pixels * ss + 32, half(7)), right(pixels * bs + 32, half(7));
    std::vector<half> output(pixels * ds + 32, half(9));
    std::vector<float> metadata(64, 5.0f);
    auto fill = [dataset](std::vector<half>& values, int count, int salt) {
      for (int i = 0; i < count; ++i) {
        if (dataset == 0) {
          values[16 + i] = half(((i * 7 + salt) % 53 - 26) / 32.0f);
        } else {
          const unsigned pattern = (static_cast<unsigned>(i) * 73 + salt) % 63488;
          const uint16_t bits = static_cast<uint16_t>(pattern < 31744 ? pattern : (pattern - 31744) | 0x8000);
          std::memcpy(&values[16 + i], &bits, sizeof(bits));
        }
      }
    };
    fill(left, pixels * ss, 11);
    fill(right, pixels * bs, 211);
    // Exercise both zero signs through the real wrapper, in addition to the
    // all-finite-bit-pattern cycle in the longer profile.
    for (int c = 0; c < 4; ++c) {
      const uint16_t a = (c & 1) ? 0x8000 : 0, b = (c & 2) ? 0x8000 : 0;
      std::memcpy(&left[16 + so + c], &a, sizeof(a));
      std::memcpy(&right[16 + bo + c], &b, sizeof(b));
    }
    const auto leftBefore = left, rightBefore = right;
    const auto metadataBefore = metadata;
    std::array<Tensor, 4> tensors;
    const uint32_t shapes[4] = {32, pixels * ss, pixels * bs, pixels * ds};
    void* addresses[4] = {metadata.data(), left.data(), right.data(), output.data()};
    const uint64_t sizes[4] = {metadata.size() * sizeof(float), left.size() * sizeof(half),
                               right.size() * sizeof(half), output.size() * sizeof(half)};
    std::array<int64_t, 4> args{};
    for (int i = 0; i < 4; ++i) {
      tensors[i].init_external(addresses[i], sizes[i], &shapes[i], 1,
                               i == 0 ? DataType::FLOAT32 : DataType::FLOAT16, 0);
      tensors[i].start_offset = 16;
      args[i] = reinterpret_cast<int64_t>(&tensors[i]);
    }
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    for (int row = 0; row < pixels; ++row) {
      for (int col = 0; col < ds; ++col) {
        half expected = half(9);
        if (col >= dest && col < dest + channels) {
          expected = half(static_cast<float>(left[16 + row * ss + so + col - dest]) +
                          static_cast<float>(right[16 + row * bs + bo + col - dest]));
        }
        if (std::bit_cast<uint16_t>(output[16 + row * ds + col]) != std::bit_cast<uint16_t>(expected)) {
          std::cerr << "dataset=" << dataset << " mismatch=" << row << ',' << col << '\n';
          return 1;
        }
      }
    }
    for (int i = 0; i < 16; ++i) {
      if (output[i] != half(9) || output[16 + pixels * ds + i] != half(9)) return 2;
    }
    if (std::memcmp(left.data(), leftBefore.data(), left.size() * sizeof(half)) ||
        std::memcmp(right.data(), rightBefore.data(), right.size() * sizeof(half)) ||
        std::memcmp(metadata.data(), metadataBefore.data(), metadata.size() * sizeof(float)))
      return 3;
  }
  return 0;
}
