/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
  constexpr int pixels = COPY_PIXELS, channels = COPY_CHANNELS;
  constexpr int ss = COPY_SOURCE_STRIDE, ds = COPY_DESTINATION_STRIDE;
  constexpr int so = COPY_SOURCE_OFFSET, dest = COPY_DESTINATION_OFFSET;
  for (int dataset : {0, 1}) {
    std::vector<half> source(pixels * ss + 32, half(7));
    std::vector<half> second(48, half(5)), output(pixels * ds + 32, half(9));
    std::vector<float> metadata(64, 5.0f);
    // Odd multipliers permute every FP16 bit pattern, including NaN payloads,
    // infinities and signed zeros. Reference selection copies words only.
    for (int i = 0; i < pixels * ss; ++i) {
      const auto bits = static_cast<uint16_t>(i * (dataset ? 255u : 73u) + 11u);
      std::memcpy(&source[16 + i], &bits, sizeof(bits));
    }
    const auto source_before = source, second_before = second;
    const auto metadata_before = metadata;
    std::array<Tensor, 4> tensors;
    const uint32_t shapes[4] = {32, pixels * ss, 16, pixels * ds};
    void* addresses[4] = {metadata.data(), source.data(), second.data(), output.data()};
    const uint64_t sizes[4] = {metadata.size() * sizeof(float), source.size() * sizeof(half),
                               second.size() * sizeof(half), output.size() * sizeof(half)};
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
    const auto guard = std::bit_cast<uint16_t>(half(9));
    for (int row = 0; row < pixels; ++row) {
      for (int col = 0; col < ds; ++col) {
        const auto expected = (col >= dest && col < dest + channels)
                                  ? std::bit_cast<uint16_t>(source[16 + row * ss + so + col - dest])
                                  : guard;
        if (std::bit_cast<uint16_t>(output[16 + row * ds + col]) != expected) {
          std::cerr << "dataset=" << dataset << " mismatch=" << row << ',' << col << '\n';
          return 1;
        }
      }
    }
    for (int i = 0; i < 16; ++i) {
      if (std::bit_cast<uint16_t>(output[i]) != guard ||
          std::bit_cast<uint16_t>(output[16 + pixels * ds + i]) != guard)
        return 2;
    }
    if (std::memcmp(source.data(), source_before.data(), source.size() * sizeof(half)) ||
        std::memcmp(second.data(), second_before.data(), second.size() * sizeof(half)) ||
        std::memcmp(metadata.data(), metadata_before.data(), metadata.size() * sizeof(float)))
      return 3;
  }
  return 0;
}
