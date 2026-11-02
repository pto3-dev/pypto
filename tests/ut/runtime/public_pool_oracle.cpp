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
  constexpr int height = POOL_HEIGHT, width = POOL_WIDTH, channels = POOL_CHANNELS;
  constexpr int ss = POOL_SOURCE_STRIDE, ds = POOL_DESTINATION_STRIDE;
  constexpr int so = POOL_SOURCE_OFFSET, dest = POOL_DESTINATION_OFFSET;
  constexpr int radius = POOL_KERNEL / 2, pixels = height * width;
  for (int dataset : {0, 1, 2}) {
    std::vector<half> source(pixels * ss + 32, half(65504));
    std::vector<half> output(pixels * ds + 32, half(9)), second(48, half(-3));
    std::vector<float> metadata(64, 5.0f);
    for (int pixel = 0; pixel < pixels; ++pixel) {
      for (int channel = 0; channel < channels; ++channel) {
        const int index = 16 + pixel * ss + so + channel;
        if (dataset == 0) {
          source[index] = half(((pixel * 7 + channel * 11) % 53 - 26) / 32.0f);
        } else {
          const unsigned pattern = (static_cast<unsigned>(pixel * channels + channel) * 73 + 11) % 63488;
          const uint16_t bits =
              dataset == 2 ? ((pixel + channel) % 3 == 0 ? 0x8000 : 0)
                           : static_cast<uint16_t>(pattern < 31744 ? pattern : (pattern - 31744) | 0x8000);
          std::memcpy(&source[index], &bits, sizeof(bits));
        }
      }
    }
    const auto sourceBefore = source, secondBefore = second;
    const auto metadataBefore = metadata;
    std::array<Tensor, 4> tensors{};
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
    std::array<unsigned char, sizeof(tensors)> tensorBytes{};
    std::memcpy(tensorBytes.data(), tensors.data(), sizeof(tensors));
    const auto argsBefore = args;
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    // Independent scalar oracle uses the original ky/kx visit order and keeps
    // the first operand on ties, matching the existing finite TMAX contract.
    for (int pixel = 0; pixel < pixels; ++pixel) {
      for (int col = 0; col < ds; ++col) {
        half expected = half(9);
        if (col >= dest && col < dest + channels) {
          expected = half(-65504);
          for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
              const int sy = pixel / width + ky, sx = pixel % width + kx;
              if (sy < 0 || sy >= height || sx < 0 || sx >= width) {
                continue;
              }
              const half value = source[16 + (sy * width + sx) * ss + so + col - dest];
              if (static_cast<float>(expected) < static_cast<float>(value)) {
                expected = value;
              }
            }
          }
        }
        if (std::bit_cast<uint16_t>(output[16 + pixel * ds + col]) != std::bit_cast<uint16_t>(expected)) {
          std::cerr << "dataset=" << dataset << " mismatch=" << pixel << ',' << col << '\n';
          return 1;
        }
      }
    }
    for (int i = 0; i < 16; ++i) {
      if (output[i] != half(9) || output[16 + pixels * ds + i] != half(9)) {
        return 2;
      }
    }
    if (std::memcmp(source.data(), sourceBefore.data(), source.size() * sizeof(half)) ||
        std::memcmp(second.data(), secondBefore.data(), second.size() * sizeof(half)) ||
        std::memcmp(metadata.data(), metadataBefore.data(), metadata.size() * sizeof(float)) ||
        std::memcmp(tensors.data(), tensorBytes.data(), sizeof(tensors)) || args != argsBefore) {
      return 3;
    }
  }
  return 0;
}
