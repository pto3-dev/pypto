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

// Appended to unmodified generated Tensor ABI code. The reference indexes
// original NHWC channels only; it does not reuse the phase-pack implementation.
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  constexpr int rows = DP_ROWS, channels = DP_CHANNELS;
  constexpr int columns = (channels + 63) / 64 * 16;
  constexpr int stride = channels + 32, pixels = 256, outputRows = 4 * rows + 32, guard = 16;
  for (int dataset = 0; dataset < 3; ++dataset) {
    std::vector<half> input(pixels * stride + 2 * guard, half(7));
    std::vector<half> weights(columns * columns + 2 * guard, half(11));
    std::vector<float> output(outputRows * columns + 2 * guard, 123.0f);
    auto expected = output;
    for (int i = 0; i < pixels * stride; ++i) {
      const unsigned pattern = (static_cast<unsigned>(i) * 73 + dataset * 7919) % 63488;
      const uint16_t bits = static_cast<uint16_t>(pattern < 31744 ? pattern : (pattern - 31744) | 0x8000);
      input[guard + i] =
          dataset == 2 ? std::bit_cast<half>(bits) : half((i * (dataset ? 17 : 11) % 101 - 50) / 37.0f);
    }
    for (int k = 0; k < columns; ++k)
      for (int n = 0; n < columns; ++n) weights[guard + k * columns + n] = half(k == n ? 1 : 0);
    for (int phase = 0; phase < 4; ++phase)
      for (int row = 0; row < rows; ++row)
        for (int n = 0; n < columns; ++n) {
          float sum = 0;
          // Match the independently defined identity Cube operation, including
          // signed-zero behavior from the full ascending FP32 FMA reduction.
          for (int k = 0; k < columns; ++k) {
            const int channel = k * 4 + phase;
            const float value = channel < channels
                                    ? static_cast<float>(input[guard + (16 + row) * stride + 16 + channel])
                                    : 0.0f;
            sum = std::fma(value, k == n ? 1.0f : 0.0f, sum);
          }
          expected[guard + (16 + phase * rows + row) * columns + n] = sum;
        }
    const auto originalInput = input, originalWeights = weights;
    std::array<Tensor, 3> tensors;
    const uint32_t shapes[][2] = {{pixels, stride}, {columns, columns}, {outputRows, columns}};
    void* addresses[] = {input.data(), weights.data(), output.data()};
    const uint64_t bytes[] = {input.size() * sizeof(half), weights.size() * sizeof(half),
                              output.size() * sizeof(float)};
    std::array<int64_t, 3> args{};
    for (unsigned i = 0; i < tensors.size(); ++i) {
      tensors[i].init_external(addresses[i], bytes[i], shapes[i], 2,
                               i == 2 ? DataType::FLOAT32 : DataType::FLOAT16, 0);
      tensors[i].start_offset = guard;
      args[i] = reinterpret_cast<int64_t>(&tensors[i]);
    }
    {
      pto::cpu_sim::ScopedExecutionContext context(0, 0);
      kernel_entry(args.data());
    }
    if (std::memcmp(input.data(), originalInput.data(), input.size() * sizeof(half)) ||
        std::memcmp(weights.data(), originalWeights.data(), weights.size() * sizeof(half)))
      return 1;
    if (std::memcmp(output.data(), expected.data(), output.size() * sizeof(float))) {
      for (unsigned i = 0; i < output.size(); ++i)
        if (std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(expected[i])) {
          std::fprintf(stderr, "phase input dataset=%d index=%u actual=%08x expected=%08x\n", dataset, i,
                       std::bit_cast<uint32_t>(output[i]), std::bit_cast<uint32_t>(expected[i]));
          break;
        }
      return 2;
    }
  }
  return 0;
}
