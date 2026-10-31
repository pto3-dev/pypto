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
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#if CONVERSION_OP == 5
using Input = float;
using Output = half;
#elif CONVERSION_OP == 6
using Input = half;
using Output = float;
#else
using Input = half;
using Output = half;
#endif

namespace {
Input Value(unsigned dataset, unsigned index) {
  if (dataset == 0) return Input((static_cast<int>(index * 31 % 10001) - 5000) / 64.0f);
  const unsigned pattern = index * 73 % 63488;
  const uint16_t bits = static_cast<uint16_t>(pattern < 31744 ? pattern : (pattern - 31744) | 0x8000);
#if CONVERSION_OP == 5
  if (dataset == 1) {
    const uint16_t low = static_cast<uint16_t>((index * 17) % 0x7bfe | ((index & 1) ? 0x8000 : 0));
    const float left = static_cast<float>(std::bit_cast<half>(low));
    const float right = static_cast<float>(std::bit_cast<half>(static_cast<uint16_t>(low + 1)));
    const float midpoint = (left + right) * 0.5f;
    if (index % 3 == 0) return midpoint;
    return std::nextafter(midpoint, index % 3 == 1 ? 0.0f : std::numeric_limits<float>::infinity());
  }
  constexpr std::array<uint32_t, 12> patterns = {0U,          0x80000000U, 0x477fe000U, 0xc77fe000U,
                                                 0x477ff000U, 0xc77ff000U, 0x33800000U, 0xb3800000U,
                                                 0x33000000U, 0xb3000000U, 0x3f801000U, 0xbf801000U};
  return std::bit_cast<float>(patterns[index % patterns.size()]);
#else
  if (dataset == 1) return std::bit_cast<half>(bits);
  constexpr std::array<uint16_t, 12> patterns = {0U,      0x8000U, 1U,      0x8001U, 0x03ffU, 0x83ffU,
                                                 0x0400U, 0x8400U, 0x3c01U, 0xbc01U, 0x7bffU, 0xfbffU};
  return std::bit_cast<half>(patterns[index % patterns.size()]);
#endif
}
}  // namespace

int main() {
  constexpr unsigned length = CONVERSION_LENGTH, guard = 16;
  // Includes the exact production attention scale after FP32 metadata packing.
  for (float scale : {1.0f, 0.1767766952966369f, -0.5f, 0.0f, -0.0f}) {
    for (unsigned dataset = 0; dataset < 3; ++dataset) {
      std::vector<float> metadata(64, 5.0f);
      metadata[guard + 15] = scale;
      std::vector<Input> input(length + 2 * guard, Input(7));
      std::vector<Output> output(length + 2 * guard, Output(9));
      std::vector<half> second(48, half(11));
      auto expected = output;
      for (unsigned index = 0; index < length; ++index) {
        input[guard + index] = Value(dataset, index);
#if CONVERSION_OP == 5
        expected[guard + index] = half(input[guard + index]);
#else
        // Widen and multiply in float before any final half narrowing.
        const float product = static_cast<float>(input[guard + index]) * scale;
        expected[guard + index] = Output(product);
#endif
      }
      const auto metadata_before = metadata;
      const auto input_before = input;
      const auto second_before = second;
      std::array<Tensor, 4> tensors;
      const uint32_t shapes[] = {32, length, 16, length};
      void* addresses[] = {metadata.data(), input.data(), second.data(), output.data()};
      const uint64_t sizes[] = {metadata.size() * sizeof(float), input.size() * sizeof(Input),
                                second.size() * sizeof(half), output.size() * sizeof(Output)};
      const DataType types[] = {DataType::FLOAT32, CONVERSION_OP == 5 ? DataType::FLOAT32 : DataType::FLOAT16,
                                DataType::FLOAT16,
                                CONVERSION_OP == 6 ? DataType::FLOAT32 : DataType::FLOAT16};
      std::array<int64_t, 4> args{};
      for (unsigned index = 0; index < 4; ++index) {
        tensors[index].init_external(addresses[index], sizes[index], &shapes[index], 1, types[index], 0);
        tensors[index].start_offset = guard;
        args[index] = reinterpret_cast<int64_t>(&tensors[index]);
      }
      for (unsigned core = 0; core < get_block_num(); ++core) {
        pto::cpu_sim::ScopedExecutionContext context(core, 0);
        kernel_entry(args.data());
      }
      if (std::memcmp(input.data(), input_before.data(), input.size() * sizeof(Input)) ||
          std::memcmp(metadata.data(), metadata_before.data(), metadata.size() * sizeof(float)) ||
          std::memcmp(second.data(), second_before.data(), second.size() * sizeof(half)))
        return 1;
      if (std::memcmp(output.data(), expected.data(), output.size() * sizeof(Output))) {
        for (unsigned index = 0; index < output.size(); ++index) {
          if (std::memcmp(&output[index], &expected[index], sizeof(Output))) {
            std::fprintf(stderr, "conversion=%u length=%u dataset=%u scale=%a index=%u\n", CONVERSION_OP,
                         length, dataset, scale, index);
            break;
          }
        }
        return 2;
      }
    }
  }
  return 0;
}
