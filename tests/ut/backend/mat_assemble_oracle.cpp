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
#include <vector>

#ifndef AM_REDUCTION_BEGIN
#define AM_REDUCTION_BEGIN 0
#endif
#ifndef AM_REDUCTION_COUNT
#define AM_REDUCTION_COUNT AM_K
#endif

namespace {
constexpr int kGuard = 16;
constexpr int kM = AM_M, kK = AM_K;
constexpr int kReductionBegin = AM_REDUCTION_BEGIN, kReductionCount = AM_REDUCTION_COUNT;
constexpr int kInput = kM * kK, kWeights = kReductionCount * 16, kOutput = kM * 16;
static_assert(kReductionBegin >= 0 && kReductionCount > 0 && kReductionBegin + kReductionCount <= kK);
constexpr int kPatch = AM_PARTIAL ? AM_PR * AM_PC : 0;
constexpr int kArgs = AM_PARTIAL ? 4 : 3;
static_assert(kM > 0 && kK > 0 && kM % 16 == 0 && kK % 16 == 0);
static_assert(!AM_PARTIAL || (AM_PR > 0 && AM_PC > 0 && AM_R0 >= 0 && AM_C0 >= 0 && AM_R0 + AM_PR <= kM &&
                              AM_C0 + AM_PC <= kK));

half sample(int index, int dataset, unsigned multiplier, float divisor) {
  if (dataset == 2) {
    const unsigned value = static_cast<unsigned>(index) * multiplier % 63488;
    return std::bit_cast<half>(static_cast<uint16_t>(value < 31744 ? value : (value - 31744) | 0x8000));
  }
  return half(static_cast<float>((index * static_cast<int>(multiplier)) % 101 - 50) /
              (dataset == 0 ? 16.0f : divisor));
}
}  // namespace

int main() {
  for (int dataset = 0; dataset < 3; ++dataset) {
    std::vector<half> input(kInput + 2 * kGuard, half(7));
    std::vector<half> patch(kPatch + 2 * kGuard, half(13));
    std::vector<half> weights(kWeights + 2 * kGuard, half(11));
    std::vector<float> output(kOutput + 2 * kGuard, 19.0f);
    auto expected = output;
    for (int i = 0; i < kInput; ++i) input[kGuard + i] = sample(i, dataset, 73, 37.0f);
    for (int i = 0; i < kPatch; ++i) patch[kGuard + i] = sample(i, dataset, 41, 29.0f);
    for (int i = 0; i < kWeights; ++i) weights[kGuard + i] = sample(i, dataset, 97, 43.0f);
    const auto originalInput = input, originalPatch = patch, originalWeights = weights;
    for (int row = 0; row < kM; ++row) {
      for (int col = 0; col < 16; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < kReductionCount; ++k) {
          const int inputColumn = kReductionBegin + k;
          const bool inside = AM_PARTIAL && row >= AM_R0 && row < AM_R0 + AM_PR && inputColumn >= AM_C0 &&
                              inputColumn < AM_C0 + AM_PC;
          const half a = inside ? originalPatch[kGuard + (row - AM_R0) * AM_PC + inputColumn - AM_C0]
                                : originalInput[kGuard + row * kK + inputColumn];
          sum = std::fma(static_cast<float>(a), static_cast<float>(originalWeights[kGuard + k * 16 + col]),
                         sum);
        }
        expected[kGuard + row * 16 + col] = sum;
      }
    }
    std::array<Tensor, kArgs> tensors;
    std::array<int64_t, kArgs> args{};
#if AM_PARTIAL
    const uint32_t shapes[kArgs][2] = {{kM, kK}, {AM_PR, AM_PC}, {kReductionCount, 16}, {kM, 16}};
    void* addresses[kArgs] = {input.data(), patch.data(), weights.data(), output.data()};
    const uint64_t sizes[kArgs] = {input.size() * sizeof(half), patch.size() * sizeof(half),
                                   weights.size() * sizeof(half), output.size() * sizeof(float)};
#else
    const uint32_t shapes[kArgs][2] = {{kM, kK}, {kReductionCount, 16}, {kM, 16}};
    void* addresses[kArgs] = {input.data(), weights.data(), output.data()};
    const uint64_t sizes[kArgs] = {input.size() * sizeof(half), weights.size() * sizeof(half),
                                   output.size() * sizeof(float)};
#endif
    for (int i = 0; i < kArgs; ++i) {
      tensors[i].init_external(addresses[i], sizes[i], shapes[i], 2,
                               i == kArgs - 1 ? DataType::FLOAT32 : DataType::FLOAT16, 0);
      tensors[i].start_offset = kGuard;
      args[i] = reinterpret_cast<int64_t>(&tensors[i]);
    }
    pto::cpu_sim::ScopedExecutionContext context(0, 0);
    kernel_entry(args.data());
    if (std::memcmp(input.data(), originalInput.data(), input.size() * sizeof(half)) ||
        std::memcmp(patch.data(), originalPatch.data(), patch.size() * sizeof(half)) ||
        std::memcmp(weights.data(), originalWeights.data(), weights.size() * sizeof(half)))
      return 1;
    if (std::memcmp(output.data(), expected.data(), output.size() * sizeof(float))) {
      for (unsigned i = 0; i < output.size(); ++i) {
        if (std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(expected[i])) {
          std::fprintf(stderr, "assemble dataset=%d index=%u actual=%08x expected=%08x\n", dataset, i,
                       std::bit_cast<uint32_t>(output[i]), std::bit_cast<uint32_t>(expected[i]));
          break;
        }
      }
      return 2;
    }
  }
  return 0;
}
