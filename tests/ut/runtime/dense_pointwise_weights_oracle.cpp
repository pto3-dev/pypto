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

// Included before generated code to observe real CPU TLOADs, then appended
// again for the independent original-channel FP32 reference and Tensor ABI.
#ifdef DP_PROBE_ONLY
#include <cstdint>
#include <cstdlib>
#include <pto/pto-inst.hpp>

namespace dense_weight_probe {
struct Payload {
  uintptr_t begin = 0;
  uintptr_t end = 0;
  uint64_t bytes = 0;
  uint64_t calls = 0;
};
Payload input, weights;

template <typename Global>
void account(Global& source, Payload& payload) {
  const auto address = reinterpret_cast<uintptr_t>(source.data());
  if (address < payload.begin || address >= payload.end) return;
  uint64_t elements = 1, span = 1;
  for (int dim = 0; dim < pto::GlobalTensorDim::TOTAL_DIM; ++dim) {
    const auto extent = source.GetShape(dim), stride = source.GetStride(dim);
    if (extent <= 0 || stride < 0) std::abort();
    elements *= extent;
    span += (extent - 1) * stride;
  }
  constexpr auto elementBytes = sizeof(typename Global::RawDType);
  if (span * elementBytes > payload.end - address) std::abort();
  payload.bytes += elements * elementBytes;
  ++payload.calls;
}

template <typename Tile, typename Global>
auto load(Tile& destination, Global& source) {
  account(source, input);
  account(source, weights);
  return pto::TLOAD(destination, source);
}
}  // namespace dense_weight_probe
#define TLOAD(...) dense_weight_probe::load(__VA_ARGS__)
#else
#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {
std::vector<half> Read(const char* path, size_t count) {
  std::vector<half> values(count);
  std::ifstream stream(path, std::ios::binary);
  if (!stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(half)) ||
      stream.peek() != std::ifstream::traits_type::eof())
    std::abort();
  return values;
}
}  // namespace

// Existing direct-load cases retain their exact four-request contract. New
// staged cases supply only the expected request count; all original CHW FMA,
// payload, input immutability and output guard checks remain identical.
#ifndef DW_WEIGHT_LOAD_CALLS
#define DW_WEIGHT_LOAD_CALLS 4
#endif

int main(int argc, char** argv) {
  if (argc != 4) return 3;
  constexpr int rows = DW_ROWS, channels = DW_CHANNELS, outputs = DW_OUTPUTS;
  constexpr int columns = DW_COLS, first = DW_FIRST, phaseRows = (channels + 63) / 64 * 16;
  constexpr int stride = channels + 32, pixels = 1024, co = (outputs + 15) / 16 * 16;
  constexpr int inputCount = pixels * stride, packedCount = 4 * phaseRows * co, rawCount = outputs * channels;
  constexpr int outputCount = (8 * rows + 32) * columns, guard = 16;
  const auto images = Read(argv[1], 3 * inputCount), packed = Read(argv[2], 3 * packedCount);
  const auto original = Read(argv[3], 3 * rawCount);
  for (int dataset = 0; dataset < 3; ++dataset) {
    std::vector<half> input(inputCount + 2 * guard, half(7)), weight(packedCount + 2 * guard, half(11));
    std::copy_n(images.data() + dataset * inputCount, inputCount, input.data() + guard);
    std::copy_n(packed.data() + dataset * packedCount, packedCount, weight.data() + guard);
    std::vector<float> output(outputCount + 2 * guard, 123.0f);
    auto expected = output;
    for (int window = 0; window < 2; ++window)
      for (int phase = 0; phase < 4; ++phase)
        for (int row = 0; row < rows; ++row)
          for (int n = 0; n < columns; ++n) {
            float sum = 0;
            // Original CHW channel order, never the packed phase indices.
            if (first + n < outputs) {
              for (int channel = phase; channel < channels; channel += 4) {
                const float x =
                    static_cast<float>(input[guard + (16 + window * rows + row) * stride + 16 + channel]);
                const float w =
                    static_cast<float>(original[dataset * rawCount + (first + n) * channels + channel]);
                sum = std::fma(x, w, sum);
              }
            }
            expected[guard + (16 + (window * 4 + phase) * rows + row) * columns + n] = sum;
          }
    const auto inputBefore = input, weightBefore = weight;
    std::array<Tensor, 3> tensors;
    const uint32_t shapes[][2] = {{pixels, stride}, {4 * phaseRows, co}, {8 * rows + 32, columns}};
    void* addresses[] = {input.data(), weight.data(), output.data()};
    const uint64_t sizes[] = {input.size() * sizeof(half), weight.size() * sizeof(half),
                              output.size() * sizeof(float)};
    std::array<int64_t, 3> args{};
    for (unsigned i = 0; i < tensors.size(); ++i) {
      tensors[i].init_external(addresses[i], sizes[i], shapes[i], 2,
                               i == 2 ? DataType::FLOAT32 : DataType::FLOAT16, 0);
      tensors[i].start_offset = guard;
      args[i] = reinterpret_cast<int64_t>(&tensors[i]);
    }
    using dense_weight_probe::Payload;
    dense_weight_probe::input = Payload{reinterpret_cast<uintptr_t>(input.data() + guard),
                                        reinterpret_cast<uintptr_t>(input.data() + guard + inputCount), 0, 0};
    dense_weight_probe::weights =
        Payload{reinterpret_cast<uintptr_t>(weight.data() + guard),
                reinterpret_cast<uintptr_t>(weight.data() + guard + packedCount), 0, 0};
    {
      pto::cpu_sim::ScopedExecutionContext context(0, 0);
      kernel_entry(args.data());
    }
    if (std::memcmp(input.data(), inputBefore.data(), input.size() * sizeof(half)) ||
        std::memcmp(weight.data(), weightBefore.data(), weight.size() * sizeof(half)))
      return 1;
    if (std::memcmp(output.data(), expected.data(), output.size() * sizeof(float))) {
      for (unsigned i = 0; i < output.size(); ++i)
        if (std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(expected[i])) {
          std::fprintf(stderr, "dense weights dataset=%d index=%u actual=%08x expected=%08x\n", dataset, i,
                       std::bit_cast<uint32_t>(output[i]), std::bit_cast<uint32_t>(expected[i]));
          break;
        }
      return 2;
    }
    const auto& a = dense_weight_probe::input;
    const auto& b = dense_weight_probe::weights;
    if (a.bytes != 2 * rows * channels * sizeof(half) || b.bytes != 4 * phaseRows * columns * sizeof(half) ||
        a.calls != 2 * ((channels + 63) / 64) || b.calls != DW_WEIGHT_LOAD_CALLS)
      return 4;
    std::printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", a.bytes, b.bytes, a.calls, b.calls);
  }
  return 0;
}
#endif
