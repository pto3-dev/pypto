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
#include <vector>

int main() {
  using Data = DMA_CTYPE;
  constexpr size_t rows = DMA_ROWS, columns = DMA_COLUMNS;
  constexpr size_t ss = DMA_SOURCE_STRIDE, ds = DMA_DESTINATION_STRIDE;
  constexpr size_t so = DMA_SOURCE_COLUMN, dest = DMA_DESTINATION_COLUMN, guard = 16;
  constexpr uint32_t sentinel = 0xa55ac33cU;
  constexpr std::array<uint32_t, 8> edges = {0U,          1U,          0x7fffffffU, 0x80000000U,
                                             0xffffffffU, 0x01020304U, 0x7f800000U, 0x7fc01234U};
  for (uint32_t dataset : {0U, 1U}) {
    std::vector<Data> source(rows * ss + 2 * guard);
    std::vector<Data> output(rows * ds + 2 * guard, std::bit_cast<Data>(sentinel));
    for (size_t i = 0; i < source.size(); ++i)
      source[i] = std::bit_cast<Data>(static_cast<uint32_t>(i) * 1664525U + 1013904223U + dataset);
    for (size_t i = 0; i < edges.size(); ++i)
      source[guard + so + i] = std::bit_cast<Data>(edges[(i + dataset) % edges.size()]);
    const auto original = source;
    std::array<Tensor, 2> tensors;
    const uint32_t shapes[2] = {rows * ss, rows * ds};
    tensors[0].init_external(source.data(), source.size() * sizeof(Data), &shapes[0], 1, DMA_DTYPE, 0);
    tensors[1].init_external(output.data(), output.size() * sizeof(Data), &shapes[1], 1, DMA_DTYPE, 0);
    std::array<int64_t, 2> args{};
    const std::array<size_t, 2> order = {DMA_ARG0, DMA_ARG1};
    for (size_t i = 0; i < tensors.size(); ++i) {
      tensors[i].start_offset = guard;
      args[i] = reinterpret_cast<int64_t>(&tensors[order[i]]);
    }
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    // Storage preserves raw bytes, including FP32 NaN payloads.
    if (std::memcmp(source.data(), original.data(), source.size() * sizeof(Data))) return 1;
    for (size_t i = 0; i < output.size(); ++i) {
      uint32_t expected = sentinel;
      if (i >= guard && i < output.size() - guard) {
        const size_t index = i - guard, row = index / ds, col = index % ds;
        if (col >= dest && col < dest + columns)
          expected = std::bit_cast<uint32_t>(original[guard + row * ss + so + col - dest]);
      }
      if (std::bit_cast<uint32_t>(output[i]) != expected) return 2;
    }
  }
  return 0;
}
