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
#include <iostream>
#include <vector>

int main() {
  constexpr int m = MC_M, k = MC_K, pn = MC_PN, n = MC_N, col = MC_COL, rounds = MC_ROUNDS;
  constexpr int na = m * (MC_LEFT ? pn : k), nb = rounds * k * (MC_LEFT ? n : pn), no = rounds * m * n;
  for (int dataset : {0, 1}) {
    std::vector<half> a(na + 32, half(7)), b(nb + 32, half(7));
    std::vector<float> output(no + 32, 19.0f);
    // Dyadic data keeps every exact dense sum representable in FP32; this
    // tests address/operand semantics independently of Cube reduction trees.
    for (int i = 0; i < na; ++i) a[16 + i] = half(((i * 13 + dataset * 7) % 31 - 15) / 16.0f);
    for (int i = 0; i < nb; ++i) b[16 + i] = half(((i * 11 + dataset * 13) % 23 - 11) / 32.0f);
    const auto oldA = a, oldB = b;
    std::array<Tensor, 3> tensors;
    const uint32_t shapes[3][2] = {{m, MC_LEFT ? pn : k}, {rounds * k, MC_LEFT ? n : pn}, {rounds * m, n}};
    void* addresses[3] = {a.data(), b.data(), output.data()};
    const uint64_t sizes[3] = {a.size() * sizeof(half), b.size() * sizeof(half),
                               output.size() * sizeof(float)};
    std::array<int64_t, 3> args{};
    for (int i = 0; i < 3; ++i) {
      tensors[i].init_external(addresses[i], sizes[i], shapes[i], 2,
                               i == 2 ? DataType::FLOAT32 : DataType::FLOAT16, 0);
      tensors[i].start_offset = 16;
      args[i] = reinterpret_cast<int64_t>(&tensors[i]);
    }
    for (unsigned core = 0; core < get_block_num(); ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      kernel_entry(args.data());
    }
    for (int iteration = 0; iteration < rounds; ++iteration)
      for (int row = 0; row < m; ++row)
        for (int c = 0; c < n; ++c) {
          float expected = 0.0f;
          for (int term = 0; term < k; ++term)
            expected = std::fma(
                static_cast<float>(oldA[16 + row * (MC_LEFT ? pn : k) + (MC_LEFT ? col : 0) + term]),
                static_cast<float>(
                    oldB[16 + (iteration * k + term) * (MC_LEFT ? n : pn) + (MC_LEFT ? 0 : col) + c]),
                expected);
          const int index = (iteration * m + row) * n + c;
          if (std::bit_cast<uint32_t>(expected) != std::bit_cast<uint32_t>(output[16 + index])) {
            std::cerr << "dataset=" << dataset << " mismatch=" << index << '\n';
            return 1;
          }
        }
    for (int i = 0; i < 16; ++i)
      if (output[i] != 19.0f || output[16 + no + i] != 19.0f) return 2;
    if (std::memcmp(a.data(), oldA.data(), a.size() * sizeof(half)) ||
        std::memcmp(b.data(), oldB.data(), b.size() * sizeof(half)))
      return 3;
  }
  return 0;
}
