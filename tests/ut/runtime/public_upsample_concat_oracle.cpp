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
  constexpr int ih = UC_H, iw = UC_W, ca = UC_CA, cb = UC_CB;
  constexpr int sa = UC_SA, sb = UC_SB, oa = UC_OA, ob = UC_OB;
  constexpr int oh = 2 * ih, ow = 2 * iw, co = ca + cb;
  constexpr int na = ih * iw * sa, nb = oh * ow * sb, no = oh * ow * co;
  constexpr int ac = UC_FIRST ? 0 : cb, bc = UC_FIRST ? ca : 0;
  for (int dataset : {0, 1}) {
    std::vector<half> low(na + 32, half(7)), skip(nb + 32, half(7)), output(no + 32, half(9));
    std::vector<float> metadata(64, 5.0f);
    auto fill = [dataset](std::vector<half>& values, int count, unsigned salt) {
      for (int i = 0; i < count; ++i) {
        const uint16_t bits =
            static_cast<uint16_t>((static_cast<unsigned>(i) * (dataset == 0 ? 73 : 127) + salt) & 65535);
        std::memcpy(&values[16 + i], &bits, sizeof(bits));
      }
    };
    fill(low, na, 11);
    fill(skip, nb, 353);
    const auto lowBefore = low, skipBefore = skip;
    const auto metadataBefore = metadata;
    std::array<Tensor, 4> tensors;
    const uint32_t shapes[4] = {32, na, nb, no};
    void* addresses[4] = {metadata.data(), low.data(), skip.data(), output.data()};
    const uint64_t sizes[4] = {metadata.size() * sizeof(float), low.size() * sizeof(half),
                               skip.size() * sizeof(half), output.size() * sizeof(half)};
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
    for (int y = 0; y < oh; ++y) {
      for (int x = 0; x < ow; ++x) {
        for (int c = 0; c < co; ++c) {
          const bool fromLow = c >= ac && c < ac + ca;
          const half expected = fromLow ? lowBefore[16 + (y / 2 * iw + x / 2) * sa + oa + c - ac]
                                        : skipBefore[16 + (y * ow + x) * sb + ob + c - bc];
          if (std::bit_cast<uint16_t>(output[16 + (y * ow + x) * co + c]) !=
              std::bit_cast<uint16_t>(expected)) {
            std::cerr << "dataset=" << dataset << " mismatch=" << y << ',' << x << ',' << c << '\n';
            return 1;
          }
        }
      }
    }
    for (int i = 0; i < 16; ++i) {
      if (output[i] != half(9) || output[16 + no + i] != half(9)) return 2;
    }
    if (std::memcmp(low.data(), lowBefore.data(), low.size() * sizeof(half)) ||
        std::memcmp(skip.data(), skipBefore.data(), skip.size() * sizeof(half)) ||
        std::memcmp(metadata.data(), metadataBefore.data(), metadata.size() * sizeof(float)))
      return 3;
  }
  return 0;
}
