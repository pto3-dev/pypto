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
#include <pto/pto-inst.hpp>

#include "tensor.h"  // NOLINT(build/include_subdir): runtime platform ABI include root.

using H16 = pto::Tile<pto::TileType::Vec, half, 1, 16>;
using H256 = pto::Tile<pto::TileType::Vec, half, 1, 256>;
using F16 = pto::Tile<pto::TileType::Vec, float, 1, 16>;
using F256 = pto::Tile<pto::TileType::Vec, float, 1, 256>;

extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* pa = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* pb = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* ga = reinterpret_cast<__gm__ half*>(pa->buffer.addr);
  auto* gb = reinterpret_cast<__gm__ half*>(pb->buffer.addr);
  auto* go = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  auto* meta = reinterpret_cast<__ubuf__ float*>(32768);
  copy_gm_to_ubuf(meta, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  const int m = meta[0], n = meta[1], k = meta[2], transpose_b = meta[3];
  H16 ha, rounded;
  H256 hb;
  F16 fa, input, weight, product, acc;
  F256 fb;
  pto::TASSIGN<0>(ha);
  pto::TASSIGN<1024>(hb);
  pto::TASSIGN<2048>(fa);
  pto::TASSIGN<4096>(fb);
  pto::TASSIGN<8192>(input);
  pto::TASSIGN<9216>(weight);
  pto::TASSIGN<10240>(product);
  pto::TASSIGN<11264>(acc);
  pto::TASSIGN<12288>(rounded);
  for (int work = get_block_idx(); work < m * (n / 16); work += get_block_num()) {
    const int row = work / (n / 16), col = work % (n / 16) * 16;
    pto::TEXPANDS(acc, 0.0f);
    for (int kk = 0; kk < k; kk += 16) {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      copy_gm_to_ubuf(ha.data(), ga + row * k + kk, 0, 1, 1, 0, 0);
      for (int lane = 0; lane < 16; ++lane) {
        auto* src = transpose_b ? gb + (col + lane) * k + kk : gb + (kk + lane) * n + col;
        copy_gm_to_ubuf(hb.data() + lane * 16, src, 0, 1, 1, 0, 0);
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pto::TCVT(fa, ha, pto::RoundMode::CAST_NONE);
      pto::TCVT(fb, hb, pto::RoundMode::CAST_NONE);
      set_flag(PIPE_V, PIPE_S, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
      for (int ki = 0; ki < 16; ++ki) {
        const float scalar = fa.data()[ki];
        for (int lane = 0; lane < 16; ++lane)
          weight.data()[lane] = fb.data()[transpose_b ? lane * 16 + ki : ki * 16 + lane];
        set_flag(PIPE_S, PIPE_V, EVENT_ID6);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID6);
        pto::TEXPANDS(input, static_cast<float>(scalar));
        pipe_barrier(PIPE_V);
        pto::TMUL(product, input, weight);
        pipe_barrier(PIPE_V);
        pto::TADD(acc, acc, product);
        set_flag(PIPE_V, PIPE_S, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
      }
    }
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, acc, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(go + row * n + col, rounded.data(), 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}
