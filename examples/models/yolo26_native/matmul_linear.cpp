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

using H64 = pto::Tile<pto::TileType::Vec, half, 1, 64>;
using H256 = pto::Tile<pto::TileType::Vec, half, 1, 256>;
using F64 = pto::Tile<pto::TileType::Vec, float, 1, 64>;
using F256 = pto::Tile<pto::TileType::Vec, float, 1, 256>;

AICORE inline void load_rows(__ubuf__ half* dst, __gm__ half* src, int rows, int stride) {
  if (stride / 16 - 1 <= UINT16_MAX) {
    copy_gm_to_ubuf(dst, src, 0, rows, 1, stride / 16 - 1, 0);
  } else {
    for (int row = 0; row < rows; ++row) {
      copy_gm_to_ubuf(dst + row * 16, src + static_cast<int64_t>(row) * stride, 0, 1, 1, 0, 0);
    }
  }
}

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
  H64 ha, rounded;
  H256 hb, transposed;
  F64 fa, input, weight, product, acc;
  F256 fb;
  pto::TASSIGN<0>(ha);
  pto::TASSIGN<1024>(hb);
  pto::TASSIGN<2048>(transposed);
  pto::TASSIGN<4096>(fa);
  pto::TASSIGN<8192>(fb);
  pto::TASSIGN<12288>(input);
  pto::TASSIGN<13312>(weight);
  pto::TASSIGN<14336>(product);
  pto::TASSIGN<15360>(acc);
  pto::TASSIGN<16384>(rounded);
  uint64_t source_rows[16], destination_rows[16];
  if (transpose_b) {
    for (int row = 0; row < 16; ++row) {
      source_rows[row] = reinterpret_cast<uint64_t>(hb.data() + row * 16);
      destination_rows[row] = reinterpret_cast<uint64_t>(transposed.data() + row * 16);
    }
    set_va_reg_sb(VA2, source_rows);
    set_va_reg_sb(VA3, source_rows + 8);
    set_va_reg_sb(VA0, destination_rows);
    set_va_reg_sb(VA1, destination_rows + 8);
  }
  const int rows = m * (n / 16) < 4 * get_block_num() ? 2 : 4;
  for (int work = get_block_idx(); work < (m / rows) * (n / 16); work += get_block_num()) {
    const int row = work / (n / 16) * rows, col = work % (n / 16) * 16;
    pto::TEXPANDS(acc, 0.0f);
    // Inactive rows stay zero when a small grid selects two live rows.
    pto::TEXPANDS(ha, static_cast<half>(0));
    for (int kk = 0; kk < k; kk += 16) {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      load_rows(ha.data(), ga + row * k + kk, rows, k);
      auto* source = transpose_b ? gb + col * k + kk : gb + kk * n + col;
      load_rows(hb.data(), source, 16, transpose_b ? k : n);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pto::TCVT(fa, ha, pto::RoundMode::CAST_NONE);
      if (transpose_b) {
        scatter_vnchwconv_b16(VA0, VA2, 1, 0, 0);
        pipe_barrier(PIPE_V);
        pto::TCVT(fb, transposed, pto::RoundMode::CAST_NONE);
      } else {
        pto::TCVT(fb, hb, pto::RoundMode::CAST_NONE);
      }
      set_flag(PIPE_V, PIPE_S, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
      for (int ki = 0; ki < 16; ++ki) {
        const float value0 = fa.data()[ki], value1 = fa.data()[16 + ki];
        const float value2 = fa.data()[32 + ki], value3 = fa.data()[48 + ki];
        set_flag(PIPE_S, PIPE_V, EVENT_ID6);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID6);
        set_mask_norm();
        set_vector_mask(0, UINT64_C(0xffff));
        vector_dup(input.data(), static_cast<float>(value0), 1, 1, 1, 8, 8);
        vector_dup(input.data() + 16, static_cast<float>(value1), 1, 1, 1, 8, 8);
        vector_dup(input.data() + 32, static_cast<float>(value2), 1, 1, 1, 8, 8);
        vector_dup(input.data() + 48, static_cast<float>(value3), 1, 1, 1, 8, 8);
        vmax(weight.data(), fb.data() + ki * 16, fb.data() + ki * 16, 4, 1, 1, 1, 2, 0, 0);
        set_vector_mask(UINT64_MAX, UINT64_MAX);
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
    if (n / 16 - 1 <= UINT16_MAX) {
      copy_ubuf_to_gm(go + row * n + col, rounded.data(), 0, rows, 1, 0, n / 16 - 1);
    } else {
      for (int q = 0; q < rows; ++q) {
        copy_ubuf_to_gm(go + (row + q) * n + col, rounded.data() + q * 16, 0, 1, 1, 0, 0);
      }
    }
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}
