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
using Mat =
    pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::RowMajor, 512>;
using Left =
    pto::Tile<pto::TileType::Left, half, 16, 16, pto::BLayout::ColMajor, 16, 16, pto::SLayout::RowMajor, 512>;
using Right = pto::Tile<pto::TileType::Right, half, 16, 16, pto::BLayout::RowMajor, 16, 16,
                        pto::SLayout::ColMajor, 512>;
using Acc = pto::Tile<pto::TileType::Acc, float, 16, 16, pto::BLayout::ColMajor, 16, 16,
                      pto::SLayout::RowMajor, 1024>;
using Vec = pto::Tile<pto::TileType::Vec, float, 16, 16>;
using HVec = pto::Tile<pto::TileType::Vec, half, 16, 16>;
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
  Mat ma, mb;
  Left a;
  Right b;
  Acc acc;
  Vec value;
  HVec rounded;
  pto::TASSIGN<0>(ma);
  pto::TASSIGN<512>(mb);
  pto::TASSIGN<0>(a);
  pto::TASSIGN<0>(b);
  pto::TASSIGN<0>(acc);
  pto::TASSIGN<0>(value);
  pto::TASSIGN<4096>(rounded);
  for (int work = get_block_idx(); work < (m / 16) * (n / 16); work += get_block_num()) {
    const int mr = work / (n / 16) * 16, nc = work % (n / 16) * 16;
    for (int kk = 0; kk < k; kk += 16) {
      for (int row = 0; row < 16; ++row) {
        copy_gm_to_cbuf(ma.data() + row * 16, ga + (mr + row) * k + kk, 0, 1, 1, 0, 0, (pad_t)0);
        auto* source = transpose_b ? gb + (nc + row) * k + kk : gb + (kk + row) * n + nc;
        copy_gm_to_cbuf(mb.data() + row * 16, source, 0, 1, 1, 0, 0, (pad_t)0);
      }
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      pto::TMOV(a, ma);
      if (transpose_b)
        load_cbuf_to_cb(b.data(), mb.data(), 0, 1, 1, 0, 0, false, addr_cal_mode_t(0));
      else
        pto::TMOV(b, mb);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (kk == 0)
        pto::TMATMUL(acc, a, b);
      else
        pto::TMATMUL_ACC(acc, acc, a, b);
      set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(value, acc);
    // Release L0C only after the vector-side accumulator read completes.
    set_flag(PIPE_V, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, value, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(go + mr * n + nc, rounded.data(), 0, 16, 1, 0, n / 16 - 1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}
