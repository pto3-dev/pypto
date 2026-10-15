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
using H = pto::Tile<pto::TileType::Vec, half, 1, 16>;
using F = pto::Tile<pto::TileType::Vec, float, 1, 16>;
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* px = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* pw = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* pb = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* x = reinterpret_cast<__gm__ half*>(px->buffer.addr);
  auto* weight = reinterpret_cast<__gm__ half*>(pw->buffer.addr);
  auto* bias = reinterpret_cast<__gm__ float*>(pb->buffer.addr);
  auto* out = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  auto* m = reinterpret_cast<__ubuf__ float*>(32768);
  copy_gm_to_ubuf(m, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  const int ih = m[0], iw = m[1], channels = m[2], stride = m[3], oh = m[4], ow = m[5], activation = m[7];
  H hs, hw, ho;
  F a, w, acc, scratch, ones;
  pto::TASSIGN<0>(hs);
  pto::TASSIGN<1024>(hw);
  pto::TASSIGN<2048>(ho);
  pto::TASSIGN<4096>(a);
  pto::TASSIGN<8192>(w);
  pto::TASSIGN<12288>(acc);
  pto::TASSIGN<16384>(scratch);
  pto::TASSIGN<20480>(ones);
  for (int work = get_block_idx(); work < oh * ow * (channels / 16); work += get_block_num()) {
    const int pixel = work / (channels / 16), cb = work % (channels / 16) * 16;
    pto::TEXPANDS(acc, 0.0f);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int ky = 0; ky < 3; ++ky)
      for (int kx = 0; kx < 3; ++kx) {
        const int sy = pixel / ow * stride + ky - 1, sx = pixel % ow * stride + kx - 1;
        if (sy < 0 || sy >= ih || sx < 0 || sx >= iw) continue;
        copy_gm_to_ubuf(hs.data(), x + (sy * iw + sx) * channels + cb, 0, 1, 1, 0, 0);
        copy_gm_to_ubuf(hw.data(), weight + (ky * 3 + kx) * channels + cb, 0, 1, 1, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TCVT(a, hs, pto::RoundMode::CAST_NONE);
        pto::TCVT(w, hw, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        pto::TMUL(a, a, w);
        pipe_barrier(PIPE_V);
        pto::TADD(acc, acc, a);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      }
    copy_gm_to_ubuf(w.data(), bias + cb, 0, 1, 2, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TADD(acc, acc, w);
    pipe_barrier(PIPE_V);
    pto::TCVT(ho, acc, pto::RoundMode::CAST_NONE);
    if (activation) {
      pipe_barrier(PIPE_V);
      pto::TCVT(acc, ho, pto::RoundMode::CAST_NONE);
      pto::TEXPANDS(scratch, 0.0f);
      pipe_barrier(PIPE_V);
      pto::TSUB(scratch, scratch, acc);
      pipe_barrier(PIPE_V);
      pto::TEXP(scratch, scratch);
      pto::TEXPANDS(ones, 1.0f);
      pipe_barrier(PIPE_V);
      pto::TADD(scratch, scratch, ones);
      pipe_barrier(PIPE_V);
      pto::TDIV(acc, acc, scratch);
      pipe_barrier(PIPE_V);
      pto::TCVT(ho, acc, pto::RoundMode::CAST_NONE);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(out + pixel * channels + cb, ho.data(), 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
}
