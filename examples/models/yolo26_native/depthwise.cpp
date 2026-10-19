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
template <int Width>
AICORE inline void depthwise_panel(__gm__ int64_t* args, __ubuf__ float* m) {
  using H = pto::Tile<pto::TileType::Vec, half, 1, Width>;
  using F = pto::Tile<pto::TileType::Vec, float, 1, Width>;
  using HW = pto::Tile<pto::TileType::Vec, half, 1, 9 * Width>;
  using FW = pto::Tile<pto::TileType::Vec, float, 1, 9 * Width>;
  auto* px = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* pw = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* pb = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* x = reinterpret_cast<__gm__ half*>(px->buffer.addr);
  auto* weight = reinterpret_cast<__gm__ half*>(pw->buffer.addr);
  auto* bias = reinterpret_cast<__gm__ float*>(pb->buffer.addr);
  auto* out = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  const int ih = m[0], iw = m[1], channels = m[2], stride = m[3], oh = m[4], ow = m[5], activation = m[7];
  const bool four_way = m[8] != 0.0f;
  H hs, ho;
  HW hw;
  FW weights;
  F a, w, acc, scratch, ones, p1, p2, p3;
  pto::TASSIGN<0>(hs);
  pto::TASSIGN<65536>(hw);
  pto::TASSIGN<73728>(weights);
  pto::TASSIGN<2048>(ho);
  pto::TASSIGN<4096>(a);
  pto::TASSIGN<8192>(w);
  pto::TASSIGN<12288>(acc);
  pto::TASSIGN<16384>(scratch);
  pto::TASSIGN<20480>(ones);
  pto::TASSIGN<40960>(p1);
  pto::TASSIGN<45056>(p2);
  pto::TASSIGN<49152>(p3);
  auto* cached_bias = reinterpret_cast<__ubuf__ float*>(86016);
  int cached_cb = -1;
  // Channel-panel-major work retains nine converted weight rows across pixels.
  for (int work = get_block_idx(); work < oh * ow * (channels / Width); work += get_block_num()) {
    const int pixel = work % (oh * ow), cb = work / (oh * ow) * Width;
    if (cb != cached_cb) {
      if (channels / 16 - Width / 16 <= 65535) {
        copy_gm_to_ubuf(hw.data(), weight + cb, 0, 9, Width / 16, (channels - Width) / 16, 0);
      } else {
        for (int pos = 0; pos < 9; ++pos)
          copy_gm_to_ubuf(hw.data() + pos * Width, weight + pos * channels + cb, 0, 1, Width / 16, 0, 0);
      }
      copy_gm_to_ubuf(cached_bias, bias + cb, 0, 1, Width / 8, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pto::TCVT(weights, hw, pto::RoundMode::CAST_NONE);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      cached_cb = cb;
    }
    pto::TEXPANDS(acc, 0.0f);
    if (four_way) {
      pto::TEXPANDS(p1, 0.0f);
      pto::TEXPANDS(p2, 0.0f);
      pto::TEXPANDS(p3, 0.0f);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int ky = 0; ky < 3; ++ky)
      for (int kx = 0; kx < 3; ++kx) {
        const int sy = pixel / ow * stride + ky - 1, sx = pixel % ow * stride + kx - 1;
        if (sy < 0 || sy >= ih || sx < 0 || sx >= iw) continue;
        copy_gm_to_ubuf(hs.data(), x + (sy * iw + sx) * channels + cb, 0, 1, Width / 16, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TCVT(a, hs, pto::RoundMode::CAST_NONE);
        pto::TASSIGN(w, 73728 + (ky * 3 + kx) * Width * sizeof(float));
        pipe_barrier(PIPE_V);
        pto::TMUL(a, a, w);
        pipe_barrier(PIPE_V);
        const int partial = (ky * 3 + kx) % 4;
        if (!four_way || partial == 0)
          pto::TADD(acc, acc, a);
        else if (partial == 1)
          pto::TADD(p1, p1, a);
        else if (partial == 2)
          pto::TADD(p2, p2, a);
        else
          pto::TADD(p3, p3, a);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      }
    if (four_way) {
      pipe_barrier(PIPE_V);
      pto::TADD(acc, acc, p1);
      pipe_barrier(PIPE_V);
      pto::TADD(acc, acc, p2);
      pipe_barrier(PIPE_V);
      pto::TADD(acc, acc, p3);
    }
    pto::TASSIGN(w, 86016);
    pipe_barrier(PIPE_V);
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
      pto::TDIV(scratch, ones, scratch);
      pipe_barrier(PIPE_V);
      // Keep the sigmoid narrowing boundary used by PTO2's FP16 graph.
      pto::TCVT(ho, scratch, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      pto::TCVT(scratch, ho, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      pto::TMUL(acc, acc, scratch);
      pipe_barrier(PIPE_V);
      pto::TCVT(ho, acc, pto::RoundMode::CAST_NONE);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(out + pixel * channels + cb, ho.data(), 0, 1, Width / 16, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
}

extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* m = reinterpret_cast<__ubuf__ float*>(32768);
  copy_gm_to_ubuf(m, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  const int channels = m[2];
  // Keep channel-parallel work for grids too small to fill the launched cores.
  if (static_cast<int>(m[4]) * static_cast<int>(m[5]) < get_block_num())
    depthwise_panel<16>(args, m);
  else if (channels == 64)
    depthwise_panel<64>(args, m);
  else if (channels == 80)
    depthwise_panel<80>(args, m);
  else if (channels == 128)
    depthwise_panel<128>(args, m);
  else if (channels == 256)
    depthwise_panel<256>(args, m);
  else
    depthwise_panel<16>(args, m);
}
