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
#include "yolo26_native/fp16_activation.hpp"
template <int Width, int Pixels = 1>
AICORE inline void depthwise_panel(__gm__ int64_t* args, __ubuf__ float* m, int completed_panel = 0) {
  // Adjacent interior pixels share the same valid tap set. Widening only the
  // independent vector lanes preserves each pixel's four-part reduction order.
  constexpr int Lanes = Width * Pixels;
  constexpr int weight_base = Pixels == 1 ? 73728 : 81920;
  constexpr int bias_base = Pixels == 1 ? 86016 : 110592;
  using H = pto::Tile<pto::TileType::Vec, half, 1, Lanes>;
  using F = pto::Tile<pto::TileType::Vec, float, 1, Lanes>;
  using HW = pto::Tile<pto::TileType::Vec, half, 1, 9 * Lanes>;
  using FW = pto::Tile<pto::TileType::Vec, float, 1, 9 * Lanes>;
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
  pto::TASSIGN<weight_base>(weights);
  pto::TASSIGN<2048>(ho);
  pto::TASSIGN<4096>(a);
  pto::TASSIGN<8192>(w);
  pto::TASSIGN<12288>(acc);
  pto::TASSIGN<16384>(scratch);
  pto::TASSIGN<20480>(ones);
  pto::TASSIGN<40960>(p1);
  pto::TASSIGN<45056>(p2);
  pto::TASSIGN<49152>(p3);
  auto* cached_bias = reinterpret_cast<__ubuf__ float*>(bias_base);
  int cached_cb = -1;
  // Full-channel panels can retain three NHWC input rows in the upper
  // 128 KiB of UB. Contiguous pixel stripes let neighboring outputs reuse
  // rows; each core loads its own halo and owns disjoint output pixels.
  const bool row_cache = channels == Width && iw <= 131072 / (6 * Width);
  const int groups_per_row = Pixels == 1 ? ow : (ow - 2) / Pixels;
  const int work_pixels = oh * groups_per_row;
  const int stripe = (work_pixels + get_block_num() - 1) / get_block_num();
  const int first = row_cache ? get_block_idx() * stripe : get_block_idx();
  const int limit = row_cache ? (first + stripe < work_pixels ? first + stripe : work_pixels)
                              : work_pixels * (channels / Width);
  const int step = row_cache ? 1 : get_block_num();
  int loaded_rows[3] = {-1, -1, -1};
  auto* rows = reinterpret_cast<__ubuf__ half*>(131072);
  // The fallback keeps channel-panel-major work and converted weight reuse.
  for (int work = first; work < limit; work += step) {
    const int pixel =
        Pixels == 1 ? work % work_pixels : work / groups_per_row * ow + 1 + work % groups_per_row * Pixels;
    const int cb = Pixels == 1 ? work / work_pixels * Width : 0;
    // The panel pass covers only complete interior groups. This scalar pass
    // owns both boundary columns and the short right tail, with no zero-filled
    // arithmetic on invalid taps (important for non-finite weights).
    if (completed_panel > 0 && pixel % ow >= 1 &&
        pixel % ow < 1 + (ow - 2) / completed_panel * completed_panel)
      continue;
    if (cb != cached_cb) {
      if constexpr (Pixels > 1) {
        // Replicate compact weights and bias once per core, not per pixel.
        // At 512 lanes these caches end below byte 112640; the row ring starts
        // at 131072 and remains separate from metadata and vector scratch.
        for (int replica = 0; replica < Pixels; ++replica) {
          copy_gm_to_ubuf(hw.data() + replica * Width, weight, 0, 9, Width / 16, 0, (Lanes - Width) / 16);
          copy_gm_to_ubuf(cached_bias + replica * Width, bias, 0, 1, Width / 8, 0, 0);
        }
      } else if (channels / 16 - Width / 16 <= 65535) {
        copy_gm_to_ubuf(hw.data(), weight + cb, 0, 9, Width / 16, (channels - Width) / 16, 0);
      } else {
        for (int pos = 0; pos < 9; ++pos)
          copy_gm_to_ubuf(hw.data() + pos * Width, weight + pos * channels + cb, 0, 1, Width / 16, 0, 0);
      }
      if constexpr (Pixels == 1) copy_gm_to_ubuf(cached_bias, bias + cb, 0, 1, Width / 8, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pto::TCVT(weights, hw, pto::RoundMode::CAST_NONE);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      cached_cb = cb;
    }
    if (row_cache) {
      for (int ky = 0; ky < 3; ++ky) {
        const int sy = pixel / ow * stride + ky - 1;
        if (sy < 0 || sy >= ih || loaded_rows[sy % 3] == sy) continue;
        copy_gm_to_ubuf(rows + sy % 3 * iw * Width, x + sy * iw * channels, 0, 1, iw * Width / 16, 0, 0);
        loaded_rows[sy % 3] = sy;
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }
    pto::TEXPANDS(acc, 0.0f);
    if (four_way) {
      pto::TEXPANDS(p1, 0.0f);
      pto::TEXPANDS(p2, 0.0f);
      pto::TEXPANDS(p3, 0.0f);
    }
    if (row_cache) {
      pipe_barrier(PIPE_V);
    } else {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    for (int ky = 0; ky < 3; ++ky)
      for (int kx = 0; kx < 3; ++kx) {
        const int sy = pixel / ow * stride + ky - 1, sx = pixel % ow * stride + kx - 1;
        if (sy < 0 || sy >= ih || sx < 0 || sx >= iw) continue;
        if (row_cache) {
          pto::TASSIGN(hs, 131072 + (sy % 3 * iw + sx) * Width * sizeof(half));
        } else {
          copy_gm_to_ubuf(hs.data(), x + (sy * iw + sx) * channels + cb, 0, 1, Width / 16, 0, 0);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }
        pto::TCVT(a, hs, pto::RoundMode::CAST_NONE);
        pto::TASSIGN(w, weight_base + (ky * 3 + kx) * Lanes * sizeof(float));
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
        if (row_cache) {
          pipe_barrier(PIPE_V);
        } else {
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
      }
    if (four_way) {
      pipe_barrier(PIPE_V);
      pto::TADD(acc, acc, p1);
      pipe_barrier(PIPE_V);
      pto::TADD(acc, acc, p2);
      pipe_barrier(PIPE_V);
      pto::TADD(acc, acc, p3);
    }
    pto::TASSIGN(w, bias_base);
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
      pipe_barrier(PIPE_V);
      refine_fp16_sigmoid_exp<Lanes>(scratch.data(), acc.data(), ones.data(), p1.data(), p2.data());
      pipe_barrier(PIPE_V);
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
    copy_ubuf_to_gm(out + pixel * channels + cb, ho.data(), 0, 1, Lanes / 16, 0, 0);
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
  // Four-pixel panels amortize vector setup and avoid the C80 tail on every
  // pixel. Whole interior groups use resident rows; the unchanged scalar pass
  // handles boundary columns and incomplete groups exactly.
  if (m[3] == 1 && m[5] >= 6 && static_cast<int>(m[4]) * static_cast<int>(m[5]) >= get_block_num() &&
      static_cast<int>(m[1]) <= 131072 / (6 * channels)) {
    if (channels == 64) {
      depthwise_panel<64, 4>(args, m);
      depthwise_panel<64>(args, m, 4);
      return;
    }
    if (channels == 80) {
      depthwise_panel<80, 4>(args, m);
      depthwise_panel<80>(args, m, 4);
      return;
    }
    if (channels == 128) {
      depthwise_panel<128, 4>(args, m);
      depthwise_panel<128>(args, m, 4);
      return;
    }
  }
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
