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

using pto::AccToVecMode;
using pto::BLayout;
using pto::Op;
using pto::RoundMode;
using pto::SLayout;
using pto::TADD;
using pto::TASSIGN;
using pto::TCVT;
using pto::TDIV;
using pto::TEXP;
using pto::TEXPANDS;
using pto::Tile;
using pto::TileType;
using pto::TMATMUL;
using pto::TMATMUL_ACC;
using pto::TMOV;
using pto::TSUB;

using Mat = Tile<TileType::Mat, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512>;
using Left = Tile<TileType::Left, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512>;
using Right = Tile<TileType::Right, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512>;
using Acc = Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024>;
using Vec = Tile<TileType::Vec, float, 16, 16>;
using HalfVec = Tile<TileType::Vec, half, 16, 16>;

// Metadata is validated by ConvPlan before dispatch. All input/weight/output
// rows use padded multiples of sixteen channels. One work item owns 16 pixels
// and sixteen output channels; blocks write disjoint GM regions.
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* px = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* pw = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* pb = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* pz = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* x = reinterpret_cast<__gm__ half*>(px->buffer.addr);
  auto* w = reinterpret_cast<__gm__ half*>(pw->buffer.addr);
  auto* bias = reinterpret_cast<__gm__ float*>(pb->buffer.addr);
  auto* out = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  auto* zero = reinterpret_cast<__gm__ half*>(pz->buffer.addr);
  auto* meta = reinterpret_cast<__ubuf__ float*>(32768);
  copy_gm_to_ubuf(meta, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  const int ih = meta[0], iw = meta[1], kh = meta[4], stride = meta[5], pad = meta[6];
  const int oh = meta[7], ow = meta[8], ci = meta[9], co = meta[10], activation = meta[11];
  out += static_cast<int>(meta[12]);
  const int input_blocks = ci / 16, output_blocks = co / 16;
  Mat ma, mb;
  Left a;
  Right b;
  Acc acc;
  Vec value, scratch, ones;
  HalfVec rounded;
  TASSIGN<0>(ma);
  TASSIGN<512>(mb);
  TASSIGN<0>(a);
  TASSIGN<0>(b);
  TASSIGN<0>(acc);
  TASSIGN<0>(value);
  TASSIGN<4096>(scratch);
  TASSIGN<8192>(ones);
  TASSIGN<24576>(rounded);
  auto* vbias = reinterpret_cast<__ubuf__ float*>(16384);
  const int work_count = oh * ow / 16 * output_blocks;
  for (int work = get_block_idx(); work < work_count; work += get_block_num()) {
    const int pixel_base = work / output_blocks * 16;
    const int output_block = work % output_blocks;
    bool initialize = true;
    for (int ky = 0; ky < kh; ++ky) {
      for (int kx = 0; kx < kh; ++kx) {
        for (int ic = 0; ic < input_blocks; ++ic) {
          for (int row = 0; row < 16; ++row) {
            const int pixel = pixel_base + row;
            const int sy = pixel / ow * stride + ky - pad;
            const int sx = pixel % ow * stride + kx - pad;
            auto* source =
                sy >= 0 && sy < ih && sx >= 0 && sx < iw ? x + (sy * iw + sx) * ci + ic * 16 : zero;
            copy_gm_to_cbuf(ma.data() + row * 16, source, 0, 1, 1, 0, 0, (pad_t)0);
          }
          const int weight_panel = ((output_block * kh + ky) * kh + kx) * input_blocks + ic;
          copy_gm_to_cbuf(mb.data(), w + weight_panel * 256, 0, 1, 16, 0, 0, (pad_t)0);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          TMOV(a, ma);
          TMOV(b, mb);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (initialize)
            TMATMUL(acc, a, b);
          else
            TMATMUL_ACC(acc, acc, a, b);
          initialize = false;
          set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        }
      }
    }
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    TMOV<Vec, Acc, AccToVecMode::SingleModeVec0>(value, acc);
    // Release L0C only after the vector-side accumulator read completes.
    set_flag(PIPE_V, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
    copy_gm_to_ubuf(vbias, bias + output_block * 16, 0, 1, 2, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    for (int row = 0; row < 16; ++row)
      pto::detail::A1VectorInstruction<Op::TADD, 16>(value.data() + row * 16, value.data() + row * 16, vbias);
    pipe_barrier(PIPE_V);
    TCVT(rounded, value, RoundMode::CAST_NONE);
    if (activation) {
      pipe_barrier(PIPE_V);
      TCVT(value, rounded, RoundMode::CAST_NONE);
      TEXPANDS(scratch, 0.0f);
      pipe_barrier(PIPE_V);
      TSUB(scratch, scratch, value);
      pipe_barrier(PIPE_V);
      TEXP(scratch, scratch);
      TEXPANDS(ones, 1.0f);
      pipe_barrier(PIPE_V);
      TADD(scratch, scratch, ones);
      pipe_barrier(PIPE_V);
      TDIV(value, value, scratch);
      pipe_barrier(PIPE_V);
      TCVT(rounded, value, RoundMode::CAST_NONE);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(out + pixel_base * co + output_block * 16, rounded.data(), 0, 16, 1, 0,
                    output_blocks - 1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}
