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
// Keep compact CHW weights in GM. Build each output panel's sparse K16 rows
// in bounded L1 and reuse it across spatial tiles, preserving the four FP32 sums.
// Hardware im2col moves resident NC1HWC0 image patches directly into L0A.
template <int Rows>
AICORE inline void ordered_spatial(__gm__ int64_t* args, __ubuf__ float* meta) {
  using WMat = pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::RowMajor, 16, 16,
                         pto::SLayout::RowMajor, 512>;
  using Left = pto::Tile<pto::TileType::Left, half, Rows, 16, pto::BLayout::ColMajor, Rows, 16,
                         pto::SLayout::RowMajor, 512>;
  using Right = pto::Tile<pto::TileType::Right, half, 16, 16, pto::BLayout::RowMajor, 16, 16,
                          pto::SLayout::ColMajor, 512>;
  using Acc = pto::Tile<pto::TileType::Acc, float, Rows, 16, pto::BLayout::ColMajor, Rows, 16,
                        pto::SLayout::RowMajor, 1024>;
  using Vec = pto::Tile<pto::TileType::Vec, float, Rows, 16>;
  using HalfVec = pto::Tile<pto::TileType::Vec, half, Rows, 16>;
  auto* px = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* pw = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* pb = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* x = reinterpret_cast<__gm__ half*>(px->buffer.addr);
  auto* w = reinterpret_cast<__gm__ half*>(pw->buffer.addr);
  const uint64_t weight_alias = static_cast<uint64_t>(static_cast<int>(meta[16])) |
                                (static_cast<uint64_t>(static_cast<int>(meta[17])) << 16) |
                                (static_cast<uint64_t>(static_cast<int>(meta[18])) << 32) |
                                (static_cast<uint64_t>(static_cast<int>(meta[19])) << 48);
  w = reinterpret_cast<__gm__ half*>(reinterpret_cast<uint64_t>(w) + weight_alias);
  auto* bias = reinterpret_cast<__gm__ float*>(pb->buffer.addr);
  auto* out = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  int ci = meta[9], co = meta[10], ih = meta[0], iw = meta[1], input_channels = meta[2], stride = meta[5],
      pad = meta[6], ow = meta[8];
  int pixels = static_cast<int>(meta[7]) * ow, total = input_channels * 9, full = total / 4 * 4,
      padded = (total + 3) / 4 * 4;
  int image_bytes = ih * iw * ci * 2, weight_base = (image_bytes + 511) / 512 * 512;
  int source_stride = meta[13] == 0 ? ci : static_cast<int>(meta[13]);
  x += static_cast<int>(meta[14]);
  out += static_cast<int>(meta[12]);
  // Nine L0A patches occupy <= 18 KiB, four L0C partials <= 16 KiB.
  // ConvPlan bounds the image plus sparse panels to 1 MiB of L1.
  // UB epilogue slots reserve 4 KiB each; the 8 KiB zero tile is separate.
  WMat mb;
  Left a;
  Right b;
  Acc acc0, acc1, acc2, acc3;
  Vec value, v1, v2, v3, scratch, ones;
  HalfVec rounded;
  pto::TASSIGN(mb, weight_base);
  pto::TASSIGN<0>(a);
  pto::TASSIGN<0>(b);
  pto::TASSIGN<0>(acc0);
  pto::TASSIGN<4096>(acc1);
  pto::TASSIGN<8192>(acc2);
  pto::TASSIGN<12288>(acc3);
  pto::TASSIGN<0>(value);
  pto::TASSIGN<4096>(v1);
  pto::TASSIGN<8192>(v2);
  pto::TASSIGN<12288>(v3);
  pto::TASSIGN<16384>(scratch);
  pto::TASSIGN<20480>(ones);
  pto::TASSIGN<28672>(rounded);
  auto* vbias = reinterpret_cast<__ubuf__ float*>(24576);
  auto* cache = reinterpret_cast<__cbuf__ half*>(weight_base);
  auto* image = reinterpret_cast<__cbuf__ half*>(0);
  using Zeros = pto::Tile<pto::TileType::Vec, half, 1, 4096>;
  Zeros zeros;
  pto::TASSIGN<65536>(zeros);
  pto::TEXPANDS(zeros, static_cast<half>(0));
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  for (int start = 0; start < padded * 256; start += 4096) {
    int count = padded * 256 - start < 4096 ? padded * 256 - start : 4096;
    copy_ubuf_to_cbuf(cache + start, zeros.data(), 0, 1, count / 16, 0, 0);
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  for (int ic = 0; ic < ci / 16; ++ic)
    for (int start = 0; start < ih * iw; start += 4095) {
      int count = ih * iw - start < 4095 ? ih * iw - start : 4095;
      copy_gm_to_cbuf(image + (ic * ih * iw + start) * 16,
                      x + static_cast<int64_t>(start) * source_stride + ic * 16, 0, count, 1,
                      source_stride / 16 - 1, 0, (pad_t)0);
    }
  set_fmatrix(static_cast<uint64_t>(iw) | (static_cast<uint64_t>(ih) << 16) |
              (static_cast<uint64_t>(pad) << 32) | (static_cast<uint64_t>(pad) << 40) |
              (static_cast<uint64_t>(pad) << 48) | (static_cast<uint64_t>(pad) << 56));
  set_padding(0);
  int groups = pixels / Rows, cb_cached = -1;
  for (int work = get_block_idx(); work < groups * (co / 16); work += get_block_num()) {
    int cb = work / groups, pixel = work % groups * Rows;
    if (cb != cb_cached) {
      for (int ic = 0; ic < ci / 16; ++ic) {
        int channels = input_channels - ic * 16 < 16 ? input_channels - ic * 16 : 16;
        for (int pos = 0; pos < 9; ++pos)
          copy_gm_to_cbuf(cache + (ic * 16 * 9 + pos) * 256,
                          w + (static_cast<int64_t>(cb) * padded + ic * 16 * 9 + pos) * 16, 0, channels, 1, 8,
                          144, (pad_t)0);
      }
      cb_cached = cb;
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    for (int ic = 0; ic < ci / 16; ++ic) {
      for (int pos = 0; pos < 9; ++pos) {
        pto::TASSIGN(a, pos * Rows * 32);
        for (int q = 0; q < Rows / 16; ++q)
          img2col_cbuf_to_ca(a.data() + q * 256, image + ic * ih * iw * 16, pos % 3, pos / 3,
                             (pixel + q * 16) % ow * stride - pad, (pixel + q * 16) / ow * stride - pad, 0,
                             stride, stride, 3, 3, 1, 1, 1, false, 1, CSIZE0);
      }
      int channels = input_channels - ic * 16 < 16 ? input_channels - ic * 16 : 16;
      for (int ki = 0; ki < channels; ++ki)
        for (int pos = 0; pos < 9; ++pos) {
          int k = (ic * 16 + ki) * 9 + pos, partial = k < full ? k % 4 : 0;
          pto::TASSIGN(a, pos * Rows * 32);
          pto::TASSIGN(mb, weight_base + k * 512);
          pto::TMOV(b, mb);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (partial == 0) {
            if (k == 0)
              pto::TMATMUL(acc0, a, b);
            else
              pto::TMATMUL_ACC(acc0, acc0, a, b);
          } else if (partial == 1) {
            if (k == 1)
              pto::TMATMUL(acc1, a, b);
            else
              pto::TMATMUL_ACC(acc1, acc1, a, b);
          } else if (partial == 2) {
            if (k == 2)
              pto::TMATMUL(acc2, a, b);
            else
              pto::TMATMUL_ACC(acc2, acc2, a, b);
          } else {
            if (k == 3)
              pto::TMATMUL(acc3, a, b);
            else
              pto::TMATMUL_ACC(acc3, acc3, a, b);
          }
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        }
    }
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(value, acc0);
    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(v1, acc1);
    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(v2, acc2);
    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(v3, acc3);
    set_flag(PIPE_V, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
    pipe_barrier(PIPE_V);
    pto::TADD(value, value, v1);
    pipe_barrier(PIPE_V);
    pto::TADD(value, value, v2);
    pipe_barrier(PIPE_V);
    pto::TADD(value, value, v3);
    copy_gm_to_ubuf(vbias, bias + cb * 16, 0, 1, 2, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    for (int row = 0; row < Rows; ++row)
      pto::detail::A1VectorInstruction<pto::Op::TADD, 16>(value.data() + row * 16, value.data() + row * 16,
                                                          vbias);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, value, pto::RoundMode::CAST_NONE);
    if (meta[11] != 0) {
      pipe_barrier(PIPE_V);
      pto::TCVT(value, rounded, pto::RoundMode::CAST_NONE);
      pto::TEXPANDS(scratch, 0.0f);
      pipe_barrier(PIPE_V);
      pto::TSUB(scratch, scratch, value);
      pipe_barrier(PIPE_V);
      pto::TEXP(scratch, scratch);
      pto::TEXPANDS(ones, 1.0f);
      pipe_barrier(PIPE_V);
      pto::TADD(scratch, scratch, ones);
      pipe_barrier(PIPE_V);
      pto::TDIV(scratch, ones, scratch);
      pipe_barrier(PIPE_V);
      pto::TCVT(rounded, scratch, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      pto::TCVT(scratch, rounded, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      pto::TMUL(value, value, scratch);
      pipe_barrier(PIPE_V);
      pto::TCVT(rounded, value, pto::RoundMode::CAST_NONE);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(out + static_cast<int64_t>(pixel) * co + cb * 16, rounded.data(), 0, Rows, 1, 0,
                    co / 16 - 1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* meta = reinterpret_cast<__ubuf__ float*>(32768);
  copy_gm_to_ubuf(meta, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  int pixels = static_cast<int>(meta[7]) * static_cast<int>(meta[8]), co = meta[10];
  if (pixels % 64 == 0 && pixels / 64 * (co / 16) >= get_block_num())
    ordered_spatial<64>(args, meta);
  else if (pixels % 32 == 0 && pixels / 32 * (co / 16) >= get_block_num())
    ordered_spatial<32>(args, meta);
  else
    ordered_spatial<16>(args, meta);
}
