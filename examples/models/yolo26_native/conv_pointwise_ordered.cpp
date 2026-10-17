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
// One nonzero K row per fractal keeps each FP32 partial sum in logical channel order.
// L1 holds one complete sparse weight panel; output-panel-major work reuses it.
template <int Rows>
AICORE inline void ordered_pointwise(__gm__ int64_t* args, __ubuf__ float* meta) {
  using Mat = pto::Tile<pto::TileType::Mat, half, Rows, 16, pto::BLayout::RowMajor, Rows, 16,
                        pto::SLayout::RowMajor, 512>;
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
  int ci = meta[9], co = meta[10], pixels = static_cast<int>(meta[7]) * static_cast<int>(meta[8]);
  int source_stride = meta[13] == 0 ? ci : static_cast<int>(meta[13]);
  x += static_cast<int>(meta[14]);
  out += static_cast<int>(meta[12]);
  // Rows <= 128: L1 A uses 4 KiB; weights use <= 256 KiB from byte 8192.
  // Ordered replicas use at most 64 KiB each of L0A and L0B.
  // Four L0C partials use <= 32 KiB. UB slots reserve up to 8 KiB.
  Mat ma;
  auto* operand_a = reinterpret_cast<__ca__ half*>(0);
  auto* operand_b = reinterpret_cast<__cb__ half*>(0);
  Acc acc0, acc1, acc2, acc3;
  Vec value, v1, v2, v3, scratch, ones;
  HalfVec rounded;
  pto::TASSIGN<0>(ma);
  pto::TASSIGN<0>(acc0);
  pto::TASSIGN<4096 * (Rows > 64 ? 2 : 1)>(acc1);
  pto::TASSIGN<8192 * (Rows > 64 ? 2 : 1)>(acc2);
  pto::TASSIGN<12288 * (Rows > 64 ? 2 : 1)>(acc3);
  pto::TASSIGN<0>(value);
  pto::TASSIGN<4096 * (Rows > 64 ? 2 : 1)>(v1);
  pto::TASSIGN<8192 * (Rows > 64 ? 2 : 1)>(v2);
  pto::TASSIGN<12288 * (Rows > 64 ? 2 : 1)>(v3);
  pto::TASSIGN<16384 * (Rows > 64 ? 2 : 1)>(scratch);
  pto::TASSIGN<20480 * (Rows > 64 ? 2 : 1)>(ones);
  pto::TASSIGN<28672 * (Rows > 64 ? 2 : 1)>(rounded);
  auto* vbias = reinterpret_cast<__ubuf__ float*>(24576 * (Rows > 64 ? 2 : 1));
  auto* cache = reinterpret_cast<__cbuf__ half*>(8192);
  using Zeros = pto::Tile<pto::TileType::Vec, half, 1, 4096>;
  Zeros zeros;
  pto::TASSIGN<65536>(zeros);
  pto::TEXPANDS(zeros, static_cast<half>(0));
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  for (int start = 0; start < ci * 256; start += 4096)
    copy_ubuf_to_cbuf(cache + start, zeros.data(), 0, 1, 256, 0, 0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  int groups = pixels / Rows, cb_cached = -1;
  for (int work = get_block_idx(); work < groups * (co / 16); work += get_block_num()) {
    int cb = work / groups, pixel = work % groups * Rows;
    if (cb != cb_cached) {
      for (int ic = 0; ic < ci / 16; ++ic)
        copy_gm_to_cbuf(cache + ic * 16 * 256, w + (static_cast<int64_t>(cb) * ci + ic * 16) * 16, 0, 16, 1,
                        0, 16, (pad_t)0);
      cb_cached = cb;
    }
    // One sparse K16 fractal per logical term keeps the exact sum order.
    // Fill the 64 KiB L0A budget with ordered K64..K2048 MAD batches.
    // Four replicas of each C0 input tile supply the four interleaved terms.
    constexpr int channels_per_batch = 8192 / Rows;
    for (int first = 0; first < ci; first += channels_per_batch) {
      const int count = ci - first < channels_per_batch ? ci - first : channels_per_batch;
      for (int input_block = 0; input_block < count / 16; ++input_block) {
        copy_gm_to_cbuf(ma.data(), x + static_cast<int64_t>(pixel) * source_stride + first + input_block * 16,
                        0, Rows, 1, source_stride / 16 - 1, 0, (pad_t)0);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        for (int replica = 0; replica < 4; ++replica)
          load_cbuf_to_ca(operand_a + (input_block * 4 + replica) * 256, ma.data(), 0, Rows / 16, 1,
                          count / 4 - 1, 0, false, addr_cal_mode_t(0));
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      }
#pragma unroll
      for (int partial = 0; partial < 4; ++partial) {
        // Source stride four selects increasing CHW terms of one partial.
        load_cbuf_to_cb(operand_b, cache + (first + partial) * 256, 0, count / 4, 4, 0, 0, true,
                        addr_cal_mode_t(0));
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (partial == 0)
          mad(acc0.data(), operand_a, operand_b, Rows, count * 4, 16, first == 0);
        else if (partial == 1)
          mad(acc1.data(), operand_a, operand_b, Rows, count * 4, 16, first == 0);
        else if (partial == 2)
          mad(acc2.data(), operand_a, operand_b, Rows, count * 4, 16, first == 0);
        else
          mad(acc3.data(), operand_a, operand_b, Rows, count * 4, 16, first == 0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      }
      // Drain weight reads before MTE2 can replace the next L1 output panel.
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    }
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
  auto* meta = reinterpret_cast<__ubuf__ float*>(131072);
  copy_gm_to_ubuf(meta, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  int pixels = static_cast<int>(meta[7]) * static_cast<int>(meta[8]), co = meta[10];
  if (pixels % 128 == 0 && pixels / 128 * (co / 16) >= get_block_num())
    ordered_pointwise<128>(args, meta);
  else if (pixels % 64 == 0 && pixels / 64 * (co / 16) >= get_block_num())
    ordered_pointwise<64>(args, meta);
  else if (pixels % 32 == 0 && pixels / 32 * (co / 16) >= get_block_num())
    ordered_pointwise<32>(args, meta);
  else
    ordered_pointwise<16>(args, meta);
}
