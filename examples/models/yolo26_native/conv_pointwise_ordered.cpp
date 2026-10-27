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
#include "yolo26_native/conv_bias.hpp"
#include "yolo26_native/fp16_activation.hpp"
// The A1 hardware sigmoid differs from the CPU FP16 contract at x=-15/2048:
// vexp(-x) is one FP32 ULP low near one, crossing the later FP16 tie.
// Neighboring FP16 inputs are at least 2^-18 away. Scaling their distance
// by 2^19 and clamping creates an exact single-input mask, without a
// comparison-register update per vector repeat. Add that exponent ULP
// (2^-23) before the unchanged reciprocal and FP16 sigmoid/product casts.
// The exhaustive finite-FP16 test verifies the complete input domain.
template <unsigned Elements>
AICORE inline void correct_pointwise_sigmoid_boundary(__ubuf__ float* exponent, __ubuf__ float* value,
                                                      __ubuf__ float* mask, __ubuf__ float* temporary) {
  sigmoid_scalar<SigmoidScalar::Add, Elements>(mask, value, 15.0f / 2048);
  sigmoid_scalar<SigmoidScalar::Absolute, Elements>(mask, mask);
  sigmoid_scalar<SigmoidScalar::Multiply, Elements>(mask, mask, -524288.0f);
  sigmoid_scalar<SigmoidScalar::Add, Elements>(mask, mask, 1.0f);
  sigmoid_clamp<Elements>(mask, mask, 0.0f, 1.0f, temporary);
  sigmoid_scalar<SigmoidScalar::Multiply, Elements>(mask, mask, 0x1p-23f);
  pto::detail::A1VectorInstruction<pto::Op::TADD, Elements>(exponent, exponent, mask);
  pipe_barrier(PIPE_V);
}
// One nonzero K row per fractal keeps each FP32 partial sum in logical channel order.
// Paired output panels share a resident L1 input tile without changing reduction order.
template <int Rows, int Panels = 1>
AICORE inline void ordered_pointwise(__gm__ int64_t* args, __ubuf__ float* meta) {
  // A single N32 Cube operation shares the ordered left operand across two
  // output panels; independent columns retain their original K order.
  constexpr int Columns = Panels == 2 ? 32 : 16;
  using Mat = pto::Tile<pto::TileType::Mat, half, Rows, 16, pto::BLayout::RowMajor, Rows, 16,
                        pto::SLayout::RowMajor, 512>;
  using Acc = pto::Tile<pto::TileType::Acc, float, Rows, Columns, pto::BLayout::ColMajor, Rows, Columns,
                        pto::SLayout::RowMajor, 1024>;
  using Vec = pto::Tile<pto::TileType::Vec, float, Rows*(Columns / 16), 16>;
  using HalfVec = pto::Tile<pto::TileType::Vec, half, Rows*(Columns / 16), 16>;
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
  // The fallback uses one scratch input tile and one weight panel. Paired
  // panels reserve input below byte 262144 and two weight panels above it.
  // Ordered replicas use at most 64 KiB each of L0A and L0B.
  // Four L0C partials use <= 64 KiB. UB slots reserve up to 16 KiB.
  Mat ma;
  auto* operand_a = reinterpret_cast<__ca__ half*>(0);
  auto* operand_b = reinterpret_cast<__cb__ half*>(0);
  Acc acc0, acc1, acc2, acc3;
  Vec value, v1, v2, v3, scratch, ones;
  HalfVec rounded;
  pto::TASSIGN<0>(ma);
  pto::TASSIGN<0>(acc0);
  pto::TASSIGN<4096 * (Rows > 64 ? 2 : 1) * (Columns / 16)>(acc1);
  pto::TASSIGN<8192 * (Rows > 64 ? 2 : 1) * (Columns / 16)>(acc2);
  pto::TASSIGN<12288 * (Rows > 64 ? 2 : 1) * (Columns / 16)>(acc3);
  pto::TASSIGN<0>(value);
  pto::TASSIGN<4096 * (Rows > 64 ? 2 : 1) * (Columns / 16)>(v1);
  pto::TASSIGN<8192 * (Rows > 64 ? 2 : 1) * (Columns / 16)>(v2);
  pto::TASSIGN<12288 * (Rows > 64 ? 2 : 1) * (Columns / 16)>(v3);
  pto::TASSIGN<16384 * (Rows > 64 ? 2 : 1) * (Columns / 16)>(scratch);
  pto::TASSIGN<20480 * (Rows > 64 ? 2 : 1) * (Columns / 16)>(ones);
  pto::TASSIGN<28672 * (Rows > 64 ? 2 : 1) * (Columns / 16)>(rounded);
  auto* vbias = reinterpret_cast<__ubuf__ float*>(24576 * (Rows > 64 ? 2 : 1) * (Columns / 16));
  constexpr int weight_base = Panels == 1 ? 8192 : 262144;
  auto* weight_cache = reinterpret_cast<__cbuf__ half*>(weight_base);
  using Zeros = pto::Tile<pto::TileType::Vec, half, 1, 4096>;
  Zeros zeros;
  pto::TASSIGN<65536>(zeros);
  pto::TEXPANDS(zeros, static_cast<half>(0));
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  for (int start = 0; start < Panels * ci * 256; start += 4096)
    copy_ubuf_to_cbuf(weight_cache + start, zeros.data(), 0, 1, 256, 0, 0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  int groups = pixels / Rows, cb_cached = -1;
  for (int work = get_block_idx(); work < groups * (co / (16 * Panels)); work += get_block_num()) {
    int first_cb = work / groups * Panels, pixel = work % groups * Rows;
    if (first_cb != cb_cached) {
      for (int panel = 0; panel < Panels; ++panel)
        for (int ic = 0; ic < ci / 16; ++ic)
          copy_gm_to_cbuf(weight_cache + panel * ci * 256 + ic * 16 * 256,
                          w + (static_cast<int64_t>(first_cb + panel) * ci + ic * 16) * 16, 0, 16, 1, 0, 16,
                          (pad_t)0);
      cb_cached = first_cb;
    }
    if constexpr (Panels > 1) {
      // C0-major L1 storage survives both output panels. Up to 128 KiB of input
      // and 512 KiB of weights fit without overlapping within the 1 MiB L1.
      for (int ic = 0; ic < ci / 16; ++ic)
        copy_gm_to_cbuf(ma.data() + ic * Rows * 16, x + static_cast<int64_t>(pixel) * source_stride + ic * 16,
                        0, Rows, 1, source_stride / 16 - 1, 0, (pad_t)0);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    }
    for (int panel = 0; panel < Panels; panel += Columns / 16) {
      int cb = first_cb + panel;
      auto* cache = weight_cache + panel * ci * 256;
      // One sparse K16 fractal per logical term keeps the exact sum order.
      // Fill the 64 KiB L0A budget with ordered K64..K2048 MAD batches.
      // Four replicas of each C0 input tile supply the four interleaved terms.
      // Bound both A (Rows x K) and B (K x Columns) to 64 KiB.
      constexpr int channels_per_batch = 8192 / (Rows > Columns ? Rows : Columns);
      for (int first = 0; first < ci; first += channels_per_batch) {
        const int count = ci - first < channels_per_batch ? ci - first : channels_per_batch;
        for (int input_block = 0; input_block < count / 16; ++input_block) {
          auto* input_tile = ma.data();
          if constexpr (Panels == 1) {
            copy_gm_to_cbuf(input_tile,
                            x + static_cast<int64_t>(pixel) * source_stride + first + input_block * 16, 0,
                            Rows, 1, source_stride / 16 - 1, 0, (pad_t)0);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          } else {
            input_tile += (first / 16 + input_block) * Rows * 16;
          }
          for (int replica = 0; replica < 4; ++replica)
            load_cbuf_to_ca(operand_a + (input_block * 4 + replica) * 256, input_tile, 0, Rows / 16, 1,
                            count / 4 - 1, 0, false, addr_cal_mode_t(0));
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        }
#pragma unroll
        for (int partial = 0; partial < 4; ++partial) {
          // Source stride four selects increasing CHW terms of one partial.
          load_cbuf_to_cb(operand_b, cache + (first + partial) * 256, 0, count / 4, 4, Columns / 16 - 1, 0,
                          true, addr_cal_mode_t(0));
          // Interleave the second N16 fractal in the same K-major L0B
          // stream, sharing each ordered left operand across both columns.
          if constexpr (Columns == 32)
            load_cbuf_to_cb(operand_b + 256, cache + ci * 256 + (first + partial) * 256, 0, count / 4, 4, 1,
                            0, true, addr_cal_mode_t(0));
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (partial == 0)
            mad(acc0.data(), operand_a, operand_b, Rows, count * 4, Columns, first == 0);
          else if (partial == 1)
            mad(acc1.data(), operand_a, operand_b, Rows, count * 4, Columns, first == 0);
          else if (partial == 2)
            mad(acc2.data(), operand_a, operand_b, Rows, count * 4, Columns, first == 0);
          else
            mad(acc3.data(), operand_a, operand_b, Rows, count * 4, Columns, first == 0);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        }
        // Drain weight reads before MTE2 can replace the next L1 output panel.
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      }
      set_flag(PIPE_M, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
      // Keep NZ column blocks adjacent in UB; bias and output DMA use the
      // same Columns/16 by Rows by C16 layout as these accumulator copies.
      copy_matrix_cc_to_ubuf(value.data(), acc0.data(), 0, 1, Rows / 16 * (Columns / 16), 0, 0, CRMODE_NONE);
      copy_matrix_cc_to_ubuf(v1.data(), acc1.data(), 0, 1, Rows / 16 * (Columns / 16), 0, 0, CRMODE_NONE);
      copy_matrix_cc_to_ubuf(v2.data(), acc2.data(), 0, 1, Rows / 16 * (Columns / 16), 0, 0, CRMODE_NONE);
      copy_matrix_cc_to_ubuf(v3.data(), acc3.data(), 0, 1, Rows / 16 * (Columns / 16), 0, 0, CRMODE_NONE);
      set_flag(PIPE_V, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
      pipe_barrier(PIPE_V);
      pto::TADD(value, value, v1);
      pipe_barrier(PIPE_V);
      pto::TADD(value, value, v2);
      pipe_barrier(PIPE_V);
      pto::TADD(value, value, v3);
      // The next output panel must not overwrite bias while the previous
      // panel's final vector rows still consume it.
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      copy_gm_to_ubuf(vbias, bias + cb * 16, 0, 1, Columns / 8, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pipe_barrier(PIPE_V);
      add_conv_bias<Rows, Columns>(value.data(), vbias);
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
        pipe_barrier(PIPE_V);
        correct_pointwise_sigmoid_boundary<Rows * Columns>(scratch.data(), value.data(), v1.data(),
                                                           v2.data());
        pipe_barrier(PIPE_V);
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
      for (int column = 0; column < Columns / 16; ++column)
        copy_ubuf_to_gm(out + static_cast<int64_t>(pixel) * co + (cb + column) * 16,
                        rounded.data() + column * Rows * 16, 0, Rows, 1, 0, co / 16 - 1);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
  }
}
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* meta = reinterpret_cast<__ubuf__ float*>(131072);
  copy_gm_to_ubuf(meta, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  int pixels = static_cast<int>(meta[7]) * static_cast<int>(meta[8]), co = meta[10];
  if (pixels % 128 == 0 && pixels / 128 * (co / 16) >= get_block_num()) {
    if (co % 32 == 0 && pixels / 128 * (co / 32) >= get_block_num())
      ordered_pointwise<128, 2>(args, meta);
    else
      ordered_pointwise<128>(args, meta);
  } else if (pixels % 64 == 0 && pixels / 64 * (co / 16) >= get_block_num()) {
    if (co % 32 == 0 && pixels / 64 * (co / 32) >= get_block_num())
      ordered_pointwise<64, 2>(args, meta);
    else
      ordered_pointwise<64>(args, meta);
  } else if (pixels % 32 == 0 && pixels / 32 * (co / 16) >= get_block_num()) {
    if (co % 32 == 0 && pixels / 32 * (co / 32) >= get_block_num())
      ordered_pointwise<32, 2>(args, meta);
    else
      ordered_pointwise<32>(args, meta);
  } else if (co % 32 == 0 && pixels / 16 * (co / 32) >= get_block_num()) {
    ordered_pointwise<16, 2>(args, meta);
  } else {
    ordered_pointwise<16>(args, meta);
  }
}
