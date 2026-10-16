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
using HPatch = pto::Tile<pto::TileType::Vec, half, 1, 576>;
using F16 = pto::Tile<pto::TileType::Vec, float, 1, 16>;
using F64 = pto::Tile<pto::TileType::Vec, float, 1, 64>;
using F256 = pto::Tile<pto::TileType::Vec, float, 1, 256>;
using FQuad = pto::Tile<pto::TileType::Vec, float, 4, 16>;
using FPatch = pto::Tile<pto::TileType::Vec, float, 1, 576>;

// Replicate one 16-channel weight row across four spatial pixels.
AICORE inline void repeat_row(__ubuf__ float* dst, __ubuf__ float* src) {
  set_mask_norm();
  set_vector_mask(0, 65535);
  vmax(dst, src, src, 4, 1, 1, 1, 2, 0, 0);
  set_vector_mask(UINT64_MAX, UINT64_MAX);
}

extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* px = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* pw = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* pb = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* pz = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* x = reinterpret_cast<__gm__ half*>(px->buffer.addr);
  auto* weight = reinterpret_cast<__gm__ half*>(pw->buffer.addr);
  auto* bias = reinterpret_cast<__gm__ float*>(pb->buffer.addr);
  auto* out = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  auto* zero = reinterpret_cast<__gm__ half*>(pz->buffer.addr);
  auto* meta = reinterpret_cast<__ubuf__ float*>(49152);
  copy_gm_to_ubuf(meta, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  const int ih = meta[0], iw = meta[1], input_channels = meta[2], kh = meta[4], stride = meta[5],
            pad = meta[6];
  const int oh = meta[7], ow = meta[8], ci = meta[9], co = meta[10], activation = meta[11];
  out += static_cast<int>(meta[12]);
  const int area = kh * kh, total = input_channels * area, full = total / 4 * 4, padded = (total + 3) / 4 * 4;
  HPatch hp;
  FPatch fp;
  F64 a0, a1, a2, a3, result, scratch, ones, vbias, tail_input, tail_product, tail_expanded;
  F256 acc, product, input, expanded_weight;
  F16 bias_row;
  FQuad ew0, ew1, ew2, ew3, bias_quad, tail_quad;
  H64 rounded;
  // One input-channel block has at most 16 * 9 reduction terms and 16 outputs.
  auto* weight_half = reinterpret_cast<__ubuf__ half*>(65536);
  auto* weight_float = reinterpret_cast<__ubuf__ float*>(73728);
  pto::TASSIGN<0>(hp);
  pto::TASSIGN<4096>(fp);
  pto::TASSIGN<16384>(acc);
  pto::TASSIGN<20480>(product);
  pto::TASSIGN<24576>(input);
  pto::TASSIGN<16384>(a0);
  pto::TASSIGN<16640>(a1);
  pto::TASSIGN<16896>(a2);
  pto::TASSIGN<17152>(a3);
  pto::TASSIGN<28672>(result);
  pto::TASSIGN<32768>(scratch);
  pto::TASSIGN<36864>(ones);
  pto::TASSIGN<40960>(rounded);
  pto::TASSIGN<45056>(vbias);
  pto::TASSIGN<24576>(tail_input);
  pto::TASSIGN<20480>(tail_product);
  pto::TASSIGN<53248>(bias_row);
  pto::TASSIGN<45056>(bias_quad);
  pto::TASSIGN<57344>(expanded_weight);
  pto::TASSIGN<57344>(ew0);
  pto::TASSIGN<57600>(ew1);
  pto::TASSIGN<57856>(ew2);
  pto::TASSIGN<58112>(ew3);
  pto::TASSIGN<57344>(tail_quad);
  pto::TASSIGN<57344>(tail_expanded);
  const int output_blocks = co / 16;
  for (int work = get_block_idx(); work < oh * ow / 4 * output_blocks; work += get_block_num()) {
    const int pixel = work / output_blocks * 4, cb = work % output_blocks;
    pto::TEXPANDS(acc, 0.0f);
    for (int ic = 0; ic < (input_channels + 15) / 16; ++ic) {
      const int begin = ic * 16 * area;
      const int end = total < begin + 16 * area ? total : begin + 16 * area;
      const int cached_terms = (end - begin + 3) / 4 * 4;
      pto::TEXPANDS(hp, static_cast<half>(0));
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      for (int q = 0; q < 4; ++q)
        for (int pos = 0; pos < area; ++pos) {
          const int sy = (pixel + q) / ow * stride + pos / kh - pad;
          const int sx = (pixel + q) % ow * stride + pos % kh - pad;
          auto* source = sy >= 0 && sy < ih && sx >= 0 && sx < iw ? x + (sy * iw + sx) * ci + ic * 16 : zero;
          copy_gm_to_ubuf(hp.data() + (q * area + pos) * 16, source, 0, 1, 1, 0, 0);
        }
      // The packed CHW weight block is contiguous, including only its padded tail.
      copy_gm_to_ubuf(weight_half, weight + (cb * padded + begin) * 16, 0, 1, cached_terms, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pto::TCVT(fp, hp, pto::RoundMode::CAST_NONE);
      set_mask_norm();
      set_vector_mask(0, UINT64_MAX);
      vconv_f162f32(weight_float, weight_half, cached_terms / 4, 1, 1, 8, 4);
      set_vector_mask(UINT64_MAX, UINT64_MAX);
      set_flag(PIPE_V, PIPE_S, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
      for (int k = begin; k < end; k += 4) {
        const int local = k - begin;
        if (k < full) {
          const float s0_0 = fp.data()[0 * area * 16 + ((local + 0) % area) * 16 + (local + 0) / area];
          const float s0_1 = fp.data()[1 * area * 16 + ((local + 0) % area) * 16 + (local + 0) / area];
          const float s0_2 = fp.data()[2 * area * 16 + ((local + 0) % area) * 16 + (local + 0) / area];
          const float s0_3 = fp.data()[3 * area * 16 + ((local + 0) % area) * 16 + (local + 0) / area];
          const float s1_0 = fp.data()[0 * area * 16 + ((local + 1) % area) * 16 + (local + 1) / area];
          const float s1_1 = fp.data()[1 * area * 16 + ((local + 1) % area) * 16 + (local + 1) / area];
          const float s1_2 = fp.data()[2 * area * 16 + ((local + 1) % area) * 16 + (local + 1) / area];
          const float s1_3 = fp.data()[3 * area * 16 + ((local + 1) % area) * 16 + (local + 1) / area];
          const float s2_0 = fp.data()[0 * area * 16 + ((local + 2) % area) * 16 + (local + 2) / area];
          const float s2_1 = fp.data()[1 * area * 16 + ((local + 2) % area) * 16 + (local + 2) / area];
          const float s2_2 = fp.data()[2 * area * 16 + ((local + 2) % area) * 16 + (local + 2) / area];
          const float s2_3 = fp.data()[3 * area * 16 + ((local + 2) % area) * 16 + (local + 2) / area];
          const float s3_0 = fp.data()[0 * area * 16 + ((local + 3) % area) * 16 + (local + 3) / area];
          const float s3_1 = fp.data()[1 * area * 16 + ((local + 3) % area) * 16 + (local + 3) / area];
          const float s3_2 = fp.data()[2 * area * 16 + ((local + 3) % area) * 16 + (local + 3) / area];
          const float s3_3 = fp.data()[3 * area * 16 + ((local + 3) % area) * 16 + (local + 3) / area];
          set_flag(PIPE_S, PIPE_V, EVENT_ID6);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID6);
          set_mask_norm();
          set_vector_mask(0, 65535);
          vector_dup(input.data() + 0, static_cast<float>(s0_0), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 16, static_cast<float>(s0_1), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 32, static_cast<float>(s0_2), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 48, static_cast<float>(s0_3), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 64, static_cast<float>(s1_0), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 80, static_cast<float>(s1_1), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 96, static_cast<float>(s1_2), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 112, static_cast<float>(s1_3), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 128, static_cast<float>(s2_0), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 144, static_cast<float>(s2_1), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 160, static_cast<float>(s2_2), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 176, static_cast<float>(s2_3), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 192, static_cast<float>(s3_0), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 208, static_cast<float>(s3_1), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 224, static_cast<float>(s3_2), 1, 1, 1, 8, 8);
          vector_dup(input.data() + 240, static_cast<float>(s3_3), 1, 1, 1, 8, 8);
          set_vector_mask(UINT64_MAX, UINT64_MAX);
          repeat_row(ew0.data(), weight_float + (local + 0) * 16);
          repeat_row(ew1.data(), weight_float + (local + 1) * 16);
          repeat_row(ew2.data(), weight_float + (local + 2) * 16);
          repeat_row(ew3.data(), weight_float + (local + 3) * 16);
          pipe_barrier(PIPE_V);
          pto::TMUL(product, input, expanded_weight);
          pipe_barrier(PIPE_V);
          pto::TADD(acc, acc, product);
        } else {
          // The final one to three CHW terms extend partial sum zero in order.
          for (int tail = k; tail < end; ++tail) {
            const int t = tail - begin;
            const float value0 = fp.data()[0 * area * 16 + (t % area) * 16 + t / area];
            const float value1 = fp.data()[1 * area * 16 + (t % area) * 16 + t / area];
            const float value2 = fp.data()[2 * area * 16 + (t % area) * 16 + t / area];
            const float value3 = fp.data()[3 * area * 16 + (t % area) * 16 + t / area];
            set_flag(PIPE_S, PIPE_V, EVENT_ID6);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID6);
            set_mask_norm();
            set_vector_mask(0, 65535);
            vector_dup(tail_input.data() + 0, static_cast<float>(value0), 1, 1, 1, 8, 8);
            vector_dup(tail_input.data() + 16, static_cast<float>(value1), 1, 1, 1, 8, 8);
            vector_dup(tail_input.data() + 32, static_cast<float>(value2), 1, 1, 1, 8, 8);
            vector_dup(tail_input.data() + 48, static_cast<float>(value3), 1, 1, 1, 8, 8);
            set_vector_mask(UINT64_MAX, UINT64_MAX);
            repeat_row(tail_quad.data(), weight_float + t * 16);
            pipe_barrier(PIPE_V);
            pto::TMUL(tail_product, tail_input, tail_expanded);
            pipe_barrier(PIPE_V);
            pto::TADD(a0, a0, tail_product);
          }
        }
      }
    }
    pipe_barrier(PIPE_V);
    pto::TADD(result, a0, a1);
    pipe_barrier(PIPE_V);
    pto::TADD(result, result, a2);
    pipe_barrier(PIPE_V);
    pto::TADD(result, result, a3);
    copy_gm_to_ubuf(bias_row.data(), bias + cb * 16, 0, 1, 2, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    repeat_row(bias_quad.data(), bias_row.data());
    pipe_barrier(PIPE_V);
    pto::TADD(result, result, vbias);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, result, pto::RoundMode::CAST_NONE);
    if (activation) {
      pipe_barrier(PIPE_V);
      pto::TCVT(result, rounded, pto::RoundMode::CAST_NONE);
      pto::TEXPANDS(scratch, 0.0f);
      pipe_barrier(PIPE_V);
      pto::TSUB(scratch, scratch, result);
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
      pto::TMUL(result, result, scratch);
      pipe_barrier(PIPE_V);
      pto::TCVT(rounded, result, pto::RoundMode::CAST_NONE);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    for (int q = 0; q < 4; ++q)
      copy_ubuf_to_gm(out + (pixel + q) * co + cb * 16, rounded.data() + q * 16, 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}
