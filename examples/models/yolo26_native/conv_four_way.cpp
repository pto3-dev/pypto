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
  F64 result, scratch, ones, vbias, tail_input, tail_product, tail_expanded;
  F256 acc, product, input, expanded_weight;
  F16 bias_row;
  FQuad bias_quad, tail_quad;
  H64 rounded;
  // One input-channel block has at most 16 * 9 reduction terms and 16 outputs.
  auto* weight_half = reinterpret_cast<__ubuf__ half*>(65536);
  auto* weight_float = reinterpret_cast<__ubuf__ float*>(73728);
  pto::TASSIGN<0>(hp);
  pto::TASSIGN<4096>(fp);
  pto::TASSIGN<16384>(acc);
  pto::TASSIGN<20480>(product);
  pto::TASSIGN<24576>(input);
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
  pto::TASSIGN<57344>(tail_quad);
  pto::TASSIGN<57344>(tail_expanded);
  auto* expanded_half = reinterpret_cast<__ubuf__ half*>(98304);
  auto* expanded_float = reinterpret_cast<__ubuf__ float*>(131072);
  uint64_t source_rows[4][16], destination_rows[4][16];
  for (int q = 0; q < 4; ++q)
    for (int row = 0; row < 16; ++row) {
      // Repeated source rows transpose each channel into 16 equal output lanes.
      source_rows[q][row] = reinterpret_cast<uint64_t>(hp.data() + q * area * 16);
      destination_rows[q][row] = reinterpret_cast<uint64_t>(expanded_half + row * 64 + q * 16);
    }
  const int output_blocks = co / 16, pixel_groups = oh * ow / 4;
  const bool cache_pointwise = kh == 1 && padded <= 512;
  // Reserve at most 512 * 16 FP32 values beyond the legacy scratch region.
  auto* pointwise_weight = reinterpret_cast<__ubuf__ float*>(196608);
  int cached_cb = -1;
  for (int work = get_block_idx(); work < oh * ow / 4 * output_blocks; work += get_block_num()) {
    const int pixel = cache_pointwise ? work % pixel_groups * 4 : work / output_blocks * 4;
    const int cb = cache_pointwise ? work / pixel_groups : work % output_blocks;
    if (cache_pointwise && cached_cb != cb) {
      // The full panel survives spatial work items until the output block changes.
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      copy_gm_to_ubuf(weight_half, weight + cb * padded * 16, 0, 1, padded, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      set_mask_norm();
      set_vector_mask(0, UINT64_MAX);
      vconv_f162f32(pointwise_weight, weight_half, padded / 4, 1, 1, 8, 4);
      set_vector_mask(UINT64_MAX, UINT64_MAX);
      pipe_barrier(PIPE_V);
      cached_cb = cb;
    }
    pto::TEXPANDS(acc, 0.0f);
    for (int ic = 0; ic < (input_channels + 15) / 16; ++ic) {
      const int begin = ic * 16 * area;
      const int end = total < begin + 16 * area ? total : begin + 16 * area;
      const int cached_terms = (end - begin + 3) / 4 * 4;
      // Every live patch lane is loaded below; only the scalar tail reads fp.
      if (end > full) pto::TEXPANDS(hp, static_cast<half>(0));
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
      if (!cache_pointwise)
        copy_gm_to_ubuf(weight_half, weight + (cb * padded + begin) * 16, 0, 1, cached_terms, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      if (end > full) pto::TCVT(fp, hp, pto::RoundMode::CAST_NONE);
      auto* block_weight = cache_pointwise ? pointwise_weight + begin * 16 : weight_float;
      if (!cache_pointwise) {
        set_mask_norm();
        set_vector_mask(0, UINT64_MAX);
        vconv_f162f32(block_weight, weight_half, cached_terms / 4, 1, 1, 8, 4);
        set_vector_mask(UINT64_MAX, UINT64_MAX);
      }
      if (begin < full) {
        for (int q = 0; q < 4; ++q) {
          set_va_reg_sb(VA2, source_rows[q]);
          set_va_reg_sb(VA3, source_rows[q] + 8);
          set_va_reg_sb(VA0, destination_rows[q]);
          set_va_reg_sb(VA1, destination_rows[q] + 8);
          // Single-repeat transposes require zero strides on DAV_2002.
          scatter_vnchwconv_b16(VA0, VA2, area, area == 1 ? 0 : 64, area == 1 ? 0 : 1);
          pipe_barrier(PIPE_V);
        }
        set_mask_norm();
        set_vector_mask(0, UINT64_MAX);
        vconv_f162f32(expanded_float, expanded_half, 16 * area, 1, 1, 8, 4);
        set_vector_mask(UINT64_MAX, UINT64_MAX);
      }
      set_flag(PIPE_V, PIPE_S, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
      for (int k = begin; k < end; k += 4) {
        const int local = k - begin;
        if (k < full) {
          set_mask_norm();
          set_vector_mask(0, UINT64_C(0xffff));
          for (int partial = 0; partial < 4; ++partial) {
            const int term = local + partial;
            auto* values = expanded_float + (term % area * 16 + term / area) * 64;
            vmax(input.data() + partial * 16, values, values, 4, 1, 1, 1, 8, 2, 2);
          }
          set_vector_mask(UINT64_MAX, UINT64_MAX);
          // Each repeat holds one pixel's four ordered partial sums.
          set_mask_norm();
          set_vector_mask(0, UINT64_MAX);
          vmax(expanded_weight.data(), block_weight + local * 16, block_weight + local * 16, 4, 1, 1, 1, 8, 0,
               0);
          set_vector_mask(UINT64_MAX, UINT64_MAX);
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
            repeat_row(tail_quad.data(), block_weight + t * 16);
            pipe_barrier(PIPE_V);
            pto::TMUL(tail_product, tail_input, tail_expanded);
            pipe_barrier(PIPE_V);
            // Only partial zero receives the final one to three CHW terms.
            set_mask_norm();
            set_vector_mask(0, UINT64_C(0xffff));
            vadd(acc.data(), acc.data(), tail_product.data(), 4, 1, 1, 1, 8, 8, 2);
            set_vector_mask(UINT64_MAX, UINT64_MAX);
          }
        }
      }
    }
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(0, UINT64_C(0xffff));
    vadd(result.data(), acc.data(), acc.data() + 16, 4, 1, 1, 1, 2, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(result.data(), result.data(), acc.data() + 32, 4, 1, 1, 1, 2, 2, 8);
    pipe_barrier(PIPE_V);
    vadd(result.data(), result.data(), acc.data() + 48, 4, 1, 1, 1, 2, 2, 8);
    set_vector_mask(UINT64_MAX, UINT64_MAX);
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
