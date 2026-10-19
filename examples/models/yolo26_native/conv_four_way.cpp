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

using F16 = pto::Tile<pto::TileType::Vec, float, 1, 16>;

// Replicate one 16-channel weight row across the live spatial pixels.
template <int SpatialPixels>
AICORE inline void repeat_row(__ubuf__ float* dst, __ubuf__ float* src) {
  set_mask_norm();
  set_vector_mask(0, 65535);
  vmax(dst, src, src, SpatialPixels, 1, 1, 1, 2, 0, 0);
  set_vector_mask(UINT64_MAX, UINT64_MAX);
}

template <int SpatialPixels, int KernelSize>
AICORE inline void convolution(__gm__ int64_t* args, __ubuf__ float* meta) {
  using HPatch = pto::Tile<pto::TileType::Vec, half, 1, SpatialPixels * 144>;
  using FPatch = pto::Tile<pto::TileType::Vec, float, 1, SpatialPixels * 144>;
  using FPixels = pto::Tile<pto::TileType::Vec, float, 1, SpatialPixels * 16>;
  using FAcc = pto::Tile<pto::TileType::Vec, float, 1, SpatialPixels * 64>;
  using FQuad = pto::Tile<pto::TileType::Vec, float, SpatialPixels, 16>;
  using HPixels = pto::Tile<pto::TileType::Vec, half, 1, SpatialPixels * 16>;
  auto* px = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* pw = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* pb = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* pz = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* x = reinterpret_cast<__gm__ half*>(px->buffer.addr);
  auto* weight = reinterpret_cast<__gm__ half*>(pw->buffer.addr);
  // Explicit runtime offset for constant weights; outputs and activations keep their normal policy.
  const uint64_t weight_alias = static_cast<uint64_t>(static_cast<int>(meta[16])) |
                                (static_cast<uint64_t>(static_cast<int>(meta[17])) << 16) |
                                (static_cast<uint64_t>(static_cast<int>(meta[18])) << 32) |
                                (static_cast<uint64_t>(static_cast<int>(meta[19])) << 48);
  weight = reinterpret_cast<__gm__ half*>(reinterpret_cast<uint64_t>(weight) + weight_alias);
  auto* bias = reinterpret_cast<__gm__ float*>(pb->buffer.addr);
  auto* out = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  auto* zero = reinterpret_cast<__gm__ half*>(pz->buffer.addr);
  constexpr int kh = KernelSize;
  const int ih = meta[0], iw = meta[1], input_channels = meta[2], stride = meta[5], pad = meta[6];
  const int oh = meta[7], ow = meta[8], ci = meta[9], co = meta[10], activation = meta[11];
  out += static_cast<int>(meta[12]);
  const int source_stride = meta[13] == 0 ? ci : static_cast<int>(meta[13]);
  x += static_cast<int>(meta[14]);
  const int area = kh * kh, total = input_channels * area, full = total / 4 * 4, padded = (total + 3) / 4 * 4;
  HPatch hp;
  FPatch fp;
  FPixels result, scratch, ones, vbias, tail_input, tail_product, tail_expanded;
  FAcc acc, product, input, expanded_weight;
  F16 bias_row;
  FQuad bias_quad, tail_quad;
  HPixels rounded;
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
  auto* expanded_float = reinterpret_cast<__ubuf__ float*>(SpatialPixels == 4 ? 131072 : 139264);
  uint64_t source_rows[SpatialPixels][16], destination_rows[SpatialPixels][16];
  for (int q = 0; q < SpatialPixels; ++q)
    for (int row = 0; row < 16; ++row) {
      // Repeated source rows transpose each channel into 16 equal output lanes.
      source_rows[q][row] = reinterpret_cast<uint64_t>(hp.data() + q * area * 16);
      destination_rows[q][row] =
          reinterpret_cast<uint64_t>(expanded_half + row * SpatialPixels * 16 + q * 16);
    }
  // Reuse an NC1HWC0 image in the conservative 512 KiB L1 budget.
  const bool cache_image =
      kh == 3 && static_cast<int64_t>(ih) * iw * ci <= 262144 && source_stride / 16 - 1 <= UINT16_MAX;
  auto* image = reinterpret_cast<__cbuf__ half*>(0);
  auto* image_patch = reinterpret_cast<__ubuf__ half*>(86016);
  if (cache_image) {
    for (int ic = 0; ic < ci / 16; ++ic)
      for (int start = 0; start < ih * iw; start += 4095) {
        const int count = ih * iw - start < 4095 ? ih * iw - start : 4095;
        copy_gm_to_cbuf(image + (ic * ih * iw + start) * 16,
                        x + static_cast<int64_t>(start) * source_stride + ic * 16, 0, count, 1,
                        source_stride / 16 - 1, 0, (pad_t)0);
      }
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    set_fmatrix(static_cast<uint64_t>(iw) | (static_cast<uint64_t>(ih) << 16) |
                (static_cast<uint64_t>(pad) << 32) | (static_cast<uint64_t>(pad) << 40) |
                (static_cast<uint64_t>(pad) << 48) | (static_cast<uint64_t>(pad) << 56));
    set_padding(0);
  }
  const int output_blocks = co / 16, pixel_groups = oh * ow / SpatialPixels;
  const bool cache_pointwise = kh == 1 && padded <= 512;
  // Reserve at most 512 * 16 FP32 values beyond the legacy scratch region.
  auto* pointwise_weight = reinterpret_cast<__ubuf__ float*>(196608);
  int cached_cb = -1;
  for (int work = get_block_idx(); work < oh * ow / SpatialPixels * output_blocks; work += get_block_num()) {
    const int pixel =
        cache_pointwise ? work % pixel_groups * SpatialPixels : work / output_blocks * SpatialPixels;
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
      if (cache_image) {
        const int first_pixel = pixel / 16 * 16;
        set_flag(PIPE_V, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID1);
        for (int pos = 0; pos < area; ++pos)
          img2col_cbuf_to_ub(image_patch + pos * 256, image + ic * ih * iw * 16, pos % kh, pos / kh,
                             first_pixel % ow * stride - pad, first_pixel / ow * stride - pad, 0, stride,
                             stride, kh, kh, 1, 1, 1, false, 1, CSIZE0);
        set_flag(PIPE_MTE1, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_V, EVENT_ID1);
        set_mask_norm();
        set_vector_mask(0, UINT64_C(0xffff));
        for (int pos = 0; pos < area; ++pos) {
          auto* patch = image_patch + pos * 256 + pixel % 16 * 16;
          vmax(hp.data() + pos * 16, patch, patch, SpatialPixels, 1, 1, 1, area, 1, 1);
        }
        set_vector_mask(UINT64_MAX, UINT64_MAX);
        pipe_barrier(PIPE_V);
      } else {
        for (int q = 0; q < SpatialPixels; ++q)
          for (int pos = 0; pos < area; ++pos) {
            const int sy = (pixel + q) / ow * stride + pos / kh - pad;
            const int sx = (pixel + q) % ow * stride + pos % kh - pad;
            auto* source = sy >= 0 && sy < ih && sx >= 0 && sx < iw
                               ? x + static_cast<int64_t>(sy * iw + sx) * source_stride + ic * 16
                               : zero;
            copy_gm_to_ubuf(hp.data() + (q * area + pos) * 16, source, 0, 1, 1, 0, 0);
          }
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
        for (int q = 0; q < SpatialPixels; ++q) {
          set_va_reg_sb(VA2, source_rows[q]);
          set_va_reg_sb(VA3, source_rows[q] + 8);
          set_va_reg_sb(VA0, destination_rows[q]);
          set_va_reg_sb(VA1, destination_rows[q] + 8);
          // Single-repeat transposes require zero strides on DAV_2002.
          scatter_vnchwconv_b16(VA0, VA2, area, area == 1 ? 0 : SpatialPixels * 16, area == 1 ? 0 : 1);
          pipe_barrier(PIPE_V);
        }
        set_mask_norm();
        set_vector_mask(0, UINT64_MAX);
        // Split at 144 repeats so every 3x3 conversion fits the 8-bit repeat field.
        for (int part = 0; part < SpatialPixels / 4; ++part)
          vconv_f162f32(expanded_float + part * 1024 * area, expanded_half + part * 1024 * area, 16 * area, 1,
                        1, 8, 4);
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
            auto* values = expanded_float + (term % area * 16 + term / area) * SpatialPixels * 16;
            vmax(input.data() + partial * 16, values, values, SpatialPixels, 1, 1, 1, 8, 2, 2);
          }
          set_vector_mask(UINT64_MAX, UINT64_MAX);
          // Each repeat holds one pixel's four ordered partial sums.
          set_mask_norm();
          set_vector_mask(0, UINT64_MAX);
          vmax(expanded_weight.data(), block_weight + local * 16, block_weight + local * 16, SpatialPixels, 1,
               1, 1, 8, 0, 0);
          set_vector_mask(UINT64_MAX, UINT64_MAX);
          pipe_barrier(PIPE_V);
          pto::TMUL(product, input, expanded_weight);
          pipe_barrier(PIPE_V);
          pto::TADD(acc, acc, product);
        } else {
          // The final one to three CHW terms extend partial sum zero in order.
          for (int tail = k; tail < end; ++tail) {
            const int t = tail - begin;
            float values[SpatialPixels];
            for (int q = 0; q < SpatialPixels; ++q)
              values[q] = fp.data()[q * area * 16 + (t % area) * 16 + t / area];
            set_flag(PIPE_S, PIPE_V, EVENT_ID6);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID6);
            set_mask_norm();
            set_vector_mask(0, 65535);
            for (int q = 0; q < SpatialPixels; ++q)
              vector_dup(tail_input.data() + q * 16, static_cast<float>(values[q]), 1, 1, 1, 8, 8);
            set_vector_mask(UINT64_MAX, UINT64_MAX);
            repeat_row<SpatialPixels>(tail_quad.data(), block_weight + t * 16);
            pipe_barrier(PIPE_V);
            pto::TMUL(tail_product, tail_input, tail_expanded);
            pipe_barrier(PIPE_V);
            // Only partial zero receives the final one to three CHW terms.
            set_mask_norm();
            set_vector_mask(0, UINT64_C(0xffff));
            vadd(acc.data(), acc.data(), tail_product.data(), SpatialPixels, 1, 1, 1, 8, 8, 2);
            set_vector_mask(UINT64_MAX, UINT64_MAX);
          }
        }
      }
    }
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(0, UINT64_C(0xffff));
    vadd(result.data(), acc.data(), acc.data() + 16, SpatialPixels, 1, 1, 1, 2, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(result.data(), result.data(), acc.data() + 32, SpatialPixels, 1, 1, 1, 2, 2, 8);
    pipe_barrier(PIPE_V);
    vadd(result.data(), result.data(), acc.data() + 48, SpatialPixels, 1, 1, 1, 2, 2, 8);
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    copy_gm_to_ubuf(bias_row.data(), bias + cb * 16, 0, 1, 2, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    repeat_row<SpatialPixels>(bias_quad.data(), bias_row.data());
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
    // ConvPlan bounds co to 1024, so this aligned DMA gap fits sixteen bits.
    copy_ubuf_to_gm(out + pixel * co + cb * 16, rounded.data(), 0, SpatialPixels, 1, 0, co / 16 - 1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}

extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* meta = reinterpret_cast<__ubuf__ float*>(49152);
  copy_gm_to_ubuf(meta, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  const int channels = meta[2], kernel = meta[4], oh = meta[7], ow = meta[8], co = meta[10];
  const int padded = (channels + 3) / 4 * 4;
  if (kernel == 1 && padded <= 512 && static_cast<int64_t>(oh) * ow / 8 * (co / 16) >= get_block_num())
    convolution<8, 1>(args, meta);
  else if (kernel == 1)
    convolution<4, 1>(args, meta);
  else if (static_cast<int64_t>(oh) * ow / 8 * (co / 16) >= get_block_num())
    convolution<8, 3>(args, meta);
  else
    convolution<4, 3>(args, meta);
}
