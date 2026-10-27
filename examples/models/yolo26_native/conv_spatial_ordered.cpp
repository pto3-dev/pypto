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
// Ordered mode keeps compact CHW weights in GM and builds sparse K16 rows
// in bounded L1 and reuse it across spatial tiles, preserving the four FP32 sums.
// Hardware im2col moves resident NC1HWC0 image patches directly into L0A.
// Non-ordered mode retains dense K144 weights and one FP32 accumulator.
AICORE inline void set_spatial_image(int height, int width, int pad) {
  set_fmatrix(static_cast<uint64_t>(width) | (static_cast<uint64_t>(height) << 16) |
              (static_cast<uint64_t>(pad) << 32) | (static_cast<uint64_t>(pad) << 40) |
              (static_cast<uint64_t>(pad) << 48) | (static_cast<uint64_t>(pad) << 56));
}
AICORE inline void load_weight_panel(__cbuf__ half* cache, __gm__ half* w, int first, int count,
                                     int input_channels, int padded, int cb) {
  int blocks = (input_channels + 15) / 16;
  for (int local = 0; local < count && first + local < blocks; ++local) {
    int ic = first + local, channels = input_channels - ic * 16 < 16 ? input_channels - ic * 16 : 16;
    for (int pos = 0; pos < 9; ++pos)
      copy_gm_to_cbuf(cache + (local * 144 + pos) * 256,
                      w + (static_cast<int64_t>(cb) * padded + ic * 144 + pos) * 16, 0, channels, 1, 8, 144,
                      (pad_t)0);
  }
}
template <int Columns, bool Compact = false>
AICORE inline void load_weight_columns(__cbuf__ half* cache, __gm__ half* w, int first, int count,
                                       int input_channels, int padded, int allocated_terms, int cb) {
  if constexpr (Compact) {
    // Dense GM weights are already [N16, filter position, C1, C0, N0].
    copy_gm_to_cbuf(cache, w + static_cast<int64_t>(cb) * padded * 16, 0, 1, padded * (Columns / 16), 0, 0,
                    (pad_t)0);
  } else {
    load_weight_panel(cache, w, first, count, input_channels, padded, cb);
    if constexpr (Columns == 32)
      load_weight_panel(cache + allocated_terms * 256, w, first, count, input_channels, padded, cb + 1);
  }
}
AICORE inline void stage_spatial_rows(__cbuf__ half* image, __gm__ half* x, int y0, int y1, int iw, int ci,
                                      int source_stride) {
  const int resident_pixels = (y1 - y0) * iw;
  if (ci >= 32) {
    // Read complete NHWC channel rows, then scatter locally into NC1HWC0.
    // Two 32 KiB UB banks are dead before the vector epilogue starts.
    int rows = 128;
    while (rows * ci > 16384) rows /= 2;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    int slot = 0;
    for (int first = 0; first < resident_pixels; first += rows) {
      const int count = resident_pixels - first < rows ? resident_pixels - first : rows;
      auto* panel = reinterpret_cast<__ubuf__ half*>(slot * 32768);
      if (slot == 0)
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      else
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      copy_gm_to_ubuf(panel, x + (static_cast<int64_t>(y0) * iw + first) * source_stride, 0, count, ci / 16,
                      (source_stride - ci) / 16, 0);
      if (slot == 0) {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
      } else {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
      }
      for (int ic = 0; ic < ci / 16; ++ic)
        copy_ubuf_to_cbuf(image + (ic * resident_pixels + first) * 16, panel + ic * 16, 0, count, 1,
                          ci / 16 - 1, 0);
      if (slot == 0)
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      else
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      slot ^= 1;
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID3);
  } else {
    for (int ic = 0; ic < ci / 16; ++ic)
      for (int start = 0; start < resident_pixels; start += 4095) {
        int count = resident_pixels - start < 4095 ? resident_pixels - start : 4095;
        copy_gm_to_cbuf(image + (ic * resident_pixels + start) * 16,
                        x + (static_cast<int64_t>(y0) * iw + start) * source_stride + ic * 16, 0, count, 1,
                        source_stride / 16 - 1, 0, (pad_t)0);
      }
  }
}
template <int Rows, bool Streamed, int Columns = 16, bool Compact = false>
AICORE inline void ordered_spatial(__gm__ int64_t* args, __ubuf__ float* meta) {
  using WMat = pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::RowMajor, 16, 16,
                         pto::SLayout::RowMajor, 512>;
  using Left = pto::Tile<pto::TileType::Left, half, Rows, 16, pto::BLayout::ColMajor, Rows, 16,
                         pto::SLayout::RowMajor, 512>;
  using Right = pto::Tile<pto::TileType::Right, half, 16, 16, pto::BLayout::RowMajor, 16, 16,
                          pto::SLayout::ColMajor, 512>;
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
  int ci = meta[9], co = meta[10], ih = meta[0], iw = meta[1], input_channels = meta[2], stride = meta[5],
      pad = meta[6], ow = meta[8];
  const bool wide_windows = (Rows == 16 || Columns == 32) && input_channels % 4 == 0;
  int pixels = static_cast<int>(meta[7]) * ow, total = input_channels * 9, full = total / 4 * 4,
      padded = (total + 3) / 4 * 4;
  // Bound the worst row-crossing M tile plus its halo; weights start after
  // this allocation, while each C0 plane uses the actual clipped row count.
  const int full_image_bytes = ih * iw * ci * 2;
  const int bounded_terms = padded < 576 ? padded : 576;
  const bool windowed =
      !Compact && (full_image_bytes + 511) / 512 * 512 + bounded_terms * 512 * (Columns / 16) > 1048576;
  const int window_height = ((Rows + ow - 2) / ow) * stride + 3;
  const int storage_height = windowed && window_height < ih ? window_height : ih;
  int image_bytes = storage_height * iw * ci * 2, weight_base = (image_bytes + 511) / 512 * 512;
  int source_stride = meta[13] == 0 ? ci : static_cast<int>(meta[13]);
  x += static_cast<int>(meta[14]);
  out += static_cast<int>(meta[12]);
  // L0A patches occupy <= 36 KiB, four L0C partials <= 64 KiB.
  // ConvPlan bounds the image plus sparse panels to 1 MiB of L1.
  // UB epilogue slots reserve up to 16 KiB; initialization zeros are dead
  // before overlapping vector scratch is reused. Metadata starts at 128 KiB.
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
  auto* cache = reinterpret_cast<__cbuf__ half*>(weight_base);
  auto* image = reinterpret_cast<__cbuf__ half*>(0);
  const int panel_terms = Streamed ? 288 : padded;
  const int allocated_terms = Compact ? ci / 16 * 9 : Streamed ? 576 : padded;
  if constexpr (!Compact) {
    using Zeros = pto::Tile<pto::TileType::Vec, half, 1, 4096>;
    Zeros zeros;
    pto::TASSIGN<65536>(zeros);
    pto::TEXPANDS(zeros, static_cast<half>(0));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // Two 32-channel banks: each holds 288 logical K terms (144 KiB).
    // MTE2 prefetches the next bank while MTE1 and Cube consume this one.
    for (int start = 0; start < allocated_terms * 256 * (Columns / 16); start += 4096) {
      int count = allocated_terms * 256 * (Columns / 16) - start < 4096
                      ? allocated_terms * 256 * (Columns / 16) - start
                      : 4096;
      copy_ubuf_to_cbuf(cache + start, zeros.data(), 0, 1, count / 16, 0, 0);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  }
  const bool row_bands = windowed || (ci == 16 && co == 16 && ih >= 32);
  const int groups = pixels / Rows;
  // Every work item assigned to this core has the same spatial tile when
  // the block stride is a multiple of the spatial grid. Keep its halo once.
  const bool resident_band = !row_bands && groups > 1 && get_block_num() % groups == 0;
  const int first_pixel = (get_block_idx() % groups) * Rows;
  int resident_y0 = resident_band ? first_pixel / ow * stride - pad : 0;
  int resident_y1 = resident_band ? (first_pixel + Rows - 1) / ow * stride - pad + 3 : ih;
  if (resident_y0 < 0) resident_y0 = 0;
  if (resident_y1 > ih) resident_y1 = ih;
  const int resident_height = resident_y1 - resident_y0;
  const int resident_pixels = resident_height * iw;
  if (!row_bands) stage_spatial_rows(image, x, resident_y0, resident_y1, iw, ci, source_stride);
  set_spatial_image(resident_height, iw, pad);
  set_padding(0);
  int cb_cached = -1, cached_y0 = -1, cached_y1 = -1;
  const int panels = co / Columns;
  // Resident weights retain the bounded M128/C64 spatial-first policy.
  // Streamed N32 panels reload both weight banks for every work tile in
  // either traversal. Retain their input halo across output-channel panels
  // without introducing additional weight transfers or changing Cube sums.
  const bool reuse_input = windowed && ((Rows == 128 && ci <= 64 && (panels > 1 || ow >= 2 * Rows)) ||
                                        (Streamed && Columns == 32 && panels > 1));
  const int groups_per_core = (groups + get_block_num() - 1) / get_block_num();
  const int scheduled_groups = reuse_input ? groups_per_core * get_block_num() : groups;
  for (int work = get_block_idx(); work < scheduled_groups * panels; work += get_block_num()) {
    int cb = work / groups * (Columns / 16), pixel = work % groups * Rows;
    if (reuse_input) {
      // Own a contiguous spatial stripe and consume every output panel before
      // replacing its L1 input window. Mask the final core's ragged stripe.
      const int local = work / get_block_num();
      const int group = get_block_idx() * groups_per_core + local / panels;
      if (group >= groups) continue;
      cb = local % panels * (Columns / 16);
      pixel = group * Rows;
    }
    int image_y0 = resident_y0, image_pixels = resident_pixels;
    if (row_bands) {
      image_y0 = pixel / ow * stride - pad;
      if (image_y0 < 0) image_y0 = 0;
      int image_y1 = (pixel + Rows - 1) / ow * stride - pad + 3;
      if (image_y1 > ih) image_y1 = ih;
      image_pixels = (image_y1 - image_y0) * iw;
      if (!reuse_input || image_y0 != cached_y0 || image_y1 != cached_y1) {
        stage_spatial_rows(image, x, image_y0, image_y1, iw, ci, source_stride);
        set_spatial_image(image_y1 - image_y0, iw, pad);
        cached_y0 = image_y0;
        cached_y1 = image_y1;
      }
    }
    if constexpr (Streamed) {
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      load_weight_columns<Columns, Compact>(cache, w, 0, 2, input_channels, padded, allocated_terms, cb);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    } else {
      if (cb != cb_cached) {
        load_weight_columns<Columns, Compact>(cache, w, 0, ci / 16, input_channels, padded, allocated_terms,
                                              cb);
        cb_cached = cb;
      }
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    }
    for (int ic = 0; ic < ci / 16; ++ic) {
      if constexpr (Compact) {
        auto* dense_a = reinterpret_cast<__ca__ half*>(0);
        auto* dense_b = reinterpret_cast<__cb__ half*>(0);
        for (int pos = 0; pos < 9; ++pos)
          for (int q = 0; q < Rows / 16; ++q)
            img2col_cbuf_to_ca(dense_a + (q * 9 + pos) * 256, image + ic * image_pixels * 16, pos % 3,
                               pos / 3, (pixel + q * 16) % ow * stride - pad,
                               (pixel + q * 16) / ow * stride - pad - image_y0, 0, stride, stride, 3, 3, 1, 1,
                               1, false, 1, CSIZE0);
        load_cbuf_to_cb(dense_b, cache + ic * 256, 0, 9, ci / 16, Columns / 16 - 1, 0, true,
                        addr_cal_mode_t(0));
        if constexpr (Columns == 32)
          load_cbuf_to_cb(dense_b + 256, cache + allocated_terms * 256 + ic * 256, 0, 9, ci / 16, 1, 0, true,
                          addr_cal_mode_t(0));
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        mad(acc0.data(), dense_a, dense_b, Rows, 144, Columns, ic == 0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        continue;
      }
      int bank = Streamed ? (ic / 2) % 2 : 0;
      if constexpr (Streamed) {
        if (ic % 2 == 0) {
          if (bank == 0)
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          else
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          if (ic + 2 < ci / 16) {
            if (bank == 0)
              wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            else
              wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            load_weight_columns<Columns, Compact>(cache + (1 - bank) * panel_terms * 256, w, ic + 2, 2,
                                                  input_channels, padded, allocated_terms, cb);
            if (bank == 0)
              set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            else
              set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          }
        }
      }
      if (wide_windows) {
        if constexpr (Rows == 16) {
          // Four K144 windows overlap in a period-nine filter permutation.
          for (int j = 0; j < 16; ++j) {
            const int pos = j * 4 % 9;
            auto* patch = reinterpret_cast<__ca__ half*>(j * 512);
            img2col_cbuf_to_ca(patch, image + ic * image_pixels * 16, pos % 3, pos / 3,
                               pixel % ow * stride - pad, pixel / ow * stride - pad - image_y0, 0, stride,
                               stride, 3, 3, 1, 1, 1, false, 1, CSIZE0);
          }
        } else if constexpr (Rows < 64 || Columns != 32) {
          // Each phase has M-major K144 windows. Two M16 fragments share
          // one L0B load; four M32 windows occupy exactly 36 KiB of L0A.
          for (int phase = 0; phase < 4; ++phase)
            for (int j = 0; j < 9; ++j) {
              const int pos = (phase + j * 4) % 9;
              auto* patch = reinterpret_cast<__ca__ half*>(phase * Rows * 144 * 2);
              for (int q = 0; q < Rows / 16; ++q)
                img2col_cbuf_to_ca(patch + (q * 9 + j) * 256, image + ic * image_pixels * 16, pos % 3,
                                   pos / 3, (pixel + q * 16) % ow * stride - pad,
                                   (pixel + q * 16) / ow * stride - pad - image_y0, 0, stride, stride, 3, 3,
                                   1, 1, 1, false, 1, CSIZE0);
            }
        }
      } else {
        for (int pos = 0; pos < 9; ++pos) {
          const bool grouped = input_channels % 4 == 0;
          const int phase = pos % 4, count = phase == 0 ? 3 : 2;
          const int first = phase == 0 ? 0 : phase * 2 + 1;
          auto* patch = reinterpret_cast<__ca__ half*>(grouped ? first * Rows * 32 : pos * Rows * 32);
          for (int q = 0; q < Rows / 16; ++q)
            img2col_cbuf_to_ca(patch + (grouped ? q * count + pos / 4 : q) * 256,
                               image + ic * image_pixels * 16, pos % 3, pos / 3,
                               (pixel + q * 16) % ow * stride - pad,
                               (pixel + q * 16) / ow * stride - pad - image_y0, 0, stride, stride, 3, 3, 1, 1,
                               1, false, 1, CSIZE0);
        }
      }
      int channels = input_channels - ic * 16 < 16 ? input_channels - ic * 16 : 16;
      if (input_channels % 4 == 0) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      }
      if constexpr (Columns == 32 && Rows >= 64) {
        // Reuse one period-nine K144 im2col window for all four-channel
        // groups of this partial. CHW terms start at (ic*16+ki*4)*9+phase
        // and advance by four, so each partial retains its increasing K order.
        // M128 needs 36 KiB of L0A; keeping four phase windows would not fit.
        // Drain both MAD slots before replacing A. Alternating 16 KiB B slots
        // retain overlap with Cube, and the existing outer loop releases the
        // streamed L1 weight bank only after every phase has consumed it.
#pragma unroll
        for (int phase = 0; phase < 4; ++phase) {
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          auto* phase_a = reinterpret_cast<__ca__ half*>(0);
          for (int j = 0; j < 9; ++j) {
            const int pos = (phase + j * 4) % 9;
            for (int q = 0; q < Rows / 16; ++q)
              img2col_cbuf_to_ca(phase_a + (q * 9 + j) * 256, image + ic * image_pixels * 16, pos % 3,
                                 pos / 3, (pixel + q * 16) % ow * stride - pad,
                                 (pixel + q * 16) / ow * stride - pad - image_y0, 0, stride, stride, 3, 3, 1,
                                 1, 1, false, 1, CSIZE0);
          }
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          for (int ki = 0; ki < channels / 4; ++ki) {
            const int k = (ic * 16 + ki * 4) * 9 + phase;
            const int local = Streamed ? bank * panel_terms + (ic % 2) * 144 + ki * 36 + phase : k;
            const int slot = ki % 2;
            auto* phase_b = reinterpret_cast<__cb__ half*>(slot * 16384);
            if (slot == 0)
              wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            else
              wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            pto::TASSIGN(mb, weight_base + local * 512);
            load_cbuf_to_cb(phase_b, mb.data(), 0, 9, 4, 1, 0, true, addr_cal_mode_t(0));
            load_cbuf_to_cb(phase_b + 256, mb.data() + allocated_terms * 256, 0, 9, 4, 1, 0, true,
                            addr_cal_mode_t(0));
            if (slot == 0) {
              set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
              wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            } else {
              set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
              wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
            }
            if (phase == 0)
              mad(acc0.data(), phase_a, phase_b, Rows, 144, 32, ic == 0 && ki == 0);
            else if (phase == 1)
              mad(acc1.data(), phase_a, phase_b, Rows, 144, 32, ic == 0 && ki == 0);
            else if (phase == 2)
              mad(acc2.data(), phase_a, phase_b, Rows, 144, 32, ic == 0 && ki == 0);
            else
              mad(acc3.data(), phase_a, phase_b, Rows, 144, 32, ic == 0 && ki == 0);
            if (slot == 0)
              set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            else
              set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          }
        }
      } else {
        for (int ki = 0; ki < (wide_windows ? channels / 4 : channels); ++ki) {
          if (input_channels % 4 == 0) {
#pragma unroll
            for (int phase = 0; phase < 4; ++phase) {
              const int count = wide_windows ? 9 : phase == 0 ? 3 : 2;
              const int first = phase == 0 ? 0 : phase * 2 + 1;
              const int k = (ic * 16 + ki * (wide_windows ? 4 : 1)) * 9 + phase;
              const int local =
                  Streamed ? bank * panel_terms + (ic % 2) * 144 + ki * (wide_windows ? 36 : 9) + phase : k;
              auto* grouped_a =
                  reinterpret_cast<__ca__ half*>(Rows == 16     ? (phase == 0 ? 0 : 9 - 2 * phase) * 512
                                                 : wide_windows ? phase * Rows * 144 * 2
                                                                : first * Rows * 32);
              const int slot = phase % 2;
              auto* grouped_b =
                  reinterpret_cast<__cb__ half*>(slot * (wide_windows ? 8192 : 2048) * (Columns / 16));
              if (slot == 0)
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
              else
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
              pto::TASSIGN(mb, weight_base + local * 512);
              // Group increasing CHW terms of one partial into K32/K48/K144.
              load_cbuf_to_cb(grouped_b, mb.data(), 0, count, 4, Columns / 16 - 1, 0, true,
                              addr_cal_mode_t(0));
              if constexpr (Columns == 32)
                load_cbuf_to_cb(grouped_b + 256, mb.data() + allocated_terms * 256, 0, count, 4, 1, 0, true,
                                addr_cal_mode_t(0));
              if (slot == 0) {
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
              } else {
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
              }
              const int partial = wide_windows ? phase : (ki + phase) % 4;
              if (partial == 0)
                mad(acc0.data(), grouped_a, grouped_b, Rows, count * 16, Columns, ic == 0 && ki == 0);
              else if (partial == 1)
                mad(acc1.data(), grouped_a, grouped_b, Rows, count * 16, Columns, ic == 0 && ki == 0);
              else if (partial == 2)
                mad(acc2.data(), grouped_a, grouped_b, Rows, count * 16, Columns, ic == 0 && ki == 0);
              else
                mad(acc3.data(), grouped_a, grouped_b, Rows, count * 16, Columns, ic == 0 && ki == 0);
              if (slot == 0)
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
              else
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            }
          } else if constexpr (Columns == 16) {
            for (int pos = 0; pos < 9; ++pos) {
              int k = (ic * 16 + ki) * 9 + pos, partial = k < full ? k % 4 : 0;
              pto::TASSIGN(a, pos * Rows * 32);
              pto::TASSIGN(mb, weight_base +
                                   (Streamed ? bank * panel_terms + (ic % 2) * 144 + ki * 9 + pos : k) * 512);
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
        }
      }
      // Both L0B reads and the shared L0A image patches must be free before
      // the next channel block overwrites operands or releases its L1 bank.
      if (input_channels % 4 == 0) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      }
      if constexpr (Streamed) {
        if (ic % 2 == 1 || ic + 1 == ci / 16) {
          if (bank == 0)
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          else
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        }
      }
    }
    if constexpr (Streamed) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    }
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    copy_matrix_cc_to_ubuf(value.data(), acc0.data(), 0, 1, Rows / 16 * (Columns / 16), 0, 0, CRMODE_NONE);
    if constexpr (!Compact) {
      copy_matrix_cc_to_ubuf(v1.data(), acc1.data(), 0, 1, Rows / 16 * (Columns / 16), 0, 0, CRMODE_NONE);
      copy_matrix_cc_to_ubuf(v2.data(), acc2.data(), 0, 1, Rows / 16 * (Columns / 16), 0, 0, CRMODE_NONE);
      copy_matrix_cc_to_ubuf(v3.data(), acc3.data(), 0, 1, Rows / 16 * (Columns / 16), 0, 0, CRMODE_NONE);
    }
    set_flag(PIPE_V, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
    pipe_barrier(PIPE_V);
    if constexpr (!Compact) {
      pto::TADD(value, value, v1);
      pipe_barrier(PIPE_V);
      pto::TADD(value, value, v2);
      pipe_barrier(PIPE_V);
      pto::TADD(value, value, v3);
    } else {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
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
      refine_fp16_sigmoid_exp<Rows * Columns>(scratch.data(), value.data(), ones.data(), v1.data(),
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
template <int Rows>
AICORE inline void dispatch(__gm__ int64_t* args, __ubuf__ float* meta) {
  int image_bytes = static_cast<int>(meta[0]) * static_cast<int>(meta[1]) * static_cast<int>(meta[9]) * 2;
  int terms = (static_cast<int>(meta[2]) * 9 + 3) / 4 * 4;
  if ((image_bytes + 511) / 512 * 512 + (terms < 576 ? terms : 576) * 512 > 1048576) {
    int height =
        ((Rows + static_cast<int>(meta[8]) - 2) / static_cast<int>(meta[8])) * static_cast<int>(meta[5]) + 3;
    if (height > static_cast<int>(meta[0])) height = meta[0];
    image_bytes = height * static_cast<int>(meta[1]) * static_cast<int>(meta[9]) * 2;
  }
  if ((image_bytes + 511) / 512 * 512 + terms * 512 > 1048576)
    ordered_spatial<Rows, true>(args, meta);
  else
    ordered_spatial<Rows, false>(args, meta);
}
// Keep one phase-major K144 window live for M64/M128 and share each
// operand across N32 columns. Recompute the two-panel image/weight bound
// before selecting resident or double-buffered streamed weights.
template <int Rows>
AICORE inline bool try_large_spatial(__gm__ int64_t* args, __ubuf__ float* meta) {
  const int pixels = static_cast<int>(meta[7]) * static_cast<int>(meta[8]);
  const int ci = meta[9], co = meta[10], input_channels = meta[2], ih = meta[0], iw = meta[1], ow = meta[8];
  if (meta[15] != 0 || ci < 16 || input_channels % 4 != 0 || co % 32 != 0 || pixels % Rows != 0 ||
      pixels / Rows < get_block_num())
    return false;
  const int terms = (input_channels * 9 + 3) / 4 * 4, paired = terms < 576 ? terms : 576;
  int bytes = (ih * iw * ci * 2 + 511) / 512 * 512;
  if (bytes + 2 * paired * 512 > 1048576) {
    const int height = ((Rows + ow - 2) / ow) * static_cast<int>(meta[5]) + 3;
    bytes = ((height < ih ? height : ih) * iw * ci * 2 + 511) / 512 * 512;
  }
  if (bytes + 2 * paired * 512 > 1048576) return false;
  if (bytes + 2 * terms * 512 <= 1048576)
    ordered_spatial<Rows, false, 32>(args, meta);
  else
    ordered_spatial<Rows, true, 32>(args, meta);
  return true;
}
// A1 single-launch loads this object without a device link step. Force
// the wrapper entry flat so a larger dispatch does not leave a .rela.text call.
extern "C" AICORE inline __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* meta = reinterpret_cast<__ubuf__ float*>(131072);
  copy_gm_to_ubuf(meta, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  if (try_large_spatial<128>(args, meta) || try_large_spatial<64>(args, meta)) return;
  int pixels = static_cast<int>(meta[7]) * static_cast<int>(meta[8]), co = meta[10];
  const int image_bytes =
      (static_cast<int>(meta[0]) * static_cast<int>(meta[1]) * static_cast<int>(meta[9]) * 2 + 511) / 512 *
      512;
  const int terms = (static_cast<int>(meta[2]) * 9 + 3) / 4 * 4;
  const int paired_terms = terms < 576 ? terms : 576;
  // Share each ordered im2col operand across two output-channel panels.
  constexpr int wide_rows = 16;
  int wide_image_bytes = image_bytes;
  if (wide_image_bytes + 2 * paired_terms * 512 > 1048576) {
    const int height = ((wide_rows + static_cast<int>(meta[8]) - 2) / static_cast<int>(meta[8])) *
                           static_cast<int>(meta[5]) +
                       3;
    wide_image_bytes = ((height < static_cast<int>(meta[0]) ? height : static_cast<int>(meta[0])) *
                            static_cast<int>(meta[1]) * static_cast<int>(meta[9]) * 2 +
                        511) /
                       512 * 512;
  }
  // Large low-channel windows benefit from the existing M128 spatial-first
  // input reuse. Small M16 panels would repeat staging and sparse weight loads.
  const bool keep_large_window =
      static_cast<int>(meta[9]) <= 64 && image_bytes + paired_terms * 512 > 1048576;
  if (meta[15] == 0 && pixels > 64 && static_cast<int>(meta[9]) >= 32 && !keep_large_window && co % 32 == 0 &&
      static_cast<int>(meta[2]) % 4 == 0 && pixels / wide_rows * (co / 32) >= get_block_num() &&
      wide_image_bytes + 2 * paired_terms * 512 <= 1048576) {
    if (wide_image_bytes + 2 * terms * 512 <= 1048576)
      ordered_spatial<16, false, 32>(args, meta);
    else
      ordered_spatial<16, true, 32>(args, meta);
    return;
  }
  // M16/N32 shares one set of im2col windows across adjacent output panels.
  // Keep eight work tiles and bound both per-column weight banks in L1.
  const int compact_panels = co >= 64 && co % 32 == 0 ? 2 : 1;
  if (meta[15] != 0 && pixels == 64 && static_cast<int>(meta[2]) % 16 == 0 &&
      image_bytes + static_cast<int>(meta[9]) * 288 * compact_panels <= 1048576) {
    if (compact_panels == 2) {
      if (pixels / 32 * (co / 32) >= get_block_num())
        ordered_spatial<32, false, 32, true>(args, meta);
      else
        ordered_spatial<16, false, 32, true>(args, meta);
    } else if (pixels / 32 * (co / 16) >= get_block_num()) {
      ordered_spatial<32, false, 16, true>(args, meta);
    } else {
      ordered_spatial<16, false, 16, true>(args, meta);
    }
  } else if (pixels <= 64 && co >= 128 && static_cast<int>(meta[2]) % 4 == 0 && co % 32 == 0 &&
             pixels / 16 * (co / 32) >= get_block_num() && image_bytes + 2 * paired_terms * 512 <= 1048576) {
    if (pixels % 32 == 0 && pixels / 32 * (co / 32) >= get_block_num()) {
      if (image_bytes + 2 * terms * 512 <= 1048576)
        ordered_spatial<32, false, 32>(args, meta);
      else
        ordered_spatial<32, true, 32>(args, meta);
    } else if (image_bytes + 2 * terms * 512 <= 1048576) {
      ordered_spatial<16, false, 32>(args, meta);
    } else {
      ordered_spatial<16, true, 32>(args, meta);
    }
  } else if (pixels % 128 == 0 && pixels / 128 * (co / 16) >= get_block_num()) {
    dispatch<128>(args, meta);
  } else if (pixels % 64 == 0 && pixels / 64 * (co / 16) >= get_block_num()) {
    dispatch<64>(args, meta);
  } else if (pixels % 32 == 0 && pixels / 32 * (co / 16) >= get_block_num()) {
    dispatch<32>(args, meta);
  } else {
    dispatch<16>(args, meta);
  }
}
