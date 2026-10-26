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
using H16 = pto::Tile<pto::TileType::Vec, half, 1, 16>;
using H256 = pto::Tile<pto::TileType::Vec, half, 1, 256>;
using F16 = pto::Tile<pto::TileType::Vec, float, 1, 16>;
using F256 = pto::Tile<pto::TileType::Vec, float, 1, 256>;

extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* pa = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* pb = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* x = reinterpret_cast<__gm__ half*>(pa->buffer.addr);
  auto* b = reinterpret_cast<__gm__ half*>(pb->buffer.addr);
  auto* out = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  auto* m = reinterpret_cast<__ubuf__ float*>(32768);
  copy_gm_to_ubuf(m, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  const int op = m[0], h = m[1], w = m[2], channels = m[3], src_stride = m[4], dst_stride = m[5];
  const int src_offset = m[6], dst_offset = m[7], iw = m[9], b_stride = m[10], b_offset = m[11];
  const float scale = m[15];
  if (op == 1 && channels <= 1024 && (src_stride - channels) / 16 <= 65535 &&
      (dst_stride - channels) / 16 <= 65535) {
    // Full-channel row panels use two 16 KiB UB slots below metadata.
    int rows = 16;
    while (rows > 1 && (rows * channels > 8192 || (h * w + rows - 1) / rows < get_block_num())) rows /= 2;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    int slot = 0;
    for (int first = get_block_idx() * rows; first < h * w; first += get_block_num() * rows) {
      const int count = h * w - first < rows ? h * w - first : rows;
      auto* panel = reinterpret_cast<__ubuf__ half*>(slot * 16384);
      if (slot == 0)
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      else
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      copy_gm_to_ubuf(panel, x + static_cast<int64_t>(first) * src_stride + src_offset, 0, count,
                      channels / 16, (src_stride - channels) / 16, 0);
      if (slot == 0) {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      }
      copy_ubuf_to_gm(out + static_cast<int64_t>(first) * dst_stride + dst_offset, panel, 0, count,
                      channels / 16, 0, (dst_stride - channels) / 16);
      if (slot == 0)
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      else
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      slot ^= 1;
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    return;
  }
  if (op == 3 && channels <= 1024 && (src_stride - channels) / 16 <= 65535 &&
      (b_stride - channels) / 16 <= 65535 && (dst_stride - channels) / 16 <= 65535) {
    // Three operands each have two 8 KiB slots. Metadata at 32 KiB stays
    // untouched; MTE3 releases a slot only after consuming its output.
    int rows = 16;
    while (rows > 1 && (rows * channels > 4096 || (h * w + rows - 1) / rows < get_block_num())) rows /= 2;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    int slot = 0;
    for (int first = get_block_idx() * rows; first < h * w; first += get_block_num() * rows) {
      const int count = h * w - first < rows ? h * w - first : rows;
      auto* left = reinterpret_cast<__ubuf__ half*>(slot * 8192);
      auto* right = reinterpret_cast<__ubuf__ half*>(16384 + slot * 8192);
      auto* result = reinterpret_cast<__ubuf__ half*>(49152 + slot * 8192);
      if (slot == 0)
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      else
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      copy_gm_to_ubuf(left, x + static_cast<int64_t>(first) * src_stride + src_offset, 0, count,
                      channels / 16, (src_stride - channels) / 16, 0);
      copy_gm_to_ubuf(right, b + static_cast<int64_t>(first) * b_stride + b_offset, 0, count, channels / 16,
                      (b_stride - channels) / 16, 0);
      if (slot == 0) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      }
      const int elements = count * channels, full = elements / 128 * 128, tail = elements - full;
      set_mask_norm();
      set_vector_mask(UINT64_MAX, UINT64_MAX);
      if (full != 0) vadd(result, left, right, full / 128, 1, 1, 1, 8, 8, 8);
      if (tail != 0) {
        const uint64_t low = tail >= 64 ? UINT64_MAX : (UINT64_C(1) << tail) - 1;
        const uint64_t high = tail <= 64 ? 0 : (UINT64_C(1) << (tail - 64)) - 1;
        set_vector_mask(high, low);
        vadd(result + full, left + full, right + full, 1, 1, 1, 1, 8, 8, 8);
      }
      if (slot == 0) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      } else {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      }
      copy_ubuf_to_gm(out + static_cast<int64_t>(first) * dst_stride + dst_offset, result, 0, count,
                      channels / 16, 0, (dst_stride - channels) / 16);
      if (slot == 0)
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      else
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      slot ^= 1;
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    return;
  }
  if (op == 4 && channels <= 1024 && static_cast<int64_t>(h) * w * channels <= 65536 &&
      (src_stride - channels) / 16 <= 65535) {
    // Retain a compact full-channel input map in the upper UB. The lower
    // scratch and metadata stay disjoint; larger maps retain the C16 path.
    const int pixels = h * w;
    if (get_block_idx() >= pixels) return;
    auto* resident = reinterpret_cast<__ubuf__ half*>(65536);
    for (int first = 0; first < pixels; first += 4095) {
      const int count = pixels - first < 4095 ? pixels - first : 4095;
      copy_gm_to_ubuf(resident + first * channels, x + static_cast<int64_t>(first) * src_stride + src_offset,
                      0, count, channels / 16, (src_stride - channels) / 16, 0);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    using PoolResult = pto::Tile<pto::TileType::Vec, half, 1, 1024>;
    PoolResult result;
    pto::TASSIGN<0>(result);
    const int full = channels / 128 * 128, tail = channels - full;
    for (int pixel = get_block_idx(); pixel < pixels; pixel += get_block_num()) {
      pto::TEXPANDS(result, static_cast<half>(-65504));
      pipe_barrier(PIPE_V);
      // Keep the original ky/kx comparison order and finite padding value.
      // Only transfers and the number of independent channel lanes change.
      for (int ky = -2; ky <= 2; ++ky)
        for (int kx = -2; kx <= 2; ++kx) {
          const int sy = pixel / w + ky, sx = pixel % w + kx;
          if (sy < 0 || sy >= h || sx < 0 || sx >= w) continue;
          auto* neighbor = resident + (sy * w + sx) * channels;
          set_mask_norm();
          set_vector_mask(UINT64_MAX, UINT64_MAX);
          if (full != 0) vmax(result.data(), result.data(), neighbor, full / 128, 1, 1, 1, 8, 8, 8);
          if (tail != 0) {
            const uint64_t low = tail >= 64 ? UINT64_MAX : (UINT64_C(1) << tail) - 1;
            const uint64_t high = tail <= 64 ? 0 : (UINT64_C(1) << (tail - 64)) - 1;
            set_vector_mask(high, low);
            vmax(result.data() + full, result.data() + full, neighbor + full, 1, 1, 1, 1, 8, 8, 8);
          }
          pipe_barrier(PIPE_V);
        }
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      copy_ubuf_to_gm(out + static_cast<int64_t>(pixel) * dst_stride + dst_offset, result.data(), 0, 1,
                      channels / 16, 0, 0);
      // The output DMA must consume this result before the next pixel resets it.
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    return;
  }
  if (op == 2 && channels <= 1024 && (dst_stride - channels) / 16 <= 65535) {
    // One full-channel input pixel produces a pair reused for both output
    // rows. Integer OR copies all FP16 bits, including NaNs and signed zero.
    // Two independent input/pair slots let the next DMA overlap output stores.
    const int input_pixels = h / 2 * iw;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    int slot = 0;
    const int full = channels / 128 * 128, tail = channels - full;
    for (int pixel = get_block_idx(); pixel < input_pixels; pixel += get_block_num()) {
      auto* input = reinterpret_cast<__ubuf__ uint16_t*>(slot * 4096);
      auto* pair = reinterpret_cast<__ubuf__ uint16_t*>(8192 + slot * 8192);
      if (slot == 0)
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      else
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      copy_gm_to_ubuf(reinterpret_cast<__ubuf__ half*>(input),
                      x + static_cast<int64_t>(pixel) * src_stride + src_offset, 0, 1, channels / 16, 0, 0);
      if (slot == 0) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      }
      for (int replica = 0; replica < 2; ++replica) {
        auto* target = pair + replica * channels;
        set_mask_norm();
        set_vector_mask(UINT64_MAX, UINT64_MAX);
        if (full != 0) vor(target, input, input, full / 128, 1, 1, 1, 8, 8, 8);
        if (tail != 0) {
          const uint64_t low = tail >= 64 ? UINT64_MAX : (UINT64_C(1) << tail) - 1;
          const uint64_t high = tail <= 64 ? 0 : (UINT64_C(1) << (tail - 64)) - 1;
          set_vector_mask(high, low);
          vor(target + full, input + full, input + full, 1, 1, 1, 1, 8, 8, 8);
        }
      }
      if (slot == 0) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      } else {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      }
      const int first = pixel / iw * 2 * w + pixel % iw * 2;
      for (int row = 0; row < 2; ++row)
        copy_ubuf_to_gm(out + static_cast<int64_t>(first + row * w) * dst_stride + dst_offset,
                        reinterpret_cast<__ubuf__ half*>(pair), 0, 2, channels / 16, 0,
                        (dst_stride - channels) / 16);
      if (slot == 0)
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      else
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      slot ^= 1;
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    return;
  }
  H16 a, bb, c;
  H256 hs, hd;
  F256 fs;
  pto::TASSIGN<0>(a);
  pto::TASSIGN<1024>(bb);
  pto::TASSIGN<2048>(c);
  pto::TASSIGN<0>(hs);
  pto::TASSIGN<1024>(hd);
  pto::TASSIGN<4096>(fs);
  if (op == 0) {
    // NCHW FP32 input -> padded NHWC FP16, including the input cast.
    auto* source = reinterpret_cast<__gm__ float*>(pa->buffer.addr);
    auto* loaded = reinterpret_cast<__ubuf__ float*>(4096);
    H16 channel_half;
    F16 channel_float;
    uint64_t source_rows[16], destination_rows[16];
    for (int row = 0; row < 16; ++row) {
      source_rows[row] = reinterpret_cast<uint64_t>(hs.data() + row * 16);
      destination_rows[row] = reinterpret_cast<uint64_t>(hd.data() + row * 16);
    }
    set_va_reg_sb(VA2, source_rows);
    set_va_reg_sb(VA3, source_rows + 8);
    set_va_reg_sb(VA0, destination_rows);
    set_va_reg_sb(VA1, destination_rows + 8);
    for (int block = get_block_idx(); block < h * w / 16; block += get_block_num()) {
      pto::TEXPANDS(hs, static_cast<half>(0));
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      for (int ci = 0; ci < channels; ++ci)
        copy_gm_to_ubuf(loaded + ci * 16, source + ci * h * w + block * 16, 0, 1, 2, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      for (int ci = 0; ci < channels; ++ci) {
        pto::TASSIGN(channel_half, ci * 32);
        pto::TASSIGN(channel_float, 4096 + ci * 64);
        pto::TCVT(channel_half, channel_float, pto::RoundMode::CAST_NONE);
      }
      pipe_barrier(PIPE_V);
      scatter_vnchwconv_b16(VA0, VA2, 1, 0, 0);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      copy_ubuf_to_gm(out + block * 16 * dst_stride, hd.data(), 0, 1, dst_stride, 0, 0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    return;
  }
  if (op == 5 || op == 6 || op == 7) {
    for (int block = get_block_idx(); block < h / 256; block += get_block_num()) {
      if (op == 5)
        copy_gm_to_ubuf(fs.data(), reinterpret_cast<__gm__ float*>(pa->buffer.addr) + block * 256, 0, 1, 32,
                        0, 0);
      else
        copy_gm_to_ubuf(hs.data(), x + block * 256, 0, 1, 16, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      if (op == 5) {
        pto::TCVT(hd, fs, pto::RoundMode::CAST_NONE);
      } else {
        pto::TCVT(fs, hs, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(0, UINT64_MAX);
        vmuls(fs.data(), fs.data(), static_cast<float>(scale), 4, 1, 1, 8, 8);
        if (op == 7) {
          pipe_barrier(PIPE_V);
          pto::TCVT(hd, fs, pto::RoundMode::CAST_NONE);
        }
      }
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      if (op == 5 || op == 7)
        copy_ubuf_to_gm(out + block * 256, hd.data(), 0, 1, 16, 0, 0);
      else
        copy_ubuf_to_gm(reinterpret_cast<__gm__ float*>(po->buffer.addr) + block * 256, fs.data(), 0, 1, 32,
                        0, 0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    return;
  }
  const int channel_blocks = channels / 16;
  for (int work = get_block_idx(); work < h * w * channel_blocks; work += get_block_num()) {
    const int pixel = work / channel_blocks, cb = work % channel_blocks * 16;
    if (op == 4) {
      pto::TEXPANDS(c, static_cast<half>(-65504));
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      for (int ky = -2; ky <= 2; ++ky)
        for (int kx = -2; kx <= 2; ++kx) {
          const int sy = pixel / w + ky, sx = pixel % w + kx;
          if (sy < 0 || sy >= h || sx < 0 || sx >= w) continue;
          copy_gm_to_ubuf(a.data(), x + (sy * w + sx) * src_stride + src_offset + cb, 0, 1, 1, 0, 0);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          pto::detail::A1VectorInstruction<pto::Op::TMAX, 16>(c.data(), c.data(), a.data());
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      copy_ubuf_to_gm(out + pixel * dst_stride + dst_offset + cb, c.data(), 0, 1, 1, 0, 0);
      // The next pool work item initializes c before its first input DMA.
      // Hold vector reuse until the previous store has consumed the tile.
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    } else {
      const int source_pixel = op == 2 ? (pixel / w / 2 * iw + (pixel % w) / 2) : pixel;
      copy_gm_to_ubuf(a.data(), x + source_pixel * src_stride + src_offset + cb, 0, 1, 1, 0, 0);
      if (op == 3) {
        copy_gm_to_ubuf(bb.data(), b + pixel * b_stride + b_offset + cb, 0, 1, 1, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TADD(c, a, bb);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      }
      copy_ubuf_to_gm(out + pixel * dst_stride + dst_offset + cb, op == 3 ? c.data() : a.data(), 0, 1, 1, 0,
                      0);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
}
