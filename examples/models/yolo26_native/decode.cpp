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
using H80 = pto::Tile<pto::TileType::Vec, half, 1, 80>;
using F80 = pto::Tile<pto::TileType::Vec, float, 1, 80>;
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* pb = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* pc = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* m = reinterpret_cast<__ubuf__ float*>(32768);
  copy_gm_to_ubuf(m, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  const int height = m[0], width = m[1], stride = m[2], offset = m[3];
  auto* boxes = reinterpret_cast<__gm__ half*>(pb->buffer.addr);
  auto* classes = reinterpret_cast<__gm__ half*>(pc->buffer.addr);
  auto* output = reinterpret_cast<__gm__ float*>(po->buffer.addr);
  auto* box = reinterpret_cast<__ubuf__ half*>(0);
  auto* row = reinterpret_cast<__ubuf__ float*>(24576);
  H80 h;
  F80 value, zero, ones;
  pto::TASSIGN<1024>(h);
  pto::TASSIGN<4096>(value);
  pto::TASSIGN<8192>(zero);
  pto::TASSIGN<12288>(ones);
  for (int pixel = get_block_idx(); pixel < height * width; pixel += get_block_num()) {
    copy_gm_to_ubuf(box, boxes + pixel * 16, 0, 1, 1, 0, 0);
    copy_gm_to_ubuf(h.data(), classes + pixel * 80, 0, 1, 5, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TCVT(value, h, pto::RoundMode::CAST_NONE);
    pto::TEXPANDS(zero, 0.0f);
    pto::TEXPANDS(ones, 1.0f);
    pipe_barrier(PIPE_V);
    pto::TSUB(value, zero, value);
    pipe_barrier(PIPE_V);
    pto::TEXP(value, value);
    pipe_barrier(PIPE_V);
    pto::TADD(value, value, ones);
    pipe_barrier(PIPE_V);
    pto::TDIV(value, ones, value);
    pipe_barrier(PIPE_V);
    pto::TCVT(h, value, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(value, h, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const half ax = static_cast<half>(pixel % width + 0.5f), ay = static_cast<half>(pixel / width + 0.5f);
    const half x1 = static_cast<half>(static_cast<float>(ax) - static_cast<float>(box[0]));
    const half y1 = static_cast<half>(static_cast<float>(ay) - static_cast<float>(box[1]));
    const half x2 = static_cast<half>(static_cast<float>(ax) + static_cast<float>(box[2]));
    const half y2 = static_cast<half>(static_cast<float>(ay) + static_cast<float>(box[3]));
    row[0] = static_cast<float>(static_cast<half>(static_cast<float>(x1) * stride));
    row[1] = static_cast<float>(static_cast<half>(static_cast<float>(y1) * stride));
    row[2] = static_cast<float>(static_cast<half>(static_cast<float>(x2) * stride));
    row[3] = static_cast<float>(static_cast<half>(static_cast<float>(y2) * stride));
    for (int ci = 0; ci < 80; ++ci) row[ci + 4] = value.data()[ci];
    for (int ci = 84; ci < 96; ++ci) row[ci] = 0.0f;
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(output + (offset + pixel) * 96, row, 0, 1, 12, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
}
