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
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* meta = reinterpret_cast<__ubuf__ float*>(131072);
  copy_gm_to_ubuf(meta, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  __gm__ half* inputs[4];
  for (int i = 0; i < 4; ++i) {
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[i + 1]);
    inputs[i] = reinterpret_cast<__gm__ half*>(tensor->buffer.addr);
  }
  auto* output = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  const int pixels = meta[0], channels = meta[1], sources = meta[2];
  // Two 64 KiB UB banks end at metadata; cap the row panel to one bank.
  int rows = 16;
  while (rows > 1 && (rows * channels > 32768 || (pixels + rows - 1) / rows < get_block_num())) rows /= 2;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  int slot = 0;
  for (int first = get_block_idx() * rows; first < pixels; first += get_block_num() * rows) {
    const int count = pixels - first < rows ? pixels - first : rows;
    auto* panel = reinterpret_cast<__ubuf__ half*>(slot * 65536);
    if (slot == 0)
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    else
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    int destination = 0;
    for (int i = 0; i < sources; ++i) {
      const int width = meta[4 + i], stride = meta[8 + i], offset = meta[12 + i];
      copy_gm_to_ubuf(panel + destination, inputs[i] + static_cast<int64_t>(first) * stride + offset, 0,
                      count, width / 16, (stride - width) / 16, (channels - width) / 16);
      destination += width;
    }
    if (slot == 0) {
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    } else {
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    }
    copy_ubuf_to_gm(output + static_cast<int64_t>(first) * channels, panel, 0, 1, count * channels / 16, 0,
                    0);
    if (slot == 0)
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    else
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    slot ^= 1;
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}
