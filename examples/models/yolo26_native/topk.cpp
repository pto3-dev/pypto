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

// Deterministic score-descending insertion; ties keep the earlier candidate.
AICORE inline void insert_candidate(__ubuf__ float* scores, __ubuf__ int* indices, float score, int index,
                                    int k) {
  if (score <= scores[k - 1]) return;
  int position = k - 1;
  while (position > 0 && score > scores[position - 1]) {
    scores[position] = scores[position - 1];
    indices[position] = indices[position - 1];
    --position;
  }
  scores[position] = score;
  indices[position] = index;
}
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* pm = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* px = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* m = reinterpret_cast<__ubuf__ float*>(32768);
  copy_gm_to_ubuf(m, reinterpret_cast<__gm__ float*>(pm->buffer.addr), 0, 1, 4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
  const int anchors = m[0], k = 300;
  auto* input = reinterpret_cast<__gm__ float*>(px->buffer.addr);
  auto* output = reinterpret_cast<__gm__ float*>(po->buffer.addr);
  auto* row = reinterpret_cast<__ubuf__ float*>(0);
  auto* anchor_scores = reinterpret_cast<__ubuf__ float*>(1024);
  auto* anchor_indices = reinterpret_cast<__ubuf__ int*>(4096);
  auto* scores = reinterpret_cast<__ubuf__ float*>(8192);
  auto* indices = reinterpret_cast<__ubuf__ int*>(12288);
  auto* detections = reinterpret_cast<__ubuf__ float*>(16384);
  for (int i = 0; i < k; ++i) {
    anchor_scores[i] = -1.0f;
    anchor_indices[i] = 0;
    scores[i] = -1.0f;
    indices[i] = 0;
  }
  for (int anchor = 0; anchor < anchors; ++anchor) {
    copy_gm_to_ubuf(row, input + anchor * 96, 0, 1, 12, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    float maximum = -1.0f;
    for (int ci = 0; ci < 80; ++ci)
      if (row[ci + 4] > maximum) maximum = row[ci + 4];
    insert_candidate(anchor_scores, anchor_indices, maximum, anchor, k);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  }
  for (int selected = 0; selected < k; ++selected) {
    const int anchor = anchor_indices[selected];
    copy_gm_to_ubuf(row, input + anchor * 96, 0, 1, 12, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    for (int ci = 0; ci < 80; ++ci) insert_candidate(scores, indices, row[ci + 4], anchor * 80 + ci, k);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  }
  for (int i = 0; i < k; ++i) {
    const int anchor = indices[i] / 80, ci = indices[i] % 80;
    copy_gm_to_ubuf(row, input + anchor * 96, 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    for (int c = 0; c < 4; ++c) detections[i * 6 + c] = row[c];
    detections[i * 6 + 4] = scores[i];
    detections[i * 6 + 5] = static_cast<float>(ci);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  copy_ubuf_to_gm(output, detections, 0, 1, 225, 0, 0);
}
