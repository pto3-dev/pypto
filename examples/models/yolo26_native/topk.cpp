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

// A lower score, then a later traversal index, is the worse heap candidate.
AICORE inline bool worse(float a, int ai, float b, int bi) { return a < b || (a == b && ai > bi); }
AICORE inline void sift_candidate(__ubuf__ float* scores, __ubuf__ int* indices, float score, int index,
                                  int size) {
  int position = 0;
  while (position * 2 + 1 < size) {
    int child = position * 2 + 1;
    if (child + 1 < size && worse(scores[child + 1], indices[child + 1], scores[child], indices[child]))
      ++child;
    if (!worse(scores[child], indices[child], score, index)) break;
    scores[position] = scores[child];
    indices[position] = indices[child];
    position = child;
  }
  scores[position] = score;
  indices[position] = index;
}
AICORE inline void heap_candidate(__ubuf__ float* scores, __ubuf__ int* indices, float score, int index,
                                  int k) {
  if (!worse(scores[0], indices[0], score, index)) return;
  sift_candidate(scores, indices, score, index, k);
}
AICORE inline void sort_candidates(__ubuf__ float* scores, __ubuf__ int* indices, int k) {
  for (int end = k - 1; end > 0; --end) {
    const float score = scores[end];
    const int index = indices[end];
    scores[end] = scores[0];
    indices[end] = indices[0];
    sift_candidate(scores, indices, score, index, end);
  }
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
  auto* maxima = reinterpret_cast<__ubuf__ float*>(512);
  auto* anchor_scores = reinterpret_cast<__ubuf__ float*>(1024);
  auto* anchor_indices = reinterpret_cast<__ubuf__ int*>(4096);
  auto* scores = reinterpret_cast<__ubuf__ float*>(8192);
  auto* indices = reinterpret_cast<__ubuf__ int*>(12288);
  auto* detections = reinterpret_cast<__ubuf__ float*>(16384);
  for (int i = 0; i < k; ++i) {
    anchor_scores[i] = -1.0f;
    anchor_indices[i] = 2147483647;
    scores[i] = -1.0f;
    indices[i] = 2147483647;
  }
  for (int anchor = 0; anchor < anchors; ++anchor) {
    copy_gm_to_ubuf(row, input + anchor * 96, 0, 1, 12, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_mask_norm();
    // Ignore four box coordinates and reduce only the 80 class scores.
    set_vector_mask(0, UINT64_MAX ^ UINT64_C(15));
    vcmax(maxima, row, 1, 1, 1, 8, false, false, false);
    pipe_barrier(PIPE_V);
    set_vector_mask(0, UINT64_C(0xfffff));
    vcmax(maxima + 8, row + 64, 1, 1, 1, 8, false, false, false);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    const float maximum = maxima[0] > maxima[8] ? maxima[0] : maxima[8];
    heap_candidate(anchor_scores, anchor_indices, maximum, anchor, k);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  }
  sort_candidates(anchor_scores, anchor_indices, k);
  for (int selected = 0; selected < k; ++selected) {
    const int anchor = anchor_indices[selected];
    copy_gm_to_ubuf(row, input + anchor * 96, 0, 1, 12, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    for (int ci = 0; ci < 80; ++ci) heap_candidate(scores, indices, row[ci + 4], selected * 80 + ci, k);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  }
  sort_candidates(scores, indices, k);
  for (int i = 0; i < k; ++i) {
    const int anchor = anchor_indices[indices[i] / 80], ci = indices[i] % 80;
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
