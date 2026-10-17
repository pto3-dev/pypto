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
// Sort 512 padded FP32 score/index pairs with A1's 16-proposal primitive.
// A1 proposal records have eight fields; score is field 4 and index field 5.
// These two 16 KiB banks end exactly where the dense input cache starts.
AICORE inline void sort_candidates(__ubuf__ float* scores, __ubuf__ int* indices, int k) {
  auto* proposals = reinterpret_cast<__ubuf__ float*>(32768);
  auto* temporary = reinterpret_cast<__ubuf__ float*>(49152);
  for (int i = k; i < 512; ++i) {
    scores[i] = -__builtin_inff();
    indices[i] = 2147483647;
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_mask_norm();
  set_vector_mask(UINT64_MAX, UINT64_MAX);
  uint64_t fields[8] = {0, 0, 0, 0, reinterpret_cast<uint64_t>(scores), reinterpret_cast<uint64_t>(indices),
                        0, 0};
  set_va_reg_sb(VA0, fields);
  scatter_vconcat_f32(proposals, VA0, 32, 0b110000);
  pipe_barrier(PIPE_V);
  vbitsort(temporary, proposals, 32);
  pipe_barrier(PIPE_V);
  for (int group = 0; group < 8; ++group) {
    auto* base = temporary + group * 512;
    __ubuf__ float* lists[4] = {base, base + 128, base + 256, base + 384};
    vmrgsort4(proposals + group * 512, lists, 1, 16, 16, 16, 16, false, 0b1111);
  }
  pipe_barrier(PIPE_V);
  for (int group = 0; group < 2; ++group) {
    auto* base = proposals + group * 2048;
    __ubuf__ float* lists[4] = {base, base + 512, base + 1024, base + 1536};
    vmrgsort4(temporary + group * 2048, lists, 1, 64, 64, 64, 64, false, 0b1111);
  }
  pipe_barrier(PIPE_V);
  __ubuf__ float* lists[4] = {temporary, temporary + 2048, temporary, temporary};
  vmrgsort4(proposals, lists, 1, 256, 256, 0, 0, false, 0b0011);
  set_flag(PIPE_V, PIPE_S, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
  for (int i = 0; i < k; ++i) {
    scores[i] = proposals[i * 8 + 4];
    indices[i] = reinterpret_cast<__ubuf__ int*>(proposals)[i * 8 + 5];
  }
  // Proposal instructions compare scores only. Restore ascending traversal
  // indices within every equal-score run without changing any score bits.
  for (int first = 0; first < k;) {
    int last = first + 1;
    while (last < k && scores[last] == scores[first]) ++last;
    const int count = last - first;
    for (int step = count / 2 - 1; step >= -(count - 1); --step) {
      const bool extract = step < 0;
      const int size = extract ? count + step : count;
      const int root = extract ? 0 : step;
      if (extract) {
        const float score = scores[first + size];
        const int index = indices[first + size];
        scores[first + size] = scores[first];
        indices[first + size] = indices[first];
        scores[first] = score;
        indices[first] = index;
      }
      const float score = scores[first + root];
      const int index = indices[first + root];
      int position = root;
      while (position * 2 + 1 < size) {
        int child = position * 2 + 1;
        if (child + 1 < size && indices[first + child + 1] > indices[first + child]) ++child;
        if (indices[first + child] <= index) break;
        scores[first + position] = scores[first + child];
        indices[first + position] = indices[first + child];
        position = child;
      }
      scores[first + position] = score;
      indices[first + position] = index;
    }
    first = last;
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
  auto* row_buffer = reinterpret_cast<__ubuf__ float*>(0);
  auto* cached = reinterpret_cast<__ubuf__ float*>(65536);
  const bool resident = anchors <= 384;
  // At most 144 KiB ends at UB byte 212992, clear of heap and output slots.
  if (resident) {
    copy_gm_to_ubuf(cached, input, 0, 1, anchors * 12, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  }
  auto* row = row_buffer;
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
  if (resident) {
    // Reuse the later proposal banks for two sets of per-row maxima. The
    // largest resident case needs 12 KiB per bank. Repeat stride 12 is the
    // physical [96] input row. Destination stride four value/index pairs
    // places each result in a separate 32-byte block.
    auto* low = reinterpret_cast<__ubuf__ float*>(32768);
    auto* high = reinterpret_cast<__ubuf__ float*>(49152);
    set_mask_norm();
    for (int first = 0; first < anchors; first += 255) {
      const int count = anchors - first < 255 ? anchors - first : 255;
      set_vector_mask(0, UINT64_MAX ^ UINT64_C(15));
      vcmax(low + first * 8, cached + first * 96, count, 4, 1, 12, false, false, false);
      pipe_barrier(PIPE_V);
      set_vector_mask(0, UINT64_C(0xfffff));
      vcmax(high + first * 8, cached + first * 96 + 64, count, 4, 1, 12, false, false, false);
      pipe_barrier(PIPE_V);
    }
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (int anchor = 0; anchor < anchors; ++anchor) {
      anchor_scores[anchor] = low[anchor * 8] > high[anchor * 8] ? low[anchor * 8] : high[anchor * 8];
      anchor_indices[anchor] = anchor;
    }
    // Repair ties before truncating to 300, including tied boundary members.
    sort_candidates(anchor_scores, anchor_indices, anchors);
  } else {
    for (int anchor = 0; anchor < anchors; ++anchor) {
      row = resident ? cached + anchor * 96 : row_buffer;
      if (!resident) {
        copy_gm_to_ubuf(row, input + anchor * 96, 0, 1, 12, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      }
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
      if (!resident) {
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      }
    }
    sort_candidates(anchor_scores, anchor_indices, k);
  }
  for (int selected = 0; selected < k; ++selected) {
    // Sorted anchor maxima bound every score in this and later rows. Equal
    // scores cannot replace earlier traversal indices already in the heap.
    if (!worse(scores[0], indices[0], anchor_scores[selected], selected * 80)) break;
    const int anchor = anchor_indices[selected];
    row = resident ? cached + anchor * 96 : row_buffer;
    if (!resident) {
      copy_gm_to_ubuf(row, input + anchor * 96, 0, 1, 12, 0, 0);
      set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    }
    for (int ci = 0; ci < 80; ++ci) heap_candidate(scores, indices, row[ci + 4], selected * 80 + ci, k);
    if (!resident) {
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
  }
  sort_candidates(scores, indices, k);
  for (int i = 0; i < k; ++i) {
    const int anchor = anchor_indices[indices[i] / 80], ci = indices[i] % 80;
    row = resident ? cached + anchor * 96 : row_buffer;
    if (!resident) {
      copy_gm_to_ubuf(row, input + anchor * 96, 0, 1, 1, 0, 0);
      set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    }
    for (int c = 0; c < 4; ++c) detections[i * 6 + c] = row[c];
    detections[i * 6 + 4] = scores[i];
    detections[i * 6 + 5] = static_cast<float>(ci);
    if (!resident) {
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
  }
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  copy_ubuf_to_gm(output, detections, 0, 1, 225, 0, 0);
}
