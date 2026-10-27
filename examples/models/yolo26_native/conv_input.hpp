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

#ifndef EXAMPLES_MODELS_YOLO26_NATIVE_CONV_INPUT_HPP_
#define EXAMPLES_MODELS_YOLO26_NATIVE_CONV_INPUT_HPP_
#include <pto/pto-inst.hpp>

AICORE inline void stage_nc1hwc0_rows(__cbuf__ half* image, __gm__ half* x, int y0, int y1, int iw, int ci,
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

#endif  // EXAMPLES_MODELS_YOLO26_NATIVE_CONV_INPUT_HPP_
