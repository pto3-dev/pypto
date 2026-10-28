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

// Match the parent's 16x400 reduction shape, FP32 operators and FP16 casts.
// The device loop coalesces rows; it never combines reductions across rows.
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* px = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* x = reinterpret_cast<__gm__ half*>(px->buffer.addr) + px->start_offset;
  auto* output = reinterpret_cast<__gm__ half*>(po->buffer.addr) + po->start_offset;
  using F =
      pto::Tile<pto::TileType::Vec, float, 16, 400, pto::BLayout::RowMajor, 16, 400, pto::SLayout::NoneBox>;
  using R = pto::Tile<pto::TileType::Vec, float, 16, 1, pto::BLayout::ColMajor, 16, 1, pto::SLayout::NoneBox>;
  using H =
      pto::Tile<pto::TileType::Vec, half, 16, 400, pto::BLayout::RowMajor, 16, 400, pto::SLayout::NoneBox>;
  // The two FP32 matrices use bytes [0, 51200), followed by the reduced
  // column at 51200 and disjoint FP16 DMA spans at 76800/89600. The last
  // span ends at 102400, inside the conservative 128 KiB vector budget.
  F value, scratch;
  R reduced;
  H input, rounded;
  pto::TASSIGN<0>(value);
  pto::TASSIGN<25600>(scratch);
  pto::TASSIGN<51200>(reduced);
  pto::TASSIGN<76800>(input);
  pto::TASSIGN<89600>(rounded);
  set_mask_norm();
  set_vector_mask(UINT64_MAX, UINT64_MAX);
  for (int row = get_block_idx() * 16; row < 400; row += get_block_num() * 16) {
    copy_gm_to_ubuf(input.data(), x + row * 400, 0, 1, 400, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TCVT(value, input, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TROWMAX(reduced, value, scratch);
    pipe_barrier(PIPE_V);
    pto::TROWEXPAND(scratch, reduced);
    pto::TSUB(value, value, scratch);
    pipe_barrier(PIPE_V);
    pto::TEXP(value, value);
    pto::TROWSUM(reduced, value, scratch);
    pipe_barrier(PIPE_V);
    pto::TROWEXPAND(scratch, reduced);
    pto::TDIV(value, value, scratch);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, value, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(output + row * 400, rounded.data(), 0, 1, 400, 0, 0);
    // Preserve the parent's completed-row ownership boundary before reuse.
    pipe_barrier(PIPE_ALL);
  }
}
