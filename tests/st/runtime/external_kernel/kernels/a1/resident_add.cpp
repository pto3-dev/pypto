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

using Data = float;
using Tile = pto::Tile<pto::TileType::Vec, Data, 1, 256>;
using Global = pto::GlobalTensor<Data, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<1, 1, 1, 256, 1>>;
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  const unsigned offset = get_block_idx() * 256;
  auto* lhs = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* rhs = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* out = reinterpret_cast<__gm__ Tensor*>(args[2]);
  Tile a, b, c;
  pto::TASSIGN<0>(a);
  pto::TASSIGN<4096>(b);
  pto::TASSIGN<8192>(c);
  Global ga(reinterpret_cast<__gm__ Data*>(lhs->buffer.addr) + offset);
  Global gb(reinterpret_cast<__gm__ Data*>(rhs->buffer.addr) + offset);
  Global gc(reinterpret_cast<__gm__ Data*>(out->buffer.addr) + offset);
  pto::TLOAD(a, ga);
  pto::TLOAD(b, gb);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#ifdef RESIDENT_MULTIPLY
  pto::TMUL(c, a, b);
#else
  pto::TADD(c, a, b);
#endif
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  pto::TSTORE(gc, c);
}
