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

// Residual transforms require individually rounded FP32 operations.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
struct Pair {
  float hi;
  float lo;
};
// Error-free transforms retain the residual discarded by one FP32 operation.
AICORE Pair add(Pair a, Pair b) {
  const float sum = a.hi + b.hi;
  const float v = sum - a.hi;
  const float error = (a.hi - (sum - v)) + (b.hi - v);
  const float tail = error + (a.lo + b.lo);
  const float hi = sum + tail;
  return {hi, tail - (hi - sum)};
}
AICORE Pair multiply(Pair a, Pair b) {
  const float ca = 4097.0f * a.hi, cb = 4097.0f * b.hi;
  const float ah = ca - (ca - a.hi), bh = cb - (cb - b.hi);
  const float al = a.hi - ah, bl = b.hi - bh;
  const float product = a.hi * b.hi;
  const float error = ((ah * bh - product) + ah * bl + al * bh) + al * bl;
  const float tail = error + (a.hi * b.lo + a.lo * b.hi);
  const float hi = product + tail;
  return {hi, tail - (hi - product)};
}
AICORE Pair exponential(float x) {
  // Smaller terms are below pair precision when the normalization sum is at least one.
  if (x < -80.0f) return {0.0f, 0.0f};
  const int n = static_cast<int>(x * 1.4426950408889634f - 0.5f);
  Pair r =
      add({x, 0.0f}, multiply({static_cast<float>(-n), 0.0f}, {0.6931471824645996f, -1.904654323148236e-9f}));
  // Degree-13 Taylor polynomial on |r| <= ln(2)/2; coefficients are high/residual pairs.
  Pair p = {1.6059044372074283e-10f, -5.3525265115627256e-18f};
  p = add(multiply(p, r), {2.0876755879584152e-09f, 1.1082839809204342e-16f});
  p = add(multiply(p, r), {2.5052107943679403e-08f, 4.4176230446483665e-16f});
  p = add(multiply(p, r), {2.755731998149713e-07f, -7.5751122090511949e-15f});
  p = add(multiply(p, r), {2.7557318844628753e-06f, 3.7935712242972291e-14f});
  p = add(multiply(p, r), {2.4801587642286904e-05f, -3.4069960936668198e-13f});
  p = add(multiply(p, r), {0.00019841270113829523f, -2.7255968749334558e-12f});
  p = add(multiply(p, r), {0.0013888889225199819f, -3.3631094437103215e-11f});
  p = add(multiply(p, r), {0.0083333337679505348f, -4.3461720333759502e-10f});
  p = add(multiply(p, r), {0.041666667908430099f, -1.2417634698280722e-09f});
  p = add(multiply(p, r), {0.1666666716337204f, -4.9670538793122887e-09f});
  p = add(multiply(p, r), {0.5f, 0.0f});
  p = add(multiply(p, r), {1.0f, 0.0f});
  p = add(multiply(p, r), {1.0f, 0.0f});
  float scale = 1.0f;
  for (int i = 0; i < -n; ++i) scale *= 0.5f;
  return {p.hi * scale, p.lo * scale};
}
}  // namespace

using H = pto::Tile<pto::TileType::Vec, half, 1, 16>;
using F = pto::Tile<pto::TileType::Vec, float, 1, 16>;
extern "C" AICORE void kernel_entry(__gm__ int64_t* args) {
  auto* px = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* po = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* x = reinterpret_cast<__gm__ half*>(px->buffer.addr);
  auto* out = reinterpret_cast<__gm__ half*>(po->buffer.addr);
  H hx, ho;
  F values, exp_hi, exp_lo;
  pto::TASSIGN<0>(hx);
  pto::TASSIGN<1024>(ho);
  pto::TASSIGN<2048>(values);
  pto::TASSIGN<4096>(exp_hi);
  pto::TASSIGN<5120>(exp_lo);
  for (int row = get_block_idx(); row < 16; row += get_block_num()) {
    copy_gm_to_ubuf(hx.data(), x + row * 16, 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TCVT(values, hx, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_S, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
    float maximum = values.data()[0];
    for (int col = 1; col < 16; ++col)
      if (values.data()[col] > maximum) maximum = values.data()[col];
    Pair total = {0.0f, 0.0f};
    for (int col = 0; col < 16; ++col) {
      const Pair e = exponential(values.data()[col] - maximum);
      exp_hi.data()[col] = e.hi;
      exp_lo.data()[col] = e.lo;
      total = add(total, e);
    }
    for (int col = 0; col < 16; ++col) {
      const Pair e = {exp_hi.data()[col], exp_lo.data()[col]};
      const float q = e.hi / total.hi;
      const Pair residual = add(e, multiply(total, {-q, 0.0f}));
      const Pair normalized = add({q, 0.0f}, {(residual.hi + residual.lo) / total.hi, 0.0f});
      values.data()[col] = normalized.hi + normalized.lo;
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID6);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID6);
    pto::TCVT(ho, values, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(out + row * 16, ho.data(), 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
}
