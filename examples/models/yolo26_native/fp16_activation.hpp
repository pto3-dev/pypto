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
#ifndef EXAMPLES_MODELS_YOLO26_NATIVE_FP16_ACTIVATION_HPP_
#define EXAMPLES_MODELS_YOLO26_NATIVE_FP16_ACTIVATION_HPP_

#include <pto/pto-inst.hpp>

// A1 scalar vector instructions use the normal 64-lane FP32 mask. Issue a
// complete tile in repeated instructions rather than serializing each repeat.
enum class SigmoidScalar { Add, Multiply, Absolute };
template <SigmoidScalar Operation>
AICORE inline void sigmoid_scalar_repeat(__ubuf__ float* dst, __ubuf__ float* src, float scalar,
                                         unsigned repeats) {
  if constexpr (Operation == SigmoidScalar::Add) vadds(dst, src, scalar, repeats, 1, 1, 8, 8);
  if constexpr (Operation == SigmoidScalar::Multiply) vmuls(dst, src, scalar, repeats, 1, 1, 8, 8);
  if constexpr (Operation == SigmoidScalar::Absolute) vabs(dst, src, repeats, 1, 1, 8, 8);
}
template <SigmoidScalar Operation, unsigned Elements>
AICORE inline void sigmoid_scalar(__ubuf__ float* dst, __ubuf__ float* src, float scalar = 0) {
  set_mask_norm();
  if constexpr (Elements / 64 > 0) {
    set_vector_mask(0, UINT64_MAX);
    sigmoid_scalar_repeat<Operation>(dst, src, scalar, Elements / 64);
  }
  if constexpr (Elements % 64 > 0) {
    set_vector_mask(0, (UINT64_C(1) << (Elements % 64)) - 1);
    sigmoid_scalar_repeat<Operation>(dst + Elements / 64 * 64, src + Elements / 64 * 64, scalar, 1);
  }
  pipe_barrier(PIPE_V);
}

// A1 supports FP32 vector/vector min and max, but not FP32 scalar min/max.
// The temporary constant span can alias another scratch whose prior value is dead.
template <unsigned Elements>
AICORE inline void sigmoid_clamp(__ubuf__ float* dst, __ubuf__ float* src, float lower, float upper,
                                 __ubuf__ float* constant) {
  pto::detail::A1VectorInstruction<pto::Op::TEXPANDS, Elements>(constant, constant, constant, upper);
  pipe_barrier(PIPE_V);
  set_mask_norm();
  if constexpr (Elements / 64 > 0) {
    set_vector_mask(0, UINT64_MAX);
    vmin(dst, src, constant, Elements / 64, 1, 1, 1, 8, 8, 8);
  }
  if constexpr (Elements % 64 > 0) {
    constexpr unsigned offset = Elements / 64 * 64;
    set_vector_mask(0, (UINT64_C(1) << (Elements % 64)) - 1);
    vmin(dst + offset, src + offset, constant + offset, 1, 1, 1, 1, 8, 8, 8);
  }
  pipe_barrier(PIPE_V);
  pto::detail::A1VectorInstruction<pto::Op::TEXPANDS, Elements>(constant, constant, constant, lower);
  pipe_barrier(PIPE_V);
  pto::detail::A1VectorInstruction<pto::Op::TMAX, Elements>(dst, dst, constant);
  pipe_barrier(PIPE_V);
}

// A1 vexp(-x) is one FP32 ULP low at the single FP16 input -15/2048,
// crossing the later sigmoid tie. Select the exact 2^-23 correction in
// FP16 (a representable subnormal), then widen before the unchanged divide
// and sigmoid/product narrowing. m200 tensor-select supports FP16 here.
// The temporary E-float span holds two disjoint E-half constants; packed
// predicates plus their address word fit the separate E-float mask span.
AICORE inline __attribute__((always_inline)) void sigmoid_select_mask_address(__ubuf__ uint32_t* address,
                                                                              __ubuf__ unsigned char* bits) {
  set_vector_mask(0, UINT64_C(0xff));
  vector_dup(address, static_cast<uint32_t>(reinterpret_cast<uint64_t>(bits)), 1, 1, 1, 8, 0);
  pipe_barrier(PIPE_V);
  set_cmpmask(address);
  pipe_barrier(PIPE_V);
}

template <unsigned Elements>
AICORE inline void correct_fp16_sigmoid_exp_boundary(__ubuf__ float* exponent, __ubuf__ float* value,
                                                     __ubuf__ float* mask, __ubuf__ float* temporary) {
  if constexpr (Elements % 64 == 0) {
    // Compare widened, exactly representable FP16 values and expand packed
    // predicate bits through m200's FP16-only tensor-select mode. Temporary
    // holds two disjoint E-half constants, inside its original E-float span.
    auto* bits = reinterpret_cast<__ubuf__ unsigned char*>(mask);
    auto* address = reinterpret_cast<__ubuf__ uint32_t*>(bits + ((Elements + 255) / 256) * 32);
    auto* positive = reinterpret_cast<__ubuf__ half*>(temporary);
    auto* zero = positive + Elements;
    set_mask_norm();
    set_vector_mask(0, UINT64_MAX);
    vcmpvs_eq(bits, value, -15.0f / 2048, Elements / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    if constexpr (Elements / 128 > 0) {
      vector_dup(positive, static_cast<half>(0x1p-23f), Elements / 128, 1, 1, 8, 0);
      vector_dup(zero, static_cast<half>(0), Elements / 128, 1, 1, 8, 0);
    }
    if constexpr (Elements % 128 > 0) {
      set_vector_mask(0, UINT64_MAX);
      vector_dup(positive + Elements / 128 * 128, static_cast<half>(0x1p-23f), 1, 1, 1, 8, 0);
      vector_dup(zero + Elements / 128 * 128, static_cast<half>(0), 1, 1, 1, 8, 0);
    }
    pipe_barrier(PIPE_V);
    sigmoid_select_mask_address(address, bits);
    if constexpr (Elements % 128 == 0) {
      set_vector_mask(UINT64_MAX, UINT64_MAX);
      vsel(positive, positive, zero, Elements / 128, 1, 1, 1, 8, 8, 8, 2, false, false);
    } else {
      // Count mode consumes the final 64 predicate bits from the aligned base.
      // A separate tail selection would need a potentially unaligned mask address.
      set_mask_count();
      set_vector_mask(0, Elements);
      vsel(positive, positive, zero, 1, 1, 1, 1, 8, 8, 8, 2, false, false);
      pipe_barrier(PIPE_V);
      set_mask_norm();
    }
    pipe_barrier(PIPE_V);
    set_vector_mask(0, UINT64_MAX);
    vconv_f162f32(mask, positive, Elements / 64, 1, 1, 8, 4);
    pipe_barrier(PIPE_V);
    pto::detail::A1VectorInstruction<pto::Op::TADD, Elements>(exponent, exponent, mask);
    pipe_barrier(PIPE_V);
  } else {
    // Partial FP32 repeats retain the original bounded distance/clamp mask.
    sigmoid_scalar<SigmoidScalar::Add, Elements>(mask, value, 15.0f / 2048);
    sigmoid_scalar<SigmoidScalar::Absolute, Elements>(mask, mask);
    sigmoid_scalar<SigmoidScalar::Multiply, Elements>(mask, mask, -524288.0f);
    sigmoid_scalar<SigmoidScalar::Add, Elements>(mask, mask, 1.0f);
    sigmoid_clamp<Elements>(mask, mask, 0.0f, 1.0f, temporary);
    sigmoid_scalar<SigmoidScalar::Multiply, Elements>(mask, mask, 0x1p-23f);
    pto::detail::A1VectorInstruction<pto::Op::TADD, Elements>(exponent, exponent, mask);
    pipe_barrier(PIPE_V);
  }
}

// Refine exp(-x) near zero before the FP16 sigmoid narrowing boundary. A1 vexp can
// shift this boundary by one FP32 ULP. Evaluate exp(-x) with a degree-five
// Horner polynomial (truncation error < 9e-11 for |x| <= 1/16).
// Clamp the polynomial input first: both blend sources must stay finite even
// when the original hardware exponent overflows. All scratch spans are dead
// at this epilogue and distinct from exponent and the original widened FP16 x.
template <unsigned Elements>
AICORE inline void refine_fp16_sigmoid_exp(__ubuf__ float* exponent, __ubuf__ float* value,
                                           __ubuf__ float* polynomial, __ubuf__ float* clamped,
                                           __ubuf__ float* mask) {
  sigmoid_clamp<Elements>(clamped, value, -0.0625f, 0.0625f, mask);
  pto::detail::A1VectorInstruction<pto::Op::TEXPANDS, Elements>(polynomial, polynomial, polynomial,
                                                                -1.0f / 120);
  pipe_barrier(PIPE_V);
  constexpr float coefficients[] = {1.0f / 24, -1.0f / 6, 0.5f, -1.0f, 1.0f};
  for (float coefficient : coefficients) {
    pto::detail::A1VectorInstruction<pto::Op::TMUL, Elements>(polynomial, polynomial, clamped);
    pipe_barrier(PIPE_V);
    sigmoid_scalar<SigmoidScalar::Add, Elements>(polynomial, polynomial, coefficient);
  }
  // FP16 inputs immediately below 1/16 are separated by 2^-15. Scaling by
  // 2^16 makes this exactly a binary mask: one strictly inside, zero outside.
  // This avoids an A1 comparison-register update and vsel for every 64 lanes.
  sigmoid_scalar<SigmoidScalar::Absolute, Elements>(mask, value);
  sigmoid_scalar<SigmoidScalar::Multiply, Elements>(mask, mask, -1.0f);
  sigmoid_scalar<SigmoidScalar::Add, Elements>(mask, mask, 0.0625f);
  sigmoid_scalar<SigmoidScalar::Multiply, Elements>(mask, mask, 65536.0f);
  sigmoid_clamp<Elements>(mask, mask, 0.0f, 1.0f, clamped);
  // Cap only exponents whose reciprocal already rounds to FP16 zero. This
  // keeps both blend sources finite (0 * infinity would otherwise be NaN),
  // without changing the separately rounded FP16 sigmoid/product contract.
  sigmoid_clamp<Elements>(exponent, exponent, 0.0f, 0x1p120f, clamped);
  pto::detail::A1VectorInstruction<pto::Op::TSUB, Elements>(polynomial, polynomial, exponent);
  pipe_barrier(PIPE_V);
  pto::detail::A1VectorInstruction<pto::Op::TMUL, Elements>(polynomial, polynomial, mask);
  pipe_barrier(PIPE_V);
  pto::detail::A1VectorInstruction<pto::Op::TADD, Elements>(exponent, exponent, polynomial);
  pipe_barrier(PIPE_V);
}
#endif  // EXAMPLES_MODELS_YOLO26_NATIVE_FP16_ACTIVATION_HPP_
