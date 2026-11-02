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

#ifndef CACHE_ROWS
#define CACHE_ROWS 2
#endif
#ifndef CACHE_BLOCKS
#define CACHE_BLOCKS CACHE_INSTANCES
#endif
#ifndef CACHE_ARG0
#define CACHE_ARG0 0
#endif
#ifndef CACHE_ARG1
#define CACHE_ARG1 1
#endif

#include <bit>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

int main() {
  using Data = CACHE_CTYPE;
  using Word = std::conditional_t<sizeof(Data) == 2, uint16_t, uint32_t>;
  constexpr uint32_t instances = CACHE_INSTANCES;
  constexpr size_t guard = 16, payload = CACHE_ROWS * 16, shift = payload + guard;
  const uint32_t shape[2] = {CACHE_ROWS, 16};
  const Word sentinel = static_cast<Word>(0xa55ac33cU);
  // Raw words include NaNs and signed extremes: cache selection must only copy
  // bytes. Distinct ordinary/alias regions expose incorrect global rebasing.
  for (uint32_t dataset : {0U, 1U}) {
    for (const uint64_t alias : {uint64_t{0}, shift * sizeof(Data)}) {
      std::vector<std::vector<Data>> sources(instances), outputs(instances);
      std::vector<Tensor> tensors(2 * instances);
      std::vector<int64_t> slots(2 * instances + (CACHE_BYPASS ? 1 : 0));
      for (uint32_t image = 0; image < instances; ++image) {
        auto& source = sources[image];
        source.resize(2 * payload + 3 * guard);
        for (size_t index = 0; index < source.size(); ++index)
          source[index] = std::bit_cast<Data>(
              static_cast<Word>(index * 1664525U + image * 0x10203U + dataset * 0x234567U + 1013904223U));
        outputs[image].assign(payload + 2 * guard, std::bit_cast<Data>(sentinel));
        tensors[2 * image].init_external(source.data(), source.size() * sizeof(Data), shape, 2, CACHE_DTYPE,
                                         0);
        tensors[2 * image + 1].init_external(outputs[image].data(), outputs[image].size() * sizeof(Data),
                                             shape, 2, CACHE_DTYPE, 0);
        for (uint32_t index = 2 * image; index < 2 * image + 2; ++index) {
          tensors[index].start_offset = guard;
          const uint32_t order[2] = {CACHE_ARG0, CACHE_ARG1};
          slots[index] = reinterpret_cast<int64_t>(&tensors[2 * image + order[index % 2]]);
        }
      }
      // The runtime appends one footer after every image signature, not after
      // each individual descriptor pair. Descriptors and tensor data are read-only.
      if (CACHE_BYPASS) slots.back() = static_cast<int64_t>(alias);
      const auto original_sources = sources;
      const auto original_tensors = tensors;
      const auto original_slots = slots;
      for (uint32_t core = 0; core < CACHE_BLOCKS; ++core) {
        pto::cpu_sim::ScopedExecutionContext context(core, 0);
        PyptoA1TensorEntry(slots.data());
      }
      if (std::memcmp(tensors.data(), original_tensors.data(), tensors.size() * sizeof(Tensor))) return 1;
      if (slots != original_slots) return 2;
      for (uint32_t image = 0; image < instances; ++image) {
        if (std::memcmp(sources[image].data(), original_sources[image].data(),
                        sources[image].size() * sizeof(Data)))
          return 3;
        for (size_t index = 0; index < outputs[image].size(); ++index) {
          Word expected = sentinel;
          if (index >= guard && index < guard + payload) {
            const size_t logical = index - guard;
            const size_t selected_alias = CACHE_BYPASS && (logical / 16) % 2 == 0 ? alias / sizeof(Data) : 0;
            expected = std::bit_cast<Word>(original_sources[image][guard + logical + selected_alias]);
          }
          if (std::bit_cast<Word>(outputs[image][index]) != expected) return 4;
        }
      }
    }
  }
}
