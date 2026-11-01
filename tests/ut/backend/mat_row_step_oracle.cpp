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
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int kGuard = 16;
constexpr int kSourceRows = AS_SR, kSourceColumns = AS_SC;
constexpr int kRows = AS_DR, kColumns = AS_DC;
constexpr int kCopies = AS_COPIES, kCores = AS_CORES;
constexpr int kMatrix = kRows * kColumns;
constexpr int kSource = kCores * AS_SOURCE_ROWS * AS_SOURCE_COLS;
constexpr int kProbeRows = AS_PROBE_ROWS, kProbeElements = kProbeRows * 16;
constexpr bool kSparse = AS_SPARSE;
constexpr int kProbeRow = AS_ROW / 16 * 16;
using Matrix = pto::Tile<pto::TileType::Mat, half, kRows, kColumns, pto::BLayout::ColMajor, kRows, kColumns,
                         pto::SLayout::RowMajor, 512>;
static_assert(AS_STEP >= 1 && AS_STEP <= 65536 && AS_ROW >= 0 && AS_COL >= 0);
static_assert(AS_ROW + (kCopies - 1) * AS_ORIGIN_STEP + (kSourceRows - 1) * AS_STEP < kRows);
static_assert(AS_COL + kSourceColumns <= kColumns);

half sample(int index, int dataset, unsigned factor) {
  if (dataset == 2) {
    const unsigned value = (static_cast<unsigned>(index) * factor) % 63488;
    return std::bit_cast<half>(static_cast<uint16_t>(value < 31744 ? value : (value - 31744) | 0x8000));
  }
  return half(static_cast<float>((index * static_cast<int>(factor)) % 101 - 50) /
              (dataset == 0 ? 16.0f : 37.0f));
}
}  // namespace

int main() {
  for (int dataset = 0; dataset < 3; ++dataset) {
    std::vector<half> initial(kCores * kMatrix + 2 * kGuard, half(7));
    std::vector<half> source(kSource + 2 * kGuard, half(11));
    std::vector<half> weights(256 + 2 * kGuard, half(13));
    std::vector<float> output(kCores * kProbeElements + 2 * kGuard, 19.0f);
    auto expected = output;
    for (int i = 0; i < kCores * kMatrix; ++i) initial[kGuard + i] = sample(i, dataset, 73);
    for (int i = 0; i < kSource; ++i) source[kGuard + i] = sample(i, dataset, 41);
    for (int i = 0; i < 256; ++i) weights[kGuard + i] = half(i / 16 == i % 16 ? 1 : 0);
    const auto initialBefore = initial, sourceBefore = source, weightsBefore = weights;
    std::array<Tensor, 4> tensors;
    std::array<int64_t, 4> args{};
    const uint32_t shapes[4][2] = {{kCores * kRows, kColumns},
                                   {kCores * AS_SOURCE_ROWS, AS_SOURCE_COLS},
                                   {16, 16},
                                   {kCores * kProbeRows, 16}};
    void* addresses[4] = {initial.data(), source.data(), weights.data(), output.data()};
    const uint64_t sizes[4] = {initial.size() * sizeof(half), source.size() * sizeof(half),
                               weights.size() * sizeof(half), output.size() * sizeof(float)};
    const int order[4] = {AS_ARG0, AS_ARG1, AS_ARG2, AS_ARG3};
    for (int i = 0; i < 4; ++i) {
      tensors[i].init_external(addresses[i], sizes[i], shapes[i], 2,
                               i == 3 ? DataType::FLOAT32 : DataType::FLOAT16, 0);
      tensors[i].start_offset = kGuard;
    }
    for (int i = 0; i < 4; ++i) args[i] = reinterpret_cast<int64_t>(&tensors[order[i]]);
    for (int core = 0; core < kCores; ++core) {
      pto::cpu_sim::ScopedExecutionContext context(core, 0);
      Matrix resident;
      pto::TASSIGN(resident, AS_L1);
      std::vector<half> matrix(initialBefore.begin() + kGuard + core * kMatrix,
                               initialBefore.begin() + kGuard + (core + 1) * kMatrix);
      // Build expected ND payloads from raw independent inputs; every row and
      // all holes are checked after generated code runs, not only the probe.
      if constexpr (kSparse) {
        // Decode the GM [output C16, input C16, channel, filter, lane]
        // layout independently. Padded inactive channels must never be read.
        for (int sourceRow = 0; sourceRow < AS_SOURCE_ROWS; ++sourceRow) {
          const int channel = sourceRow % 16;
          const int group = sourceRow / 16 % AS_GROUPS;
          const int outputPanel = sourceRow / (16 * AS_GROUPS);
          if (channel < kSourceRows) {
            for (int tap = 0; tap < 9; ++tap) {
              for (int lane = 0; lane < 16; ++lane) {
                const int destinationRow = group * 2304 + tap * 16 + channel * 145;
                matrix[destinationRow * kColumns + outputPanel * 16 + lane] =
                    sourceBefore[kGuard + (core * AS_SOURCE_ROWS + sourceRow) * AS_SOURCE_COLS + tap * 16 +
                                 lane];
              }
            }
          }
        }
      } else {
        for (int copy = 0; copy < kCopies; ++copy) {
          for (int row = 0; row < kSourceRows; ++row) {
            for (int column = 0; column < kSourceColumns; ++column) {
              matrix[(AS_ROW + copy * AS_ORIGIN_STEP + row * AS_STEP) * kColumns + AS_COL + column] =
                  sourceBefore[kGuard + ((core * kCopies + copy) * kSourceRows + row) * kSourceColumns +
                               column];
            }
          }
        }
      }
      for (int row = 0; row < kProbeRows; ++row) {
        for (int column = 0; column < 16; ++column) {
          float sum = 0.0f;
          for (int k = 0; k < 16; ++k) {
            sum = std::fma(static_cast<float>(matrix[(kProbeRow + row) * kColumns + AS_COL + k]),
                           static_cast<float>(weightsBefore[kGuard + k * 16 + column]), sum);
          }
          expected[kGuard + core * kProbeElements + row * 16 + column] = sum;
        }
      }
      kernel_entry(args.data());
      for (int row = 0; row < kRows; ++row) {
        for (int column = 0; column < kColumns; ++column) {
          if (std::bit_cast<uint16_t>(resident.GetElement(row, column)) !=
              std::bit_cast<uint16_t>(matrix[row * kColumns + column])) {
            std::fprintf(stderr, "row_step dataset=%d core=%d row=%d col=%d\n", dataset, core, row, column);
            return 1;
          }
        }
      }
    }
    if (std::memcmp(initial.data(), initialBefore.data(), initial.size() * sizeof(half)) ||
        std::memcmp(source.data(), sourceBefore.data(), source.size() * sizeof(half)) ||
        std::memcmp(weights.data(), weightsBefore.data(), weights.size() * sizeof(half)))
      return 2;
    if (std::memcmp(output.data(), expected.data(), output.size() * sizeof(float))) return 3;
  }
  return 0;
}
