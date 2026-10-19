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

#ifndef PYPTO_BACKEND_310P_BACKEND_310P_HANDLER_H_
#define PYPTO_BACKEND_310P_BACKEND_310P_HANDLER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"

namespace pypto::backend {

/** A1 capabilities do not inherit A2/A3 conversion or cross-core behavior. */
class Ascend310PHandler final : public BackendHandler {
 public:
  static const Ascend310PHandler& Instance();
  [[nodiscard]] std::string GetPtoTargetArch() const override { return "a1"; }
  [[nodiscard]] std::string GetLaunchSpecCoreCountMethod() const override { return "set_block_num"; }
  [[nodiscard]] std::string GetDefaultSimPlatform() const override {
    CHECK(false) << "Ascend310P has no supported simulator platform; use platform='a1' for compilation.";
    return {};
  }
  [[nodiscard]] std::vector<std::string> GetExtraPtoasFlags() const override { return {"--pto-arch", "a1"}; }
  [[nodiscard]] bool HonorsHighPrecisionAlgorithm() const override { return false; }
  [[nodiscard]] bool SupportsOrderingCompareDataType(const DataType&) const override { return false; }
  [[nodiscard]] bool SupportsTselsDataType(const DataType&) const override { return false; }
  [[nodiscard]] bool SupportsTselDataType(const DataType&) const override { return false; }
  [[nodiscard]] TileScratchSpec GetTselScratchSpec() const override {
    CHECK(false) << "Ascend310P tile selection is not yet supported.";
    return {};
  }
  [[nodiscard]] TileScratchSpec GetTselsScratchSpec(DataType, int64_t) const override {
    CHECK(false) << "Ascend310P scalar tile selection is not yet supported.";
    return {};
  }
  [[nodiscard]] bool SplitsCubeToVectorTransportInHardware() const override { return false; }
  [[nodiscard]] bool SupportsFixpipePreQuant(const DataType&, const DataType&, FixpipeDest) const override {
    return false;
  }
  [[nodiscard]] bool RequiresCoupledInCoreDispatch() const override { return true; }
  [[nodiscard]] bool RequiresGMPipeBuffer() const override { return false; }
  [[nodiscard]] bool RequiresSplitLoadTpopWorkaround() const override { return false; }
  [[nodiscard]] bool RequiresLevel3TmpScratch() const override { return false; }
  [[nodiscard]] bool RequiresVtoCFractalAdapt() const override { return false; }
  [[nodiscard]] bool RequiresRuntimeSubblockBridge() const override { return false; }
  [[nodiscard]] bool RequiresNoSplitDualAivDispatch() const override { return false; }
  [[nodiscard]] bool RequiresLowPrecisionMatScratch() const override { return false; }
  [[nodiscard]] bool SupportsBf16AtomicAdd() const override { return false; }
  [[nodiscard]] bool SupportsAccToGmDtype(const DataType&) const override { return false; }
  [[nodiscard]] bool SupportsAccToGmViaVec(const DataType& source,
                                           const DataType& destination) const override {
    return source == DataType::FP32 && destination == DataType::FP32;
  }
  [[nodiscard]] bool SupportsMatToBiasMove(const DataType&, const DataType&) const override { return false; }
  [[nodiscard]] ir::TileView BuildCrossCoreTransferView(ir::MemorySpace, const ir::TileView&) const override {
    CHECK(false) << "Ascend310P cross-core transfers are not yet supported.";
    return {};
  }
  [[nodiscard]] const TcvtAdjacency& GetTcvtAdjacency() const override {
    static const TcvtAdjacency adjacency = {
        {{DataType::FP16, DataType::FP32}, {DataType::FP32, DataType::FP16}}};
    return adjacency;
  }
  [[nodiscard]] uint32_t GetGmAccessGranularityBytes() const override { return 32; }
  [[nodiscard]] uint32_t GetL2CacheLineBytes() const override { return 0; }
  // Disable bandwidth advice until the A1 threshold and cache-line size are
  // established; DMA legality still requires the 32-byte granularity above.
  [[nodiscard]] uint32_t GetRecommendedInnermostDimBytes() const override { return 0; }
  [[nodiscard]] uint32_t GetL0aCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0bCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0cCapacityBytes() const override { return 256ULL * 1024; }
  [[nodiscard]] uint32_t GetBiasCapacityBytes() const override { return 0; }
  [[nodiscard]] uint64_t GetMatCapacityBytes() const override { return 512ULL * 1024; }

 private:
  Ascend310PHandler() = default;
};

}  // namespace pypto::backend
#endif  // PYPTO_BACKEND_310P_BACKEND_310P_HANDLER_H_
