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

#include "pypto/backend/310P/backend_310p.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "pypto/backend/310P/backend_310p_handler.h"
#include "pypto/backend/common/pto_ops_common.h"
#include "pypto/backend/common/soc.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/tile_view_semantics.h"

namespace pypto::backend {

namespace {
void ValidateMatrixTile(const ir::TileTypePtr& tile, const ir::Span& span) {
  const auto space = tile->GetMemorySpace();
  if (!space || (*space != ir::MemorySpace::Mat && *space != ir::MemorySpace::Left &&
                 *space != ir::MemorySpace::Right && *space != ir::MemorySpace::Acc))
    return;
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
  auto is_full = [](const std::vector<ir::ExprPtr>& shape) {
    return shape.size() == 2 && ir::As<ir::ConstInt>(shape[0]) && ir::As<ir::ConstInt>(shape[1]) &&
           ir::As<ir::ConstInt>(shape[0])->value_ == 16 && ir::As<ir::ConstInt>(shape[1])->value_ == 16;
  };
  CHECK_SPAN(is_full(tile->shape_) && is_full(view.valid_shape), span)
      << "Ascend310P matrix profile requires static full 16x16 tiles.";
  CHECK_SPAN(tile->dtype_ == (*space == ir::MemorySpace::Acc ? DataType::FP32 : DataType::FP16), span)
      << "Ascend310P matrix profile requires FP16 inputs and FP32 accumulators.";
}

std::string EmitAccMove(const ir::CallPtr& call, codegen::CodegenBase& base) {
  auto* codegen = dynamic_cast<codegen::PTOCodegen*>(&base);
  INTERNAL_CHECK_SPAN(codegen, call->span_) << "Ascend310P accumulator lowering requires PTO codegen.";
  const auto dst = ir::As<ir::TileType>(call->GetType());
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*dst);
  CHECK_SPAN(dst->dtype_ == DataType::FP32 && dst->shape_.size() == 2 &&
                 ir::As<ir::ConstInt>(dst->shape_[0]) && ir::As<ir::ConstInt>(dst->shape_[1]) &&
                 ir::As<ir::ConstInt>(dst->shape_[0])->value_ == 16 &&
                 ir::As<ir::ConstInt>(dst->shape_[1])->value_ == 16 &&
                 view.blayout == ir::TileLayout::row_major && view.slayout == ir::TileLayout::none_box,
             call->span_)
      << "Ascend310P writeback requires a full FP32 ND 16x16 Vec tile.";
  const auto src = codegen->GetExprAsCode(call->args_[0]);
  const auto src_type = codegen->GetExprTypeAnnotation(call->args_[0]);
  const auto target = codegen->GetCurrentResultTarget();
  const auto target_type = codegen->GetCurrentResultTileBufTypeString();
  codegen->Emit("pto.tmov ins(" + src + " : " + src_type + ") outs(" + target + " : " + target_type +
                ") {accToVecMode = #pto.acc_to_vec_mode<single_mode_vec0>}");
  return {};
}
}  // namespace

Backend310P::Backend310P() : Backend(Create310PSoC()) {
  RegisterPTOOps(*this);
  // Keep only shared emitters whose native A1 instructions have been validated.
  // Resolve names through the IR registry so renamed or misspelled ops fail.
  std::unordered_set<std::string> allowed;
  for (const char* name :
       {"tile.load", "tile.store", "tile.create", "tile.alloc", "tile.add", "tile.sub", "tile.mul",
        "tile.move", "tile.matmul", "tile.cast", "tile.get_block_idx", "tile.get_block_num"}) {
    allowed.insert(ir::OpRegistry::GetInstance().GetOp(name)->name_);
  }
  for (auto it = backend_op_registry_.begin(); it != backend_op_registry_.end();) {
    if (!allowed.count(it->first)) {
      it = backend_op_registry_.erase(it);
    } else {
      auto emit = it->second.codegen_func;
      it->second.codegen_func = [emit](const ir::CallPtr& call, codegen::CodegenBase& codegen) {
        auto validate = [&](const ir::TypePtr& type) {
          if (auto tile = ir::As<ir::TileType>(type)) {
            ValidateMatrixTile(tile, call->span_);
            CHECK_SPAN((tile->dtype_ == DataType::FP16 || tile->dtype_ == DataType::FP32), call->span_)
                << "Ascend310P initial vector profile requires FP16 or FP32 tile data.";
          }
          if (auto tensor = ir::As<ir::TensorType>(type))
            CHECK_SPAN((tensor->dtype_ == DataType::FP16 || tensor->dtype_ == DataType::FP32), call->span_)
                << "Ascend310P initial vector profile requires FP16 or FP32 tensor data.";
        };
        validate(call->GetType());
        for (const auto& argument : call->args_) validate(argument->GetType());
        if (ir::IsOp(call, "tile.cast")) {
          INTERNAL_CHECK_SPAN(call->args_.size() == 1 || call->args_.size() == 2, call->span_)
              << "Tile cast expects a source and optional scratch operand.";
          const bool has_scratch = call->args_.size() == 2;
          CHECK_SPAN(!has_scratch && call->GetKwarg<int>("mode", -1) == 0, call->span_)
              << "Ascend310P floating conversion requires mode='none' without scratch.";
          const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
          const auto destination = ir::As<ir::TileType>(call->GetType());
          CHECK_SPAN(source && destination && source->GetMemorySpace() == ir::MemorySpace::Vec &&
                         destination->GetMemorySpace() == ir::MemorySpace::Vec &&
                         source->dtype_ != destination->dtype_,
                     call->span_)
              << "Ascend310P floating conversion requires an FP16/FP32 Vec pair.";
        }
        if (ir::IsOp(call, "tile.move")) {
          const auto src = ir::As<ir::TileType>(call->args_[0]->GetType());
          const auto dst = ir::As<ir::TileType>(call->GetType());
          CHECK_SPAN(src && dst, call->span_) << "Ascend310P moves require tile operands.";
          if (src->GetMemorySpace() == ir::MemorySpace::Acc && dst->GetMemorySpace() == ir::MemorySpace::Vec)
            return EmitAccMove(call, codegen);
          CHECK_SPAN(src->GetMemorySpace() == ir::MemorySpace::Mat &&
                         (dst->GetMemorySpace() == ir::MemorySpace::Left ||
                          dst->GetMemorySpace() == ir::MemorySpace::Right),
                     call->span_)
              << "Ascend310P moves require Mat-to-Left/Right or Acc-to-Vec.";
        }
        if (ir::IsOp(call, "tile.store")) {
          const auto src = ir::As<ir::TileType>(call->args_[0]->GetType());
          CHECK_SPAN(src && src->GetMemorySpace() == ir::MemorySpace::Vec, call->span_)
              << "Ascend310P stores require explicit accumulator-to-vector writeback.";
        }
        return emit(call, codegen);
      };
      ++it;
    }
  }
  backend_op_registry_.at("tile.move").infer_pipe_func = [](const ir::CallPtr& call) {
    const auto src = ir::As<ir::TileType>(call->args_[0]->GetType());
    return src && src->GetMemorySpace() == ir::MemorySpace::Acc ? ir::PipeType::V : ir::PipeType::MTE1;
  };
}

Backend310P& Backend310P::Instance() {
  static Backend310P instance;
  return instance;
}

const BackendHandler* Backend310P::GetHandler() const { return &Ascend310PHandler::Instance(); }

const Ascend310PHandler& Ascend310PHandler::Instance() {
  static const Ascend310PHandler instance;
  return instance;
}

}  // namespace pypto::backend
