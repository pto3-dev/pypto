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

#include <cstdint>
#include <optional>
#include <sstream>
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
    if (shape.size() != 2) return false;
    for (const auto& dimension : shape) {
      const auto value = ir::As<ir::ConstInt>(dimension);
      if (!value || value->value_ < 16 || value->value_ > 2048 || value->value_ % 16 != 0) return false;
    }
    return true;
  };
  CHECK_SPAN(is_full(tile->shape_) && is_full(view.valid_shape) &&
                 ir::tile_view_semantics::ShapeExprListsEquivalent(tile->shape_, view.valid_shape),
             span)
      << "Ascend310P matrix profile requires full aligned dimensions in [16, 2048].";
  CHECK_SPAN(tile->dtype_ == (*space == ir::MemorySpace::Acc ? DataType::FP32 : DataType::FP16), span)
      << "Ascend310P matrix profile requires FP16 inputs and FP32 accumulators.";
  const auto rows = ir::As<ir::ConstInt>(tile->shape_[0])->value_;
  const auto columns = ir::As<ir::ConstInt>(tile->shape_[1])->value_;
  CHECK_SPAN((*space != ir::MemorySpace::Right && *space != ir::MemorySpace::Acc) || columns == 16, span)
      << "Ascend310P multi-K GEMM requires N16 Right and Acc tiles.";
  CHECK_SPAN(*space != ir::MemorySpace::Left || columns == 16 ||
                 (view.blayout == ir::TileLayout::row_major && view.slayout == ir::TileLayout::row_major),
             span)
      << "Ascend310P multi-K Left requires ZZ layout; legacy NZ is valid for K16 only.";
  const auto& handler = Ascend310PHandler::Instance();
  uint64_t capacity = handler.GetMatCapacityBytes();
  if (*space == ir::MemorySpace::Left) capacity = handler.GetL0aCapacityBytes();
  if (*space == ir::MemorySpace::Right) capacity = handler.GetL0bCapacityBytes();
  if (*space == ir::MemorySpace::Acc) capacity = handler.GetL0cCapacityBytes();
  const uint64_t bytes = static_cast<uint64_t>(rows) * columns * tile->dtype_.GetBit() / 8;
  CHECK_SPAN(bytes <= capacity, span) << "Ascend310P matrix tile exceeds its memory budget.";
}

std::string EmitLeftPanelInsert(const ir::CallPtr& call, codegen::CodegenBase& base) {
  auto* codegen = dynamic_cast<codegen::PTOCodegen*>(&base);
  INTERNAL_CHECK_SPAN(codegen && call->args_.size() == 3, call->span_)
      << "Ascend310P panel insertion requires PTO codegen and three operands.";
  const auto target = ir::As<ir::TileType>(call->args_[0]->GetType());
  const auto source = ir::As<ir::TileType>(call->args_[1]->GetType());
  const auto destination = ir::As<ir::TileType>(call->GetType());
  INTERNAL_CHECK_SPAN(target && source && destination, call->span_)
      << "Ascend310P panel insertion requires tile operands and a tile result.";
  CHECK_SPAN(
      source->GetMemorySpace() == ir::MemorySpace::Mat && target->GetMemorySpace() == ir::MemorySpace::Left &&
          destination->GetMemorySpace() == ir::MemorySpace::Left && source->dtype_ == DataType::FP16 &&
          destination->dtype_ == DataType::FP16 && ir::As<ir::ConstInt>(source->shape_[1])->value_ == 16,
      call->span_)
      << "Ascend310P assemble requires a full FP16 Mat-to-Left C16 panel.";
  CHECK_SPAN(!ir::GetOptionalDoubleKwarg(call->kwargs_, "pre_quant").has_value() &&
                 !call->GetKwarg<bool>("pre_relu", false),
             call->span_)
      << "Ascend310P panel insertion does not support fused quantization or activation.";
  const auto offset = ir::As<ir::MakeTuple>(call->args_[2]);
  INTERNAL_CHECK_SPAN(offset && offset->elements_.size() == 2, call->span_)
      << "Ascend310P panel insertion requires row and column offsets.";
  const auto dst = codegen->GetCurrentResultTarget();
  // A partial write must preserve everything outside its window. Left-to-Left
  // copying has no validated A1 instruction; require the planner's in-place
  // target reuse rather than silently losing the untouched L0A region.
  CHECK_SPAN(codegen->GetExprAsCode(call->args_[0]) == dst, call->span_)
      << "Ascend310P panel insertion requires in-place Left target reuse.";
  // PTOAS proves bounded/aligned offsets, including affine loop expressions,
  // and emits MTE1 dependencies. The ISA also guards direct native callers.
  std::ostringstream instruction;
  instruction << "pto.tinsert ins(" << codegen->GetExprAsCode(call->args_[1]) << ", "
              << codegen->GetExprAsCode(offset->elements_[0]) << ", "
              << codegen->GetExprAsCode(offset->elements_[1]) << " : "
              << codegen->GetExprTypeAnnotation(call->args_[1]) << ", index, index) outs(" << dst << " : "
              << codegen->GetCurrentResultTileBufTypeString() << ")";
  codegen->Emit(instruction.str());
  return {};
}

std::string EmitAccMove(const ir::CallPtr& call, codegen::CodegenBase& base) {
  auto* codegen = dynamic_cast<codegen::PTOCodegen*>(&base);
  INTERNAL_CHECK_SPAN(codegen, call->span_) << "Ascend310P accumulator lowering requires PTO codegen.";
  const auto dst = ir::As<ir::TileType>(call->GetType());
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*dst);
  CHECK_SPAN(dst->dtype_ == DataType::FP32 && dst->shape_.size() == 2 &&
                 ir::As<ir::ConstInt>(dst->shape_[0]) && ir::As<ir::ConstInt>(dst->shape_[1]) &&
                 ir::As<ir::ConstInt>(dst->shape_[0])->value_ >= 16 &&
                 ir::As<ir::ConstInt>(dst->shape_[0])->value_ <= 2048 &&
                 ir::As<ir::ConstInt>(dst->shape_[0])->value_ % 16 == 0 &&
                 ir::As<ir::ConstInt>(dst->shape_[1])->value_ == 16 &&
                 view.blayout == ir::TileLayout::row_major && view.slayout == ir::TileLayout::none_box,
             call->span_)
      << "Ascend310P writeback requires a full FP32 ND Nx16 Vec tile.";
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
  // Multi-K A1 L0A consumes M-major ZZ fractals. InferTileMemorySpace
  // materializes this layout while retaining the canonical NZ Mat input.
  backend_op_registry_.at("tile.matmul").tile_layout_spec =
      BackendTileLayoutSpec{{ir::TileLayout::row_major, std::nullopt}, std::nullopt};
  backend_op_registry_.at("tile.matmul_acc").tile_layout_spec =
      BackendTileLayoutSpec{{std::nullopt, ir::TileLayout::row_major, std::nullopt}, std::nullopt};
  // Keep only shared emitters whose native A1 instructions have been validated.
  // Resolve names through the IR registry so renamed or misspelled ops fail.
  std::unordered_set<std::string> allowed;
  for (const char* name :
       {"tile.load",          "tile.store",         "tile.create",  "tile.alloc",      "tile.add",
        "tile.sub",           "tile.mul",           "tile.full",    "tile.row_sum",    "tile.row_max",
        "tile.row_expand",    "tile.col_expand",    "tile.col_sum", "tile.col_max",    "tile.exp",
        "tile.div",           "tile.move",          "tile.matmul",  "tile.matmul_acc", "tile.cast",
        "tile.get_block_idx", "tile.get_block_num", "tensor.view",  "tile.assemble"}) {
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
        if (ir::IsOp(call, "tile.assemble")) return EmitLeftPanelInsert(call, codegen);
        if (ir::IsOp(call, "tensor.view")) {
          const auto tensor = ir::As<ir::TensorType>(call->GetType());
          CHECK_SPAN(tensor && tensor->tensor_view_.has_value() &&
                         tensor->tensor_view_->layout == ir::TensorLayout::ND,
                     call->span_)
              << "Ascend310P tensor views require canonical ND layout.";
        }
        if (ir::IsOp(call, "tile.col_sum")) {
          INTERNAL_CHECK_SPAN(call->args_.size() == 1 || call->args_.size() == 2, call->span_)
              << "Column sum expects a source and optional scratch operand.";
          const bool has_scratch = call->args_.size() == 2;
          CHECK_SPAN(!has_scratch, call->span_)
              << "Ascend310P column sum supports the sequential form without scratch only.";
        }
        if (ir::IsOp(call, "tile.div")) {
          CHECK_SPAN(!call->GetKwarg<bool>("high_precision", false), call->span_)
              << "Ascend310P division supports default precision only.";
        }
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
          if (src->GetMemorySpace() == ir::MemorySpace::Vec &&
              dst->GetMemorySpace() == ir::MemorySpace::Mat) {
            const auto view = ir::tile_view_semantics::GetEffectiveTileView(*src);
            CHECK_SPAN(src->dtype_ == DataType::FP16 &&
                           ir::tile_view_semantics::ShapeExprListsEquivalent(src->shape_, dst->shape_) &&
                           ir::tile_view_semantics::ShapeExprListsEquivalent(src->shape_, view.valid_shape) &&
                           view.blayout == ir::TileLayout::row_major &&
                           view.slayout == ir::TileLayout::none_box,
                       call->span_)
                << "Ascend310P Vec-to-Mat moves require matching full FP16 ND tiles.";
            // Matrix validation established static aligned dimensions. UB owns
            // the source; the existing planner also checks aggregate live usage.
            const auto rows = ir::As<ir::ConstInt>(dst->shape_[0])->value_;
            const auto columns = ir::As<ir::ConstInt>(dst->shape_[1])->value_;
            CHECK_SPAN(static_cast<uint64_t>(rows) * columns * sizeof(uint16_t) <=
                           Backend310P::Instance().GetMemSize(ir::MemorySpace::Vec),
                       call->span_)
                << "Ascend310P Vec-to-Mat source exceeds its UB budget.";
            return emit(call, codegen);
          }
          CHECK_SPAN(src->GetMemorySpace() == ir::MemorySpace::Mat &&
                         (dst->GetMemorySpace() == ir::MemorySpace::Left ||
                          dst->GetMemorySpace() == ir::MemorySpace::Right),
                     call->span_)
              << "Ascend310P moves require Vec-to-Mat, Mat-to-Left/Right or Acc-to-Vec.";
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
  backend_op_registry_.at("tile.assemble").infer_pipe_func = [](const ir::CallPtr&) {
    return ir::PipeType::MTE1;
  };
  backend_op_registry_.at("tile.move").infer_pipe_func = [](const ir::CallPtr& call) {
    const auto src = ir::As<ir::TileType>(call->args_[0]->GetType());
    const auto dst = ir::As<ir::TileType>(call->GetType());
    if (src && dst && src->GetMemorySpace() == ir::MemorySpace::Vec &&
        dst->GetMemorySpace() == ir::MemorySpace::Mat)
      return ir::PipeType::MTE3;
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
