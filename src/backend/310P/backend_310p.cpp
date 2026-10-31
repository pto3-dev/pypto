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
// Reshape and scalar access both address a flat ND byte view. Reject boxed
// storage, partial views and nonzero origins before either shared emitter.
uint64_t ValidateContiguousVector(const ir::TileTypePtr& tile, const ir::Span& span,
                                  const char* operation = "reshape", bool allow_integer_storage = false) {
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
  const bool dtype_supported =
      tile->dtype_ == DataType::FP16 || tile->dtype_ == DataType::FP32 ||
      (allow_integer_storage && (tile->dtype_ == DataType::INT32 || tile->dtype_ == DataType::UINT32));
  CHECK_SPAN(tile->GetMemorySpace() == ir::MemorySpace::Vec && tile->shape_.size() == 2 && dtype_supported &&
                 view.blayout == ir::TileLayout::row_major && view.slayout == ir::TileLayout::none_box &&
                 view.compact == ir::CompactMode::null &&
                 ir::tile_view_semantics::ShapeExprListsEquivalent(tile->shape_, view.valid_shape),
             span)
      << "Ascend310P " << operation << " requires full contiguous row-major "
      << (allow_integer_storage ? "FP16/FP32/INT32/UINT32" : "FP16/FP32") << " Vec views.";
  const auto rows = ir::As<ir::ConstInt>(tile->shape_[0]);
  const auto columns = ir::As<ir::ConstInt>(tile->shape_[1]);
  const uint64_t element_bytes = tile->dtype_.GetBit() / 8;
  const uint64_t capacity = Backend310P::Instance().GetMemSize(ir::MemorySpace::Vec);
  CHECK_SPAN(rows && columns && rows->value_ > 0 && columns->value_ > 0 &&
                 columns->value_ % (32 / element_bytes) == 0 &&
                 static_cast<uint64_t>(rows->value_) <= capacity / element_bytes / columns->value_,
             span)
      << "Ascend310P " << operation << " requires static 32-byte-aligned rows within the UB budget.";
  if (!view.stride.empty()) {
    CHECK_SPAN(view.stride.size() == 2 && ir::As<ir::ConstInt>(view.stride[0]) &&
                   ir::As<ir::ConstInt>(view.stride[1]) &&
                   ir::As<ir::ConstInt>(view.stride[0])->value_ == columns->value_ &&
                   ir::As<ir::ConstInt>(view.stride[1])->value_ == 1,
               span)
        << "Ascend310P " << operation << " does not support padded or strided Tile views.";
  }
  if (view.start_offset) {
    const auto offset = ir::As<ir::ConstInt>(view.start_offset);
    CHECK_SPAN(offset && offset->value_ == 0, span)
        << "Ascend310P " << operation << " requires a zero Tile view start offset.";
  }
  return static_cast<uint64_t>(rows->value_) * columns->value_;
}

// Sort32 emits compact score/index pairs. Both planners use the existing
// shared emitter, with A1 proposal banks supplied by InitMemRef or PTOAS.
void ValidateSort32(const ir::CallPtr& call) {
  INTERNAL_CHECK_SPAN(call->args_.size() == 2 || call->args_.size() == 3, call->span_)
      << "Internal error: sort32 requires source, indices and optional scratch.";
  const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
  const auto indices = ir::As<ir::TileType>(call->args_[1]->GetType());
  const auto result = ir::As<ir::TileType>(call->GetType());
  INTERNAL_CHECK_SPAN(source && indices && result, call->span_)
      << "Internal error: sort32 requires Tile operands and result.";
  CHECK_SPAN(source->dtype_ == DataType::FP32 && result->dtype_ == DataType::FP32 &&
                 indices->dtype_ == DataType::UINT32,
             call->span_)
      << "Ascend310P sort32 requires FP32 values/result and UINT32 indices.";
  ValidateContiguousVector(source, call->span_, "sort32");
  ValidateContiguousVector(indices, call->span_, "sort32", true);
  ValidateContiguousVector(result, call->span_, "sort32");
  const auto columns = ir::As<ir::ConstInt>(source->shape_[1])->value_;
  CHECK_SPAN(
      columns % 32 == 0 && ir::tile_view_semantics::ShapeExprListsEquivalent(source->shape_, indices->shape_),
      call->span_)
      << "Ascend310P sort32 requires complete 32-value blocks and matching indices.";
  if (call->args_.size() == 3) {
    const auto scratch = ir::As<ir::TileType>(call->args_[2]->GetType());
    INTERNAL_CHECK_SPAN(scratch, call->span_) << "Internal error: sort32 scratch must be a Tile.";
    CHECK_SPAN(scratch->dtype_ == DataType::FP32, call->span_)
        << "Ascend310P sort32 requires FP32 proposal scratch.";
    CHECK_SPAN(ValidateContiguousVector(scratch, call->span_, "sort32") >= 512, call->span_)
        << "Ascend310P sort32 requires at least 512 FP32 scratch elements.";
  }
}

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
  CHECK_SPAN(
      (*space != ir::MemorySpace::Right && *space != ir::MemorySpace::Acc) || columns == 16 || columns == 32,
      span)
      << "Ascend310P multi-K GEMM requires N16 or N32 Right and Acc tiles.";
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

std::string EmitPanelInsert(const ir::CallPtr& call, codegen::CodegenBase& base) {
  auto* codegen = dynamic_cast<codegen::PTOCodegen*>(&base);
  INTERNAL_CHECK_SPAN(codegen && call->args_.size() == 3, call->span_)
      << "Ascend310P panel insertion requires PTO codegen and three operands.";
  const auto target = ir::As<ir::TileType>(call->args_[0]->GetType());
  const auto source = ir::As<ir::TileType>(call->args_[1]->GetType());
  const auto destination = ir::As<ir::TileType>(call->GetType());
  INTERNAL_CHECK_SPAN(target && source && destination, call->span_)
      << "Ascend310P panel insertion requires tile operands and a tile result.";
  const bool vector_insert = source->GetMemorySpace() == ir::MemorySpace::Vec &&
                             target->GetMemorySpace() == ir::MemorySpace::Vec &&
                             destination->GetMemorySpace() == ir::MemorySpace::Vec;
  if (vector_insert) {
    ValidateContiguousVector(source, call->span_, "Vec panel insertion");
    ValidateContiguousVector(target, call->span_, "Vec panel insertion");
    ValidateContiguousVector(destination, call->span_, "Vec panel insertion");
    CHECK_SPAN(source->dtype_ == target->dtype_ && target->dtype_ == destination->dtype_, call->span_)
        << "Ascend310P Vec panel insertion requires matching element types.";
  } else {
    CHECK_SPAN(source->GetMemorySpace() == ir::MemorySpace::Mat &&
                   target->GetMemorySpace() == ir::MemorySpace::Left &&
                   destination->GetMemorySpace() == ir::MemorySpace::Left &&
                   source->dtype_ == DataType::FP16 && destination->dtype_ == DataType::FP16 &&
                   ir::As<ir::ConstInt>(source->shape_[1])->value_ == 16,
               call->span_)
        << "Ascend310P assemble requires a full FP16 Mat-to-Left C16 panel.";
  }
  CHECK_SPAN(!ir::GetOptionalDoubleKwarg(call->kwargs_, "pre_quant").has_value() &&
                 !call->GetKwarg<bool>("pre_relu", false),
             call->span_)
      << "Ascend310P panel insertion does not support fused quantization or activation.";
  const auto offset = ir::As<ir::MakeTuple>(call->args_[2]);
  INTERNAL_CHECK_SPAN(offset && offset->elements_.size() == 2, call->span_)
      << "Ascend310P panel insertion requires row and column offsets.";
  if (vector_insert) {
    const auto row = ir::As<ir::ConstInt>(offset->elements_[0]);
    const auto col = ir::As<ir::ConstInt>(offset->elements_[1]);
    const auto sr = ir::As<ir::ConstInt>(source->shape_[0])->value_;
    const auto sc = ir::As<ir::ConstInt>(source->shape_[1])->value_;
    const auto dr = ir::As<ir::ConstInt>(target->shape_[0])->value_;
    const auto dc = ir::As<ir::ConstInt>(target->shape_[1])->value_;
    CHECK_SPAN(row && col && sr <= 2048 && row->value_ >= 0 && col->value_ >= 0 && row->value_ <= dr - sr &&
                   col->value_ <= dc - sc && col->value_ % (32 / (source->dtype_.GetBit() / 8)) == 0,
               call->span_)
        << "Ascend310P Vec panel insertion requires bounded static offsets aligned to 32 bytes.";
  }
  const auto dst = codegen->GetCurrentResultTarget();
  // A partial write must preserve everything outside its window. Require
  // the planner's in-place target reuse to preserve the untouched region.
  // This applies to both Left and Vec destinations.
  CHECK_SPAN(codegen->GetExprAsCode(call->args_[0]) == dst, call->span_)
      << "Ascend310P panel insertion requires in-place target reuse.";
  // PTOAS proves bounded/aligned offsets and emits MTE1 or V dependencies.
  // The ISA also guards direct native callers.
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
      << "Ascend310P writeback requires a full FP32 ND N16 Vec tile.";
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
        "tile.get_block_idx", "tile.get_block_num", "tensor.view",  "tile.assemble",   "tile.maximum",
        "tile.reshape",       "tile.read",          "tile.write",   "tile.slice",      "tile.gather",
        "tile.sort32"}) {
    allowed.insert(ir::OpRegistry::GetInstance().GetOp(name)->name_);
  }
  for (auto it = backend_op_registry_.begin(); it != backend_op_registry_.end();) {
    if (!allowed.count(it->first)) {
      it = backend_op_registry_.erase(it);
    } else {
      auto emit = it->second.codegen_func;
      it->second.codegen_func = [emit](const ir::CallPtr& call, codegen::CodegenBase& codegen) {
        // Integer tiles support storage and indexed bit-copy/sort payloads. Sharing the
        // ND layout and UB checks does not admit integer arithmetic, casts,
        // moves or tile subviews.
        const bool storage_op = ir::IsOp(call, "tile.load") || ir::IsOp(call, "tile.store") ||
                                ir::IsOp(call, "tile.alloc") || ir::IsOp(call, "tile.create") ||
                                ir::IsOp(call, "tensor.view") || ir::IsOp(call, "tile.gather") ||
                                ir::IsOp(call, "tile.sort32");
        auto integer_storage = [&](DataType dtype) {
          return storage_op && (dtype == DataType::INT32 || dtype == DataType::UINT32);
        };
        auto validate = [&](const ir::TypePtr& type) {
          if (auto tile = ir::As<ir::TileType>(type)) {
            ValidateMatrixTile(tile, call->span_);
            CHECK_SPAN((tile->dtype_ == DataType::FP16 || tile->dtype_ == DataType::FP32 ||
                        integer_storage(tile->dtype_)),
                       call->span_)
                << "Ascend310P initial vector profile requires FP16 or FP32 tile data; "
                << "integer storage is limited to INT32/UINT32 allocation and DMA.";
            if (integer_storage(tile->dtype_)) {
              ValidateContiguousVector(tile, call->span_, "integer storage", true);
            }
          }
          if (auto tensor = ir::AsTensorTypeLike(type)) {
            CHECK_SPAN((tensor->dtype_ == DataType::FP16 || tensor->dtype_ == DataType::FP32 ||
                        integer_storage(tensor->dtype_)),
                       call->span_)
                << "Ascend310P initial vector profile requires FP16 or FP32 tensor data; "
                << "integer storage is limited to INT32/UINT32 allocation and DMA.";
          }
        };
        validate(call->GetType());
        for (const auto& argument : call->args_) validate(argument->GetType());
        if (ir::IsOp(call, "tile.sort32")) ValidateSort32(call);
        if (ir::IsOp(call, "tile.gather")) {
          INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
              << "Verified indexed gather requires source, indices and scratch.";
          const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
          const auto indices = ir::As<ir::TileType>(call->args_[1]->GetType());
          const auto scratch = ir::As<ir::TileType>(call->args_[2]->GetType());
          const auto result = ir::As<ir::TileType>(call->GetType());
          INTERNAL_CHECK_SPAN(source && indices && scratch && result, call->span_)
              << "Verified indexed gather requires tile operands and result.";
          ValidateContiguousVector(source, call->span_, "indexed gather", true);
          const auto count = ValidateContiguousVector(indices, call->span_, "indexed gather", true);
          const auto capacity = ValidateContiguousVector(scratch, call->span_, "indexed gather", true);
          ValidateContiguousVector(result, call->span_, "indexed gather", true);
          CHECK_SPAN(indices->dtype_ == DataType::INT32 && scratch->dtype_ == DataType::INT32, call->span_)
              << "Ascend310P indexed gather requires INT32 indices and scratch.";
          CHECK_SPAN(source->dtype_ == result->dtype_ &&
                         ir::tile_view_semantics::ShapeExprListsEquivalent(indices->shape_, result->shape_),
                     call->span_)
              << "Ascend310P indexed gather requires matching value types and destination-shaped indices.";
          CHECK_SPAN(capacity >= count * (source->dtype_.GetByte() / 2), call->span_)
              << "Ascend310P indexed gather scratch needs a 32-bit offset per 16-bit value half.";
        }
        if (ir::IsOp(call, "tile.slice")) {
          INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
              << "Verified tile.slice requires three operands.";
          CHECK_SPAN(call->kwargs_.empty(), call->span_)
              << "Ascend310P accumulator slice does not support valid-shape overrides.";
          const auto src = ir::As<ir::TileType>(call->args_[0]->GetType());
          const auto dst = ir::As<ir::TileType>(call->GetType());
          const auto offsets = call->args_.size() >= 3 ? ir::As<ir::MakeTuple>(call->args_[2]) : nullptr;
          INTERNAL_CHECK_SPAN(src && dst && offsets && offsets->elements_.size() == 2, call->span_)
              << "Verified tile.slice must contain tile types and two offsets.";
          const auto row = ir::As<ir::ConstInt>(offsets->elements_[0]);
          const auto col = ir::As<ir::ConstInt>(offsets->elements_[1]);
          const auto src_rows = ir::As<ir::ConstInt>(src->shape_[0]);
          const auto src_cols = ir::As<ir::ConstInt>(src->shape_[1]);
          const auto dst_rows = ir::As<ir::ConstInt>(dst->shape_[0]);
          const auto dst_cols = ir::As<ir::ConstInt>(dst->shape_[1]);
          if (src->GetMemorySpace() == ir::MemorySpace::Mat &&
              dst->GetMemorySpace() == ir::MemorySpace::Mat) {
            const auto source_view = ir::tile_view_semantics::GetEffectiveTileView(*src);
            const auto result_view = ir::tile_view_semantics::GetEffectiveTileView(*dst);
            // Full-height C16-aligned NZ spans keep the same physical column
            // stride as the parent. PTOAS can rebase these resident L1 views
            // without materializing or reloading each output-channel panel.
            CHECK_SPAN(row && col && src_rows && src_cols && dst_rows && dst_cols && row->value_ == 0 &&
                           dst_rows->value_ == src_rows->value_ && col->value_ >= 0 &&
                           col->value_ % 16 == 0 && col->value_ <= src_cols->value_ - dst_cols->value_ &&
                           source_view.blayout == ir::TileLayout::col_major &&
                           result_view.blayout == ir::TileLayout::col_major &&
                           source_view.slayout == ir::TileLayout::row_major &&
                           result_view.slayout == ir::TileLayout::row_major &&
                           source_view.compact == ir::CompactMode::null &&
                           result_view.compact == ir::CompactMode::null,
                       call->span_)
                << "Ascend310P Mat slices require a bounded full-height static aligned NZ column interval.";
          } else if (src->GetMemorySpace() == ir::MemorySpace::Vec &&
                     dst->GetMemorySpace() == ir::MemorySpace::Vec) {
            ValidateContiguousVector(src, call->span_, "Vec row slice");
            ValidateContiguousVector(dst, call->span_, "Vec row slice");
            CHECK_SPAN(row && col && src_rows && src_cols && dst_rows && dst_cols && col->value_ == 0 &&
                           row->value_ >= 0 && row->value_ <= src_rows->value_ - dst_rows->value_ &&
                           src_cols->value_ == dst_cols->value_ && src->dtype_ == dst->dtype_,
                       call->span_)
                << "Ascend310P Vec slices require a bounded full-width static row interval.";
          } else {
            CHECK_SPAN(src->GetMemorySpace() == ir::MemorySpace::Acc &&
                           dst->GetMemorySpace() == ir::MemorySpace::Acc,
                       call->span_)
                << "Ascend310P slice supports accumulator column panels or Vec row intervals only.";
            CHECK_SPAN(row && col && src_rows && src_cols && dst_rows && dst_cols && row->value_ == 0 &&
                           col->value_ >= 0 && col->value_ % 16 == 0 &&
                           col->value_ <= src_cols->value_ - 16 && dst_rows->value_ == src_rows->value_ &&
                           dst_cols->value_ == 16,
                       call->span_)
                << "Ascend310P accumulator slice requires one full-height aligned N16 column panel.";
          }
        }
        if (ir::IsOp(call, "tile.read") || ir::IsOp(call, "tile.write")) {
          // The Tile getter/setter uses scalar UB accesses on PIPE_S. Do not
          // expose FP16 scalar conversion or boxed matrix storage implicitly.
          const auto tile = ir::As<ir::TileType>(call->args_[0]->GetType());
          CHECK_SPAN(tile && tile->dtype_ == DataType::FP32, call->span_)
              << "Ascend310P scalar tile access requires an FP32 Vec tile.";
          ValidateContiguousVector(tile, call->span_, "scalar tile access");
        }
        if (ir::IsOp(call, "tile.reshape")) {
          INTERNAL_CHECK_SPAN(call->args_.size() == 2, call->span_)
              << "Tile reshape expects a source and shape.";
          const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
          const auto destination = ir::As<ir::TileType>(call->GetType());
          INTERNAL_CHECK_SPAN(source && destination, call->span_) << "Tile reshape expects tile types.";
          const auto source_elements = ValidateContiguousVector(source, call->span_);
          const auto destination_elements = ValidateContiguousVector(destination, call->span_);
          CHECK_SPAN(source->dtype_ == destination->dtype_ && source_elements == destination_elements,
                     call->span_)
              << "Ascend310P reshape requires matching element types and counts.";
        }
        if (ir::IsOp(call, "tile.assemble")) return EmitPanelInsert(call, codegen);
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
              dst->GetMemorySpace() == ir::MemorySpace::Vec) {
            ValidateContiguousVector(src, call->span_, "Vec-to-Vec move");
            ValidateContiguousVector(dst, call->span_, "Vec-to-Vec move");
            CHECK_SPAN(src->dtype_ == dst->dtype_ &&
                           ir::tile_view_semantics::ShapeExprListsEquivalent(src->shape_, dst->shape_),
                       call->span_)
                << "Ascend310P Vec-to-Vec moves require matching data types and shapes.";
            return emit(call, codegen);
          }
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
              << "Ascend310P moves require Vec-to-Vec, Vec-to-Mat, Mat-to-Left/Right or Acc-to-Vec.";
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
  backend_op_registry_.at("tile.assemble").infer_pipe_func = [](const ir::CallPtr& call) {
    const auto source = ir::As<ir::TileType>(call->args_[1]->GetType());
    return source && source->GetMemorySpace() == ir::MemorySpace::Vec ? ir::PipeType::V : ir::PipeType::MTE1;
  };
  backend_op_registry_.at("tile.move").infer_pipe_func = [](const ir::CallPtr& call) {
    const auto src = ir::As<ir::TileType>(call->args_[0]->GetType());
    const auto dst = ir::As<ir::TileType>(call->GetType());
    if (src && dst && src->GetMemorySpace() == ir::MemorySpace::Vec &&
        dst->GetMemorySpace() == ir::MemorySpace::Mat)
      return ir::PipeType::MTE3;
    return src && (src->GetMemorySpace() == ir::MemorySpace::Acc ||
                   (dst && src->GetMemorySpace() == ir::MemorySpace::Vec &&
                    dst->GetMemorySpace() == ir::MemorySpace::Vec))
               ? ir::PipeType::V
               : ir::PipeType::MTE1;
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
