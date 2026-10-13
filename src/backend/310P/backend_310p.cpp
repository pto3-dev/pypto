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

#include "pypto/backend/310P/backend_310p_handler.h"
#include "pypto/backend/common/pto_ops_common.h"
#include "pypto/backend/common/soc.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"

namespace pypto::backend {

Backend310P::Backend310P() : Backend(Create310PSoC()) {
  RegisterPTOOps(*this);
  // Keep only shared emitters whose native A1 instructions have been validated.
  // Resolve names through the IR registry so renamed or misspelled ops fail.
  std::unordered_set<std::string> allowed;
  for (const char* name : {"tile.load", "tile.store", "tile.create", "tile.alloc", "tile.add",
                           "tile.get_block_idx", "tile.get_block_num"}) {
    allowed.insert(ir::OpRegistry::GetInstance().GetOp(name)->name_);
  }
  for (auto it = backend_op_registry_.begin(); it != backend_op_registry_.end();) {
    if (!allowed.count(it->first)) {
      it = backend_op_registry_.erase(it);
    } else {
      auto emit = it->second.codegen_func;
      it->second.codegen_func = [emit](const ir::CallPtr& call, codegen::CodegenBase& codegen) {
        auto validate = [&](const ir::TypePtr& type) {
          if (auto tile = ir::As<ir::TileType>(type))
            CHECK_SPAN((tile->dtype_ == DataType::FP16 || tile->dtype_ == DataType::FP32), call->span_)
                << "Ascend310P initial vector profile requires FP16 or FP32 tile data.";
          if (auto tensor = ir::As<ir::TensorType>(type))
            CHECK_SPAN((tensor->dtype_ == DataType::FP16 || tensor->dtype_ == DataType::FP32), call->span_)
                << "Ascend310P initial vector profile requires FP16 or FP32 tensor data.";
        };
        validate(call->GetType());
        for (const auto& argument : call->args_) validate(argument->GetType());
        return emit(call, codegen);
      };
      ++it;
    }
  }
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
