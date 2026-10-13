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

#ifndef PYPTO_BACKEND_310P_BACKEND_310P_H_
#define PYPTO_BACKEND_310P_BACKEND_310P_H_

#include <string>

#include "pypto/backend/common/backend.h"

namespace pypto::backend {

/** Independent A1 code-generation backend; initially FP16 vector operations. */
class Backend310P final : public Backend {
 public:
  static Backend310P& Instance();
  [[nodiscard]] std::string GetTypeName() const override { return "310P"; }
  [[nodiscard]] const BackendHandler* GetHandler() const override;

 private:
  Backend310P();
};

}  // namespace pypto::backend
#endif  // PYPTO_BACKEND_310P_BACKEND_310P_H_
