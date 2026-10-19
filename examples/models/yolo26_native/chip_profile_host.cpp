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
#include <acl/acl.h>
#include <runtime/rt.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "aicpu_loader/host/load_aicpu_op.h"
#include "tensormap_and_ringbuffer/tensor.h"

static thread_local std::string error;
static void Check(int result) {
  if (result) throw std::runtime_error(std::to_string(result));
}
static std::vector<char> Read(const char* path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<char> bytes{std::istreambuf_iterator<char>(file), {}};
  if (bytes.empty()) throw std::runtime_error("empty ELF");
  return bytes;
}
struct Probe {
  host::LoadAicpuOp loader;
  aclrtStream cpu = nullptr, core = nullptr;
  void* kernel = nullptr;
  std::vector<char> image;
  ~Probe() {
    if (core) aclrtSynchronizeStreamWithTimeout(core, 5000);
    if (cpu) aclrtSynchronizeStreamWithTimeout(cpu, 5000);
    loader.Finalize();
    if (kernel) rtDevBinaryUnRegister(kernel);
    if (core) aclrtDestroyStream(core);
    if (cpu) aclrtDestroyStream(cpu);
  }
  void Init(const char* dispatcher_path, const char* cpu_path, const char* core_path) {
    const char* soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend310P") != 0)
      throw std::invalid_argument("native chip diagnostic requires Ascend310P");
    int32_t device = 0;
    Check(aclrtGetDevice(&device));
    Check(aclrtCreateStream(&cpu));
    Check(aclrtCreateStream(&core));
    auto dispatcher = Read(dispatcher_path);
    auto cpu_image = Read(cpu_path);
    image = Read(core_path);
    Check(loader.BootstrapDispatcher(dispatcher.data(), dispatcher.size(), cpu_image.data(), cpu_image.size(),
                                     cpu, device));
    Check(loader.Init({}));
    rtDevBinary_t binary{};
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.data = image.data();
    binary.length = image.size();
    Check(rtRegisterAllKernel(&binary, &kernel));
  }
  void Run(uint64_t wire, uint64_t calls) {
    Check(loader.LaunchBuiltInOp(cpu, &wire, sizeof(wire), 1, "simpler_aicpu_exec"));
    std::array<uint64_t, 2> pointers{wire, calls};
    rtArgsEx_t args{};
    args.args = pointers.data();
    args.argsSize = sizeof(pointers);
    rtTaskCfgInfo_t config{};
    config.schemMode = RT_SCHEM_MODE_BATCH;
    Check(rtKernelLaunchWithHandleV2(kernel, 0, 8, &args, nullptr, core, &config));
    Check(aclrtSynchronizeStreamWithTimeout(core, 5000));
    Check(aclrtSynchronizeStreamWithTimeout(cpu, 5000));
  }
};
extern "C" const char* ProbeError() { return error.c_str(); }
extern "C" void* ProbeCreate(const char* dispatcher, const char* cpu, const char* core) {
  try {
    auto probe = std::make_unique<Probe>();
    probe->Init(dispatcher, cpu, core);
    return probe.release();
  } catch (const std::exception& ex) {
    error = ex.what();
    return nullptr;
  }
}
extern "C" int ProbeRun(Probe* probe, uint64_t wire, uint64_t calls) {
  try {
    probe->Run(wire, calls);
    return 0;
  } catch (const std::exception& ex) {
    error = ex.what();
    return 1;
  }
}
extern "C" void ProbeClose(Probe* probe) { delete probe; }
extern "C" void Descriptor(uint64_t address, uint64_t bytes, const uint32_t* shape, uint32_t rank,
                           uint32_t dtype, void* output) {
  simpler::tmr::Tensor descriptor{};
  descriptor.init_external(reinterpret_cast<void*>(address), bytes, shape, rank, static_cast<DataType>(dtype),
                           0, false, AddressSpace::DEVICE);
  std::memcpy(output, &descriptor, sizeof(descriptor));
}
