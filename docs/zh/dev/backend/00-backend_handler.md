# BackendHandler：原则化的后端分发

> 跟踪 issue：[#948](https://github.com/hw-native-sys/pypto/issues/948)

## 背景

早期版本的 PyPTO 在 pass 与 codegen 中直接根据 `backend::BackendType` 做分支：

```cpp
if (backend::GetBackendType() != backend::BackendType::Ascend910B) { ... }
```

这种写法导致每新增一种后端，都要去整个代码库里翻这些散落的 `if`，并且为每一处
都补一条 case。`BackendHandler` 把所有这类分支集中到一个虚接口里，使新增后端
变成一次自包含的局部修改。

## 设计

`BackendHandler`（`include/pypto/backend/common/backend_handler.h`）是一个抽象
接口，它把后端之间所有"行为差异"显式命名出来。每个 `Backend` 子类各自持有一个
`BackendHandler` 子类的单例，并通过新引入的纯虚函数 `Backend::GetHandler()`
对外暴露。

```text
                       ┌────────────────────────────┐
        Pass / Codegen │  PassContext::Current()    │
                       │     ->GetBackendHandler()  │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │  BackendConfig::GetBackend │
                       │     ->GetHandler()         │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────┐
                │  Backend910B / Backend950 / ...      │
                │     -> Ascend910BHandler::Instance() │
                │     -> Ascend950Handler::Instance()  │
                └──────────────────────────────────────┘
```

`PassContext` 上的便捷访问点满足 `pass-context-config` 规则的要求：pass 通过
当前 `PassContext` 查询后端行为，而不是直接读全局状态。

## 接口

| 方法 | 用途 | Ascend910B | Ascend950 |
| ---- | ---- | ---------- | --------- |
| `GetPtoTargetArch()` | `module attributes {pto.target_arch = "..."}` | `"a2a3"` | `"a5"` |
| `GetLaunchSpecCoreCountMethod()` | `launch_spec` 上设置核数的运行时 API 名 | `"set_block_num"` | `"set_core_num"` |
| `GetDefaultSimPlatform()` | 默认仿真平台名 | `"a2a3sim"` | `"a5sim"` |
| `GetExtraPtoasFlags()` | ptoas 额外参数 | `[]` | `["--pto-arch", "a5"]` |
| `RequiresGMPipeBuffer()` | `ExpandMixedKernel` 是否注入 GM 槽位缓冲 | `true` | `false` |
| `RequiresSplitLoadTpopWorkaround()` | MemoryReuse 是否做 load + tpop_from_aic 原地复用危害规避 | `true` | `false` |
| `RequiresLevel3TmpScratch()` | InitMemRef 是否物化 level-3 显式 tmp；codegen static-view 桥接 | `true` | `false` |
| `RequiresVtoCFractalAdapt()` | AIV 端 V→C tpush 是否需要 fractal 适配 `tile.move` | `false` | `true` |
| `RequiresRuntimeSubblockBridge()` | 拆分 AIV 包装器是否从 runtime 上下文取 subblock id | `true` | `false` |
| `RequiresNoSplitDualAivDispatch()` | `no_split` 混合 kernel 是否仍需在两个 AIV lane 上同时下发 | `true` | `false` |
| `GetL0cMAlignment(dtype)` | L0C 切分、dbC 容量与分配共同使用的物理 M 行对齐 | INT32 为 32；其余为 16 | 16 |
| `BuildCrossCoreTransferView(dest, view)` | 跨核传输边界处的 tile 视图 | Mat/Left/Right 转 NZ；Vec 保持原样 | Mat/Left/Right 转 NZ（a5 硬件要求边界为 fractal）；Vec 保持原样 |

## 新增后端流程

1. 在 `src/backend/<arch>/backend_<arch>.cpp` 中实现 `Backend<Arch>` 子类，
   并把源文件加入 `CMakeLists.txt`。
2. 在 `src/backend/<arch>/backend_<arch>_handler.cpp` 中实现
   `Backend<Arch>Handler` 子类。
3. 让 `Backend<Arch>::GetHandler()` 返回你的 handler 单例。
4. 在 `src/backend/common/backend.cpp` 中给 `GetBackendInstance` 和
   `BackendTypeToString` 各加一行，并在 `BackendType` 枚举中添加新值。
   这是整个仓库里唯一按枚举展开的地方。

任何 pass / codegen 文件都不需要修改。可以用
`tests/ut/backend/test_backend_handler.py` 配合 `tests/ut/ir/transforms/` 与
`tests/ut/codegen/` 里的回归测试做验证。

## Python 访问方式

handler 也通过 Python 绑定暴露：

```python
from pypto.pypto_core import backend as _backend_core

# 全局后端类型已配置时：
handler = _backend_core.get_handler()

# 调用方已知道目标后端时：
handler = _backend_core.get_backend_instance(BackendType.Ascend950).get_handler()

handler.get_pto_target_arch()              # "a2a3" 或 "a5"
handler.requires_runtime_subblock_bridge()  # bool
handler.get_extra_ptoas_flags()            # list[str]
```

运行时相关的 Python 模块（`pypto.runtime.runner`、`pypto.ir.compiled_program`、
`pypto.backend.pto_backend`）现在通过这些访问器来获取后端行为，不再直接判断
`BackendType`，与 C++ 改造保持一致。

## Ascend 310P 编译配置

`Backend310P` 和 `Ascend310PHandler` 选择独立的 `a1` 目标。初始算子注册
支持 FP16/FP32 向量 load/add/subtract/multiply/store、tile 分配和 block 索引。PTOAS 校验静态、
完整、对齐的 ND 形状，两个内存规划器使用保守的 A1 容量。SoC 记录八个
耦合 AI Core 和七个额外向量核，不假定双 AIV 调度。当前只开放已验证的
DDR–Vec DMA 路径。转换、BF16、原子操作、GEMM 和跨核操作需要独立验证。

编译时使用 `backend_type=BackendType.Ascend310P`，默认编译平台为 `a1`。
显式选择 310P 后，旧 A3/A5 平台参数不能覆盖目标。需要使用配套的 A1
PTOAS 和 PTO-ISA 特性分支。后端可用性表示编译支持。仅包含一个静态
InCore 核函数的编译产物可通过 `compiled(*tensors)` 调用；输入为连续的
CPU FP16/FP32 张量。也可使用 `RunConfig(platform="a1", device_id=...)`
指定设备。同步执行路径使用锁定的 A1 ISA 和张量描述符运行时；图编排、
设备常驻张量、DFX 和模拟器调度尚未开放。

A1 带宽性能提示在完成校准之前保持关闭。

静态 ND 分区可读取和写回少于父张量宽度的列。行步长与列偏移须按 32 字节对齐，分区须位于源视图范围内。同步调用仍接收连续的 CPU 父张量，输出中分区以外的列保持原值。

静态完整的 FP16/FP32 向量 tile 无需凑齐 256 字节算术重复。加、减、乘对最后一次重复使用掩码，DMA 行仍须按 32 字节对齐。因此可执行单块 tile，动态 valid shape 和未对齐列尾仍未开放。

限定矩阵配置支持 K=N=16、16 至 2048 之间按 16 对齐的静态完整 M 行，使用 FP16 Mat 输入及 FP32 累加器。矩阵/向量混合函数保留为单个耦合 AIC callable。内存空间推导在 Acc 写回前插入显式 FP32 ND Vec 搬运，由正常的存活期及地址规划管理暂存，再以 `single_mode_vec0` 搬运并写回；普通 Vec 写回使用原路径。更大单 MAD K/N、部分有效行、FP32 矩阵输入、split-AIV 执行及其他搬运类型仍拒绝。

`requires_coupled_incore_dispatch()` 选择单核心混合计算路径。`supports_acc_to_gm_via_vec(FP32, FP32)` 表示显式暂存路径；A1 的 `supports_acc_to_gm_dtype()` 仍拒绝直接累加器写回。

A1 `pl.cast` 支持静态完整 FP16/FP32 ND Vec 类型互转，须显式指定 `mode="none"`。原生 FP32 到 FP16 缩窄按最近偶数舍入，覆盖有符号零、次正规数、溢出及 NaN 位置。两种类型的行均须保持 32 字节对齐。其他舍入模式、scratch 及类型组合仍拒绝。FP32 GEMM 结果可在同一耦合 kernel 内搬到 Vec 后转换成 FP16 输出。

`pl.matmul_acc(acc, lhs, rhs)` 支持同一窄矩阵配置的 FP32 原位累加；现有内存规划保持累加器存储一致。常量 `init_cond=True` 重新初始化，`False` 或省略条件则累加新的乘积。暂不支持 unit flags。复用矩阵操作数会延长 L0 生存期，所有同时存活缓冲区须满足配置预算。

静态对齐的 Mat 面板加载支持 K/N 更宽的普通 ND 输入。每条 MAD 仍限定 K=N=16；显式加载 16 宽面板，在 K 方向累加，并按对齐偏移保存各 N 面板。原生回归以单个耦合 kernel 计算 M 最大 1024 的 `M x 32 @ 32 x 32`，检查完整输出和 reload。此分块路径不代表支持更大的单 MAD 形状或动态/尾块。

A1 `pl.exp(tile)` 和 tile/tile `pl.div(lhs, rhs)` 支持默认精度（default precision）的完整静态 ND FP16/FP32 Vec tile。`high_precision=True` 除法会被拒绝。硬件近似运算支持尾部掩码与重复分批。张量形式的 SiLU 组合 `zero - x`、`exp`、与全一张量相加及除法；原生回归使用 `@pl.jit.incore`、完整编译、公开调用及产物重新加载。此验证仅覆盖激活构件，不代表完整 YOLO 模型或可移植 IEEE 极值语义。标量除法、广播和其他激活系列仍未开放。

A1 `pl.tile.full(shape, dtype, value)` 通过 `TEXPANDS` 将编译期标量填充到静态完整 ND FP16/FP32 Vec tile。SiLU 现在可在 kernel 内生成零和一，省去此前为这两个常量传入的主机张量。原生测试覆盖两种类型、尾部掩码及公开编译/调用/重新加载。运行期标量参数及通用行列广播仍需单独支持。
