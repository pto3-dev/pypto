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
DDR–Vec DMA 路径。转换、BF16、原子操作、GEMM 和跨核操作需要独立验证；当前扩展见下文。

编译时使用 `backend_type=BackendType.Ascend310P`，默认编译平台为 `a1`。
显式选择 310P 后，旧 A3/A5 平台参数不能覆盖目标。需要使用配套的 A1
PTOAS 和 PTO-ISA 特性分支。后端可用性表示编译支持。仅包含一个静态
InCore 核函数的编译产物可通过 `compiled(*tensors)` 调用；输入为连续的
CPU FP16/FP32 张量。也可使用 `RunConfig(platform="a1", device_id=...)`
指定设备。同步执行路径使用锁定的 A1 ISA 和张量描述符运行时；图编排、
DFX 和模拟器调度尚未开放；显式设备常驻张量可使用下述 `A1ResidentSession`。

A1 带宽性能提示在完成校准之前保持关闭。

静态 ND 分区可读取和写回少于父张量宽度的列。行步长与列偏移须按 32 字节对齐，分区须位于源视图范围内。同步调用仍接收连续的 CPU 父张量，输出中分区以外的列保持原值。

静态完整的 FP16/FP32 向量 tile 无需凑齐 256 字节算术重复。加、减、乘对最后一次重复使用掩码，DMA 行仍须按 32 字节对齐。因此可执行单块 tile，动态 valid shape 和未对齐列尾仍未开放。

限定矩阵配置支持 K=N=16、16 至 2048 之间按 16 对齐的静态完整 M 行，使用 FP16 Mat 输入及 FP32 累加器。矩阵/向量混合函数保留为单个耦合 AIC callable。内存空间推导在 Acc 写回前插入显式 FP32 ND Vec 搬运，由正常的存活期及地址规划管理暂存，再以 `single_mode_vec0` 搬运并写回；普通 Vec 写回使用原路径。更大单 MAD K/N、部分有效行、FP32 矩阵输入、split-AIV 执行及其他搬运类型仍拒绝。

`requires_coupled_incore_dispatch()` 选择单核心混合计算路径。`supports_acc_to_gm_via_vec(FP32, FP32)` 表示显式暂存路径；A1 的 `supports_acc_to_gm_dtype()` 仍拒绝直接累加器写回。

A1 `pl.cast` 支持静态完整 FP16/FP32 ND Vec 类型互转，须显式指定 `mode="none"`。原生 FP32 到 FP16 缩窄按最近偶数舍入，覆盖有符号零、次正规数、溢出及 NaN 位置。两种类型的行均须保持 32 字节对齐。其他舍入模式、scratch 及类型组合仍拒绝。FP32 GEMM 结果可在同一耦合 kernel 内搬到 Vec 后转换成 FP16 输出。

`pl.matmul_acc(acc, lhs, rhs)` 支持同一窄矩阵配置的 FP32 原位累加；现有内存规划保持累加器存储一致。常量 `init_cond=True` 重新初始化，`False` 或省略条件则累加新的乘积。暂不支持 unit flags。复用矩阵操作数会延长 L0 生存期，所有同时存活缓冲区须满足配置预算。

静态对齐的 Mat 面板加载支持 K/N 更宽的普通 ND 输入。每条 MAD 仍限定 K=N=16；显式加载 16 宽面板，在 K 方向累加，并按对齐偏移保存各 N 面板。原生回归以单个耦合 kernel 计算 M 最大 1024 的 `M x 32 @ 32 x 32`，检查完整输出和 reload。此分块路径不代表支持更大的单 MAD 形状或动态/尾块。

A1 `pl.exp(tile)` 和 tile/tile `pl.div(lhs, rhs)` 支持默认精度（default precision）的完整静态 ND FP16/FP32 Vec tile。`high_precision=True` 除法会被拒绝。硬件近似运算支持尾部掩码与重复分批。张量形式的 SiLU 组合 `zero - x`、`exp`、与全一张量相加及除法；原生回归使用 `@pl.jit.incore`、完整编译、公开调用及产物重新加载。此验证仅覆盖激活构件，不代表完整 YOLO 模型或可移植 IEEE 极值语义。标量除法和其他激活系列仍未开放。

A1 `pl.tile.full(shape, dtype, value)` 通过 `TEXPANDS` 将编译期标量填充到静态完整 ND FP16/FP32 Vec tile。SiLU 现在可在 kernel 内生成零和一，省去此前为这两个常量传入的主机张量。原生测试覆盖两种类型、尾部掩码及公开编译/调用/重新加载。运行期标量参数仍需单独支持。

A1 对静态完整 FP16/FP32 tile 开放带完整大小同类型临时缓冲区的 `pl.tile.row_sum`/`row_max` 及 `pl.tile.col_sum`/`col_max`。行结果保留紧凑列主序（packed column-major）`[M, 1]` 布局并通过对齐连续存储写回，M 须保证结果满足 32 字节对齐。列结果为 `[1, N]`；列求和目前拒绝可选的二叉树临时缓冲区形式。行归约缓冲区须互不重叠；列归约结果可复用首个输入行的起始地址，但不允许部分重叠。所有缓冲区须满足总 UB 预算。运算保持 tile 数据类型，注意力求和应使用 FP32 累加。公开原生回归检查输入变化、产物重新加载及对齐输出分区外的数据保留。暂不支持动态形状、未对齐尾部或特殊数值语义保证。

Ascend310P 还支持 `tile.row_expand` 将紧凑 Mx1 归约结果广播、`tile.col_expand` 将行主序 1xN 归约结果广播到静态、对齐的 FP16/FP32 Vec 分块。可组合 `sub`、`exp`、行/列 `sum` 和 `div` 实现稳定 FP32 softmax。行源必须与目标不重叠；列源可与目标首行精确同址。暂不支持融合广播算术、动态或未对齐形状。

`examples/models/yolo26_310p_softmax.py` 使用本地 Ultralytics 权重验证 YOLO26-nano 的增量集成。两个注意力 softmax 模块通过 PTO3 FP32 内核在 310P 执行；卷积、矩阵乘和其余模型仍在 CPU 执行。安装 Ultralytics 后传入 `--weights path/to/yolo26n.pt --size 128 --output-dir build_output/yolo26-hybrid`，也可选择 `--size 640`。脚本对比两个输入的全部张量输出，并将包含分配和传输的预热混合模型、softmax 分块主机耗时与编译/会话创建耗时分开记录。这些数据不代表设备内核计时或完整 PTO3 模型 FPS。完整 640x640 原生模型执行由下述独立执行器提供。

Ascend310P 独立 `@pl.jit.extern(core_type="aic", source=...)` 入口在特化时保留源文件和包含目录元数据，并编译为耦合核心包装器。相对同级包含仍以原始外部源文件为根。公开 CPU 载体调用和产物重载使用相同包装路径。运行时固定版本还提供显式设备常驻张量及多内核镜像注册；设备调用仍为同步形式。

`A1ResidentSession(compiled)` 准备独立 Ascend310P 产物。通过 `session.upload`/`empty` 创建设备张量，通过 `session.prepare(compiled)` 绑定各个精确静态签名。准备后的内核只接受匹配且属于同一会话的张量，可指定 `blocks=1..8`，不会隐式传输激活。张量的 `copy_from`/`to_cpu` 显式执行输入输出传输。源文件及包含内容相同的镜像在会话内共享注册。关闭会话时先排空流，再释放存储。此接口属于同步主机调度，不是设备图捕获。

`examples/models/yolo26_native/operators.py` 中的原生 YOLO26 算子计划支持静态 FP16 NHWC 卷积，卷积核为 1/3、步长为 1/2。通道填充、设备端边界采集、FP32 Cube 累加、偏置和可选 SiLU 均在 Ascend310P 上执行；CPU 权重打包仅用于常量预处理。输出像素数必须是 16 的倍数。完整模型的验证及计时由下述独立原生运行器提供。

`examples/models/yolo26_native/layout.py` 支持原生 FP32 NCHW 输入到 FP16 NHWC 的打包、对齐通道切片/拼接、残差加法、最近邻 2 倍上采样、5x5 最大池化和连续 FP16/FP32 转换。静态行步长和偏移使特征图路由在 310P 上完成，并支持带半精度舍入的注意力查询缩放。针对性的常驻张量测试覆盖八个计算块、变化输入、产物重载和相邻保护区。

`examples/models/yolo26_native/depthwise.py` 提供对齐 NHWC FP16 的 3x3 逐通道卷积，支持填充 1、步长 1/2、FP32 累加、偏置和可选 SiLU。常量权重按 [卷积核高度,卷积核宽度,通道] 顺序打包；空间采集及全部激活算术均在 310P 上执行。常驻张量测试覆盖变化输入、产物重载和输出保护区。

`examples/models/yolo26_native/matrix.py` 支持原生静态 FP16 矩阵乘法、FP32 Cube 累加及可选的右侧输入转置。16x16 工作块在 310P 上流式处理对齐的行主序面板。YOLO26 注意力配置覆盖 N=16/400、K=32 的 QK 乘法及 N=16/400、输出通道 64 的概率/值乘法；输入输出均保持设备常驻。 原生 Cube 内核在复用累加器前等待向量侧读取完成；同一生命周期修复也用于卷积的重复工作块。

`examples/models/yolo26_native/detection.py` 将无 NMS 的 YOLO26 检测头后处理保留在 310P 上：逐层 FP16 检测框解码、80 类带半精度舍入的 Sigmoid，再筛选 300 个候选锚点及最终 300 个类别/锚点结果。连续输出使用 FP32 [锚点数,96] 存储，最终检测结果为 [300,6]。TopK 对相等分数采用确定的候选遍历顺序，并由单个原生计算块执行；激活选择不在主机端执行。

### 完整原生 YOLO26-nano 执行器

`examples/models/yolo26_native/executor.py` 为融合后的 80 类、无 NMS 的 YOLO26-nano 权重记录静态单批次原生执行序列。所有依赖输入的布局转换、卷积、注意力、特征路由、解码及 TopK 均在 Ascend310P 上执行。CPU 准备阶段加载、融合及打包常量权重；回放阶段采用同步主机调度、显式传输及设备常驻中间张量。不支持的模块会报错。`RejectHostOperators` 通过拒绝主机 ATen 算子审计回放过程。

以 `NativeYOLO26(fused_half_model, 640, output_dir, device_id=0)` 创建上下文管理器，并向 `forward` 传入连续的 CPU FP32 `[1,3,640,640]` 输入。准备阶段编译精确签名，同一常驻会话内共享相同的内核镜像。输出为 FP32 `[1,300,6]` 检测结果。640x640 配置已针对变化输入比较全部 23 个特征输出、每个连续预测及最终 TopK 语义；使用 FP16，Cube 累加及 Softmax 使用 FP32。硬件图捕获尚未支持。

额外的 128x128 诊断几何能够执行完整序列，但目前相对严格 CPU 参考仍有 FP16 概率差异，不纳入 640x640 配置的精度验证结论。

### 原生 YOLO26 验证与性能

在代码检出、本地 Ultralytics 依赖及配套 Ascend310P 工具链就绪后，运行 `python -m examples.models.yolo26_native.profile --weights path/to/yolo26n.pt --size 640 --output-dir build_output/yolo26-native --iterations 10`。基准程序在融合权重前选择一对一检测头，验证两组 [0,1] 均匀随机合成输入，并审计原生回放期间的主机 ATen 算子数为零。程序比较全部 23 个特征张量、连续检测框/类别概率和最终检测分数；概率容差保持 rtol=0.03、atol=1e-6。TopK 成员及排序分数相对原生连续预测精确校验，允许相等分数的遍历顺序与 Torch 不同。数值断言之前保存包含严格超限计数的诊断数据。

报告包含权重/原生源代码哈希、Ultralytics 版本、实际 CANN 逻辑/物理设备身份、原生调度覆盖范围、原始样本及预热后的中位数/均值/p95 延迟和 FPS。完整前向计时包含输入输出复制和同步主机调度；常驻回放不包含这些复制，包含调度/同步。原生准备时间包含执行序列/JIT 创建、常量打包/上传及分配，不包含权重文件加载和 CPU 融合。计时是未采用独占队列隔离的主机墙钟观测。默认配置为批次 1、640x640；可选的 128x128 诊断配置保留其已知严格数值失败。

原生基准按 PTO2 的顺序融合 BatchNorm 常量：
`scale = gamma / sqrt(var + eps)`、`weight *= scale`、
`bias = (bias - mean) * scale + beta`，随后转换为 FP16。
CPU 参考及原生普通/逐通道卷积的 SiLU 均先将 Sigmoid 结果舍入为 FP16，再相乘，
这与只在输出舍入一次的融合 SiLU 不同。准备阶段仅计算常量；回放中的全部激活运算
仍在 Ascend310P 上执行。此算术对齐特性不放宽现有校验容差，也不代表所有输入配置
均已完全通过 PTO2 的生产精度契约。

`ConvPlan(..., four_way_accumulation=True)` 可选择原生向量卷积，按逻辑 CHW
顺序使用四个独立 FP32 部分和。末尾一至三项先依次加入第零部分和，再按顺序合并，
与 CPU FP16 golden 的归约策略一致，模型算术仍在设备上执行。
权重按输出通道块/CHW/输出通道内层打包，K 补齐至四的倍数。
`NativeYOLO26(..., four_way_accumulation=True)` 为普通卷积选择此路径，默认仍使用
Cube 累加。向量策略以吞吐换取可复现的累加顺序，其他算子保留各自的舍入契约。
