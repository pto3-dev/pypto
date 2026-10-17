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

`examples/models/yolo26_native/executor.py` 为融合后的 80 类、无 NMS 的 YOLO26-nano 权重记录静态单批次原生执行序列。所有依赖输入的布局转换、卷积、注意力、特征路由、解码及 TopK 均在 Ascend310P 上执行。CPU 准备阶段加载、融合及打包常量权重；回放阶段通过一个 `A1L2Worker` 提交静态设备模型，保持显式传输及设备常驻中间张量。不支持的模块会报错。`RejectHostOperators` 通过拒绝主机 ATen 算子审计回放过程。

以 `NativeYOLO26(fused_half_model, 640, output_dir, device_id=0)` 创建上下文管理器，并向 `forward` 传入连续的 CPU FP32 `[1,3,640,640]` 输入。准备阶段编译精确签名，同一常驻会话内共享相同的内核镜像。输出为 FP32 `[1,300,6]` 检测结果。640x640 配置已针对变化输入比较全部 23 个特征输出、每个连续预测及最终 TopK 语义；使用 FP16，Cube 累加及 Softmax 使用 FP32。硬件图捕获尚未支持。

128x128 几何使用下述组合 PTO2 精度策略完成验证。直接调用 NativeYOLO26 时设置 four_way_accumulation=True、linear_query_key=True 和 precise_small_softmax=True。六组合成输入的全部 23 个特征、两个原始输出头、连续分数及最终检测均通过未放宽的 PTO2 检查，严格概率检查也通过；旧 Cube 策略仍有已知的 128 精度差异。

### 原生 YOLO26 验证与性能

在代码检出、本地 Ultralytics 依赖及配套 Ascend310P 工具链就绪后，运行 `python -m examples.models.yolo26_native.profile --weights path/to/yolo26n.pt --size 640 --output-dir build_output/yolo26-native --iterations 10`。基准程序在融合权重前选择一对一检测头，验证两组 [0,1] 均匀随机合成输入，并审计原生回放期间的主机 ATen 算子数为零。程序比较全部 23 个特征张量、连续检测框/类别概率和最终检测分数；概率容差保持 rtol=0.03、atol=1e-6。TopK 成员及排序分数相对原生连续预测精确校验，允许相等分数的遍历顺序与 Torch 不同。数值断言之前保存包含严格超限计数的诊断数据。

报告包含权重/原生源代码哈希、Ultralytics 版本、实际 CANN 逻辑/物理设备身份、原生调度覆盖范围、原始样本及预热后的中位数/均值/p95 延迟和 FPS。完整前向计时包含输入输出复制和同步主机调度；常驻回放不包含这些复制，包含调度/同步。原生准备时间包含执行序列/JIT 创建、常量打包/上传及分配，不包含权重文件加载和 CPU 融合。计时是未采用独占队列隔离的主机墙钟观测。默认配置为批次 1、640x640；128x128 配置默认选择已验证的 PTO2 精度策略。

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

相同的 `four_way_accumulation` 策略也适用于对齐的 3x3 逐通道卷积。
九个乘积进入四个 FP32 向量部分和，第九项加入第零部分和，按序合并后再加偏置并
舍入为 FP16。原生模型将此策略同时传递给普通卷积和逐通道卷积。
未选择此策略时，逐通道卷积保留原来的顺序归约。

可选 `MatrixPlan(..., linear_accumulation=True)` 按 K 递增顺序执行独立的 FP32
乘法和加法，最终一次舍入为 FP16。两种 B 布局均受支持，运算使用设备向量及标量
指令，默认仍使用 Cube 策略。`NativeYOLO26(..., linear_query_key=True)` 仅为 QK
矩阵乘选择此路径，对齐已测试的批量转置布局 CPU 参考归约顺序，不更改缩放、
激活舍入或精度预算。

`NativeYOLO26(..., precise_small_softmax=True)` 为 16x16 有限 FP16 分数选择
设备补偿 Softmax（对应 128 输入的注意力形状）。FP32 高位/残差数对在区间缩减
指数、求和及归一化过程中保留舍入残差，最终舍入为 FP16，处理靠近 FP16 中点的
概率，不改变容差，也不在 CPU 上计算激活。默认保持原 DSL 路径，并继续用于更大
注意力形状。

全原生性能 CLI 默认 `--precision-policy auto`：128x128 选择组合 PTO2 策略
（普通及逐通道卷积四路归约、QK 顺序归约、小形状补偿 Softmax），640x640 保持
Cube 及常规 DSL Softmax。`--precision-policy cube` 显式保留旧的 128 行为，其
CPU 参考精度尚未通过验收。`--precision-policy pto2` 仅在 128 形状验证。
报告记录所选策略，并保留全部既有严格概率断言。参考归约策略以吞吐换取
可复现性，不能将它与稠密 Cube 策略的耗时作为相同执行策略直接比较。

`A1ResidentSession.prepare_sequence(calls)` 在检查会话归属及静态签名后，绑定
有序的 `(prepared_kernel, resident_arguments, blocks)` 元组，准备时只上传一次描述符。
直接序列 API 仍由宿主逐 kernel 提交到同一流并在末尾等待；原生 YOLO26 使用下述
设备模型 worker。输入输出保持显式传输，算术及精度容差不变。

四路卷积的 Vector 后备路径根据几何形状和核占用率，每个工作项处理四个或八个相邻输出像素，通过带掩码的重复指令将权重行及
偏置广播到空间块内的像素。每个像素独立保留四个 FP32 部分和及原有的按序合并、偏置和
FP16 激活边界。静态矩形、步长一/二、逻辑通道尾部及输出保护区保持支持。
此方式复用权重、提高向量重复利用率，不改变精度策略。

原生检测 TopK 在两级各使用容量 300 的最差候选优先堆，随后排序保留的候选。
相等分数保持原遍历顺序：第一级按锚点顺序，第二级按选中锚点的排名及类别顺序。
两次带掩码的向量最大值仅归约 80 个类别分数，排除检测框坐标及填充。
此方式替代标量扫描与插入移位，不改变候选身份、分数、输出几何或精度预算。

检测 TopK 将已排序锚点的最大分数作为上界：下一行的最大分数无法优于
当前保留类别候选中的最差项时，跳过所有剩余行。比较同时考虑遍历索引，
因此相同分数仍保持原有的精确两阶段语义。锚点不超过 384 个时，通过一次
GM 到 UB 搬运缓存 [anchors,96] 密集输入，复用于最大值、类别选择和最终框收集。
缓存满载占用 UB 字节 65536..212991，不与堆、元数据或输出重叠。
更大输入保留逐行 DMA。两条路径均不改变数值或使用宿主选择；单核任务和
输出张量 ABI 保持不变。

四路卷积为每 16 个输入通道一次加载连续的打包权重块，统一转换为 FP32 后按原顺序消费每四项。
末块传输仅覆盖既有的四项对齐填充，不会越过当前输出通道面板；权重缓存占用的私有 UB 截止字节为 82944。
累加顺序、尾项、偏置与激活舍入、打包权重 ABI 和校验容差均保持不变。精确测试覆盖重复通道块、
一/二/三项尾部、变化输入、产物重载与输出保护区。

四路卷积将每个像素的四个 16 通道部分和放在连续的 64 个向量位置中，一条向量指令即可把
缓存中的四行权重广播到空间块内的像素。跨步的 16 元素加法保留从左到右的部分和合并顺序，尾项仍只加入
第零部分和。此优化仅改变私有向量布局，不改变输入输出布局、打包权重 ABI、算术顺序或校验容差。

四路路径对每个通道块加载的 FP16 补丁执行一次展开，使用 310P scatter transpose 的重复源行。
展开结果将空间像素放在每个通道内部，通过跨步向量复制构造完整四项分组，避免标量激活读取。
FP16 到 FP32 的转换精确无损，原有的有序 FP32 归约及标量尾项保持不变。
新增私有 UB 存储截止字节为 167936，仍在 310P UB 容量内。
单次转置的 1x1 补丁将两个重复步长都设为零；精确测试覆盖 1x1 的完整及部分通道块和重复的 3x3 通道块。

仅包含一至三个 CHW 尾项的通道块准备标量 FP32 补丁缓冲。
对齐块加载所有有效 FP16 补丁元素后直接展开，跳过未使用的清零和标量补丁转换。
向量到加载的屏障及精确尾项路径保持不变。

对于补齐后归约项不超过 512 的 1x1 卷积，每核跨空间工作项保留一个完整的 FP32 输出通道面板。
遍历以输出面板为先，在面板改变时刷新缓存，四项归约顺序保持不变。
FP32 缓存截止 UB 字节 229376；FP16 暂存缓冲仅与当前未启用的逐块缓存重叠。
更大的面板及 3x3 卷积保留原有调度和块缓存。
精确测试覆盖缓存复用、面板刷新、尾项及 512 项边界两侧，既有预期和容差保持不变。

有界 3x3 特征图每核仅暂存一次到 NC1HWC0 格式的 L1 存储。
硬件 img2col 将对齐的十六像素补丁提取到私有 UB 暂存块，每个工作项选择其中四个有效像素。
图像跨空间和输出通道工作项复用；超过保守 512 KiB L1 预算的图像保留 GM 收集路径。
显式设置零填充，FP32 归约顺序及 FP16 激活边界保持不变。
九个补丁块使用互不重叠的 UB 槽，截止字节为 90624，因此整批提取仅需一对生产者/消费者屏障。

输入打包按有效 NCHW 通道执行向量 FP32 到 FP16 转换，再用散射转置将补零的 16x16 半精度块变为 NHWC16。
单次重复步长为零，减少标量激活读取、类型转换及填充写入。
精确窄化、有符号零及通道填充保持不变；测试覆盖 1/3/16 通道及半精度舍入中点。

卷积通过物理输入行步长和偏移元数据直接消费对齐通道切片，同时保留独立的逻辑通道和权重几何。
普通卷积之前无需物化切片复制；深度卷积仍使用紧凑输入路径。
Cube 和有序向量内核均支持切片输入。L1 暂存仅打包有效通道窗口，DMA 步长超过 16 位时回退到 GM 收集。
精确测试将切片外通道设为 NaN，并保留输出保护区检查。

Cube 注意力乘积使用两个 1 KiB L1 槽。MTE1 释放槽后，MTE2 预取后续 A/B 面板，同时 MAD 使用独立的 L0 操作数。
独立的 MAD 到 MTE1 屏障保护 L0 复用；包括最后一至两个块在内的所有槽就绪和释放标记均被消费。
对齐行采用批量 DMA，源行间隔过大时保留逐行传输。
K 遍历、MAD 顺序、转置解释及累加器写回保持不变。

有序注意力乘积根据行面板数及启动核数，每个工作项选择两行或四行输出，小网格仍为模型的八个核保留足够工作。
批量行加载与半精度转置为共享 FP32 权重广播提供数据，每个输出仍严格按递增 K 顺序累加乘积。
未使用行被初始化为零且不会写回；精确批量 golden 测试覆盖两种转置模式及单核/八核启动。

有序卷积通过一次跨步 MTE3 DMA 写回整个空间块，保留输出通道填充与输出偏移。
ConvPlan 将 DMA 行间隔限制为最多 63 个块。向量与存储之间的双向屏障继续保护舍入缓冲区，
FP32 归约和 FP16 激活边界保持不变。已有精确卷积测试覆盖通道填充、多输出面板、重复像素及输出保护区。

缓存权重的 1x1 卷积在八像素工作项数量足以覆盖所有启动核时，每个工作项计算八个空间像素；
小网格及其他形状保留四像素。每个输出仍采用原有四路有序 FP32 部分和及标量尾项，并保留全部 FP16 激活舍入边界。
较大的累加器与转置暂存仍位于 UB 内，FP32 展开缓冲与半精度暂存分离；每次转换最多重复 144 次。
已有精确测试覆盖两种块宽、非整块输入通道、多输出面板及通道视图。

有序卷积在入口按已校验的 1x1 或 3x3 核大小进行特化，将卷积面积变为常量，
从补丁收集及有序归约项的地址生成中移除通用除法、取模及未使用的几何分支。
空间块选择、运行时步长及填充、打包权重、缓冲布局和全部 FP32/FP16 算术边界保持不变；
两种卷积核继续覆盖精确通道尾项、源视图及输出保护区。

原生模型可显式设置 `weight_l2_bypass=True`；profile CLI 在 128x128 PTO2
精度策略下提供 `--weight-l2-bypass`。会话初始化后，准备阶段针对当前可见的
运行时设备查询 `rtGetL2CacheOffset`，通过四个精确表示的 uint16 元数据字传递
偏移，应用于有序卷积的只读权重指针。设备序号与偏移均不硬编码；偏移不可用或
请求稠密 Cube 精度策略时显式失败。激活、偏置、输出写入及其他算子保持正常缓存策略。

默认仍使用缓存权重；旁路执行成功不代表获得加速。应使用相同 CANN profiler
和未修改的精度检查比较预热后的调用。缓存 PMU 单独采集并保留原始事件，单一
L2 命中率不足以区分所有旁路事务。并发任务会影响耗时和芯片级 DHA 计数，即使
已按本模型的 stream/task ID 关联统计。此 A1 原生路径显式传递运行时偏移，不依赖
SDK 隐式全局配置，也不代表 A1 已接通前端 `cache_policy`。

四路精度策略中，输入通道为 16 的整数倍且不超过 512 的步长为一的 1x1 卷积
采用有序 Cube 内核。每次 K16 MAD 仅有一行非零权重，因而只向对应的四路
FP32 部分和之一累加一个 FP16 乘积。保留有序合并、偏置及 FP16 SiLU 舍入边界；
其他形状及源 DMA 间隔超过 16 位的情况使用 Vector 后备路径。根据整除条件及可用工作量选择 128、64、32 或 16 个
空间像素，每个核在 L1 复用当前输出通道的权重面板。四路 L0C 部分和最多占用
32 KiB，L1 面板最多占用 256 KiB，输入块单独占用至多 4 KiB。

有序 1x1 Cube 权重在 GM 中保留紧凑 CHW 常量。每次内核用独立的 8 KiB UB
零块初始化有界稀疏 L1 面板，再通过跨步 GM 到 L1 搬运，把十六行紧凑权重散射到
对应的 K16 对角行；面板在空间工作项间保持常驻。此方式移除原来的 16 倍 GM
膨胀，不改变算术、源视图、输出保护区或权重 L2 bypass。打包常量张量签名由
`co * ci * 16` 改为 `co * ci` 个 FP16 元素。使用新布局时应重新编译并打包旧图；
已有编译产物继续保持原来的张量签名。

有序 1x1 Cube 交替使用两个 512 字节 L0B 缓冲区。Cube 消费当前缓冲区时，
MTE1 预取下一通道的稀疏权重块；独立的就绪与释放事件保护两个缓冲区。
固定的十六通道循环展开后，缓冲区地址、事件和部分和选择均为静态值。
切换输入块前排空两个缓冲区。逻辑 MAD 顺序、四路 FP32 部分和、
紧凑 GM/L1 布局及 FP16 舍入边界保持不变。

有序 3x3 卷积的 Vector 后备路径在网格至少为每个启动核提供一个工作项时同样选择
八个空间像素，小网格保留四个。每个 im2col 补丁及转换后的权重块复用于两倍的输出，
不改变单个输出的归约或舍入。八像素 3x3 展开缓冲终止于 UB 字节 212992，未超过
310P 容量。该区域与仅用于 1x1 的权重缓存重叠，3x3 内核不会启用该缓存；其他
活跃缓冲保持分离。源视图、通道尾块、步长二图像、超大图像收集及标量归约尾项
继续保持原有行为。

四路策略在完整 NC1HWC0 图像与稀疏权重面板合计不超过 Ascend310P 的 1 MiB L1、且源 DMA
间隔不超过 16 位时选择有序 3x3 Cube 内核。GM 权重继续使用紧凑 CHW 布局：
先用零块初始化本地 L1 面板，再通过跨步 GM 到 L1 搬运，将逻辑权重行散射到
对应的 K16 行。此路径不分配膨胀的 GM 权重；超出预算的形状保留 Vector 后备路径。

硬件随路 im2col 直接从常驻 L1 图像向九个 L0A 补丁供数，空间块为 16/32/64/128
个像素。内核按逻辑 CHW 顺序消费通道与卷积位置，保留四路 FP32 部分和，并将
标量尾项归入第零路。偏置及 FP16 SiLU 舍入边界保持不变。九个 L0A 补丁最多
占用 36 KiB，四路 L0C 部分和最多占用 32 KiB。核切换输出通道块前复用 L1 面板。
继续支持源视图、输出保护区、填充通道、产物重载及显式权重 L2 bypass。

有序 Cube 的容量检查包含按 512 字节对齐的图像以及每个本地稀疏权重行。CANN
对 Ascend310P3 报告 `L1_SIZE=1048576`。此容量允许 128x128 RGB 首层和
128 通道 3x3 面板使用常驻 L1，不搬出或重新舍入部分和。超过 1 MiB 的形状
保留后备路径；容量检查同时计入输入通道填充。

完整稀疏权重面板超过 L1、但图像与两个有界缓冲块可容纳时，有序 3x3 内核按
每块 32 个输入通道流式处理。每块占 144 KiB。MTE1 和 Cube 消费当前块时，
MTE2 预取下一块；独立的就绪及释放事件避免覆盖，下一个空间工作项前排空两块的
生命周期。图像始终常驻复用。四路 FP32 部分和跨块保留，包含最后奇数 C0 块及
标量 CHW 尾项，不搬出或窄化部分和。可容纳的完整面板仍按输出块缓存；两种预算
均不满足时保留 Vector 后备路径。两种 Cube 路径的 GM 权重均保持紧凑布局。

仅当最小 M16-by-N16 网格至少提供八个工作项时选择流式 Cube。更小的网格无法
占满 310P3 的八核；实测 4x4、256 到 16 通道的检测头卷积继续使用更快的 Vector
调度。此利用率条件只改变调度，不改变累加或精度预算。可常驻的完整面板保持原策略。

有序 1x1 和 3x3 Cube 内核在输出网格仍可为每个启动核提供一个工作项时选择
M128，小网格保留 M64/M32/M16。九个空间 L0A 补丁最多占 36 KiB，低于
310P3 的 64 KiB L0A；四路 L0C 部分和占 32 KiB。UB 尾处理缓冲块扩展至
8 KiB，并与字节 65536 的零块及字节 131072 的元数据保持分离。此方式增加复用，
不改变任何输出的累加顺序或 FP16 边界。

### A1 静态 L2 任务图

`A1L2Worker(compiled, device_id=0)` 管理单芯片的设备常驻张量与已注册 kernel。
使用 `prepare(compiled)` 准备静态 kernel，再调用
`register_graph([(kernel, arguments, blocks), ...])` 和 `run(graph)`。
图绑定不可变的设备地址，输入数值通过显式拷贝更新。A1 后端准备 CANN 设备模型，
每次重放只提交一次模型，依赖按设备流顺序执行，逐 kernel 的准备调用不计入重放。
关闭 worker 会先等待执行完成，再释放模型、kernel 与张量，并使图句柄失效。
跨 worker 或签名不匹配的绑定会被拒绝。

此静态耦合核心后端与通用 `ChipWorker` 不同，尚未移植 A2/A3 的 TensorMap/AICPU
动态编排。runtime 版本必须提供 `A1SingleLaunch.prepare_graph`。

原生 YOLO26 执行器将整个任务序列注册给一个 `A1L2Worker`，并重放准备好的设备模型。
覆盖报告区分设备计算任务与宿主提交：216 个计算任务仍然保留，但每次回放只有一次模型提交，
没有逐任务的宿主 kernel launch。张量地址、算术和精度容差保持不变。

启动或停止 CANN profiling 后，调用 `graph.rebuild()` 更新模型任务的采集状态。
YOLO26 执行器通过 `native.rebuild_graph()` 提供相同操作。图句柄、任务顺序和
设备常驻绑定保持不变；重建不会执行模型算术。回放计时必须排除重建及其后预热。
如果设备模型在 profiling 启动前已经准备，CANN 9.1 的流水线计数可能保持为零。
此接口要求 runtime 固定版本提供 `A1PreparedSequence.rebuild`。
