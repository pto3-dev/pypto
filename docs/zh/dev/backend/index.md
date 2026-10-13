# 后端

逐架构的差异行为，从 pass 中剥离出来。

pass 从不对 `BackendType` 做分支判断。所有与架构相关的内容 —— codegen 目标、运行时 API
名称、硬件冒险规避、跨核 layout 规则 —— 都由当前 `PassContext` 提供的 `BackendHandler`
回答。新增一个架构意味着新增一个 handler，而不是修改 pass。

| 页面 | 内容 |
| ---- | ---- |
| [BackendHandler：有原则的后端分派](00-backend_handler.md) | 该虚接口、pass 如何查询它，以及新增后端需要做什么 |

## 另请参阅

- [Pass、PassContext、PassPipeline 与 PassManager](../passes/00-pass_manager.md) —— handler 的来源。
- [PTO ISA 参考](../../reference/index.md) —— handler 所抽象掉的硬件差异。

## 实验性 Ascend310P 目标

`BackendType.Ascend310P` 独立标识 310P/A1 目标。其可执行后端（executable backend）尚未提供。
通过 `get_backend_instance`、`set_backend_type`、`ir.compile` 或 `RunConfig(arch=...)` 选择该目标，
会在代码生成或运行时分派前抛出 `ValueError`。被拒绝的全局选择保留原配置，之后仍可选择已支持的目标。
显式请求 310P 时，同时传入旧的 A3/A5 平台也会报错，不能覆盖该目标。现有后端枚举值保持不变。

此标识是 A1 适配的第一个增量步骤，尚未启用 PTOAS A1 代码生成、ISA 指令、模拟器执行或
Simpler 运行时。后续特性须实现这些能力后才能解除选择限制；310P 不得静默回退到 A3 后端。

`is_backend_available(BackendType.Ascend310P)` 返回 `False`，不会配置或实例化后端。
内存图查看器等提供后端选择的工具使用此可用性查询（availability query），排除尚未提供实现的实验性目标。
`BackendType.__members__` 仍列出所有已识别的目标标识。
