# GPU 常驻曲面与多片 Class-A 工具

本版实现用户选定的两项增强：工程曲面常驻 GPU，以及更丰富的曲面表示和多片联合优化。沿用 `ModelAdaptor` / `ModelSession`；新后端为 `resident-engineering-v2`。原 `engineering-document-v1` 批处理接口和 `cuda-hood-v1` CuModel 会话保持可用。

## 运行与交付

```powershell
.\run.ps1 test-resident       # 18 项原生/GPU/模型工具验收
.\run.ps1 resident            # 三片 G3 面板、UV 开孔、缝合、STEP
.\run.ps1 perf-resident       # 一次性调用与常驻重复调用的实测
python scripts/resident_preview.py
.\run.ps1 test-engineering    # 原 20 项工程回归
.\run.ps1 test
.\run.ps1 lint
python scripts/resident_delivery.py --package
```

| 产物 | 内容 |
|---|---|
| [joint_g3_panel.step](output/resident-surfaces/demo/joint_g3_panel.step) | 三片五次面板、一个多边形 UV 开孔，3 面、0 实体 |
| [rational-skin.step](output/resident-surfaces/rational-skin.step) | 带非均匀节点和正权重的有理 B-spline 曲面 |
| [质量对比](output/resident-surfaces/demo/quality-preview.png) | 实际 GPU 求值的优化前后反射条纹、曲率梯度，共用色标 |
| [组合验收](output/resident-surfaces/demo/validation.json) | G3、形状偏差、独立密网格和 STEP 往返、实现摘要 |
| [18 项验收](output/resident-surfaces/acceptance.json) | 逐项结果及当前源文件 SHA-256 |
| [性能记录](output/resident-surfaces/benchmark/validation.json) | 启动耗时、重复调用耗时、传输量和共同输出一致性 |

源码与结果包为 `output/delivery/rust-cuda-resident-surfaces-v1.zip`；清单逐文件记录 SHA-256。既有三个交付 ZIP 保留，依赖、工具链、编译缓存和规划记录不入包。

## 1. 常驻执行

入口：[Rust worker](src/bin/surface_worker.rs)、[CUDA 内核](cuda/resident_surface.cu)、[Python 客户端](scripts/resident_surface.py)。

- 每个 worker 创建一个 CUDA context 和 stream，复用模块、固定工作区、曲面 arena 和基函数表。
- 128 个不可变曲面槽位；句柄绑定 session、slot 和 generation。fork 为设备到设备复制，drop 后旧句柄失效。
- 正权重 NURBS 控制点与权重驻留显存。CPU 为节点布局和采样参数编译局部 B-spline 基函数及一至三阶导数表；相同布局/采样缓存复用。
- `analyze` 默认只读回 **8 个 double / 64 字节**的诊断摘要。模型观察默认提供句柄、次数和形状；明确请求完整几何才读取控制网。
- `fetch` 在不可变句柄上缓存 CPU 副本。CPU B-rep 裁剪、缝合和 STEP 输出是明确的几何读回边界。
- NVRTC PTX 缓存按完整源码、编译选项和 GPU compute capability 区分；缓存键文件再核对完整源码/选项。`compilations` 统计 NVRTC 调用，不包含驱动内部 JIT；启动实测单独计时。
- 多片优化复用固定显存工作区和可执行 CUDA Graph。相同结构再次优化时，复用约束矩阵/基函数的显存数据，仅上传变化的头部和对象引用。CPU 约束分解也缓存。
- 最多 4096 次 worker 请求、64 份基函数表且基函数缓存不超过 64 MiB、32 种优化 graph 结构。超出资源域明确拒绝。

本机同一片五次曲面、17×17 采样的记录：旧一次性 API 调用中位数 **726.13 ms**；常驻 worker 启动 **613.77 ms**（PTX 缓存命中），随后完整采样网格调用中位数 **1.62 ms**，摘要调用 **0.61 ms**。预热后的 14 次调用新增上传量和控制网读回均为零。两条实现共同字段 0–44 的最大绝对差异为零。

这里测量的是旧进程/编译成本摊销后的重复工作流。完整网格与摘要的返回数据量不同，记录不作为独立 CUDA 内核吞吐量比较，也不推断其他模型/硬件的速度。

## 2. 曲面表示与局部细化

[NurbsPatch](scripts/nurbs_surface.py) 支持：

- 非周期、两端夹持、单位参数域的张量 B-spline / NURBS。
- 两方向次数分别为 1–7；每方向最多 32 个控制点。
- 正权重范围 `[1e-4, 1e4]`，控制坐标绝对值上限 `1e6 mm`。
- CUDA 有理位置、一至三阶混合 jets、法向、Jacobian、平均/高斯/主曲率、反射场。
- 解析计算曲面上的平均曲率梯度平方 `|∇S H|²` 和反射向量场梯度平方。
- CPU 或显存内齐次坐标节点插入；保留几何、更新基函数布局，拒绝超次数的内部节点重数。显存版本先编译线性插入映射，直接在 GPU 变换控制点/权重，无控制网读回。

`adaptive_quality` 从节点区间和基础网格开始，在曲率范围、法向变化或奇异采样处细分。达到深度或采样预算仍有未解决区域时返回 `unknown_budget_or_depth_limit`。它会显式读回诊断采样并报告次数；它不是连续域自交、规则性或曲率误差证明。高重数内部节点的三阶导数可能只分片存在，报告 `interior_parametric_c3`，不把单侧 jets 当作全局 C3。

## 3. 多片联合优化

[编译器](scripts/surface_network.py) 将 **2–6 片非有理 Bézier、总计最多 384 个控制点**的接缝和锁点放在同一个线性系统中。每片可使用 u0/u1/v0/v1 任一边界；支持固定比例、剪切和反向的局部参数映射，以及 G0/G1/G2/G3 对应的边界 jets。共享角点的条件同时进入系统，报告秩、依赖行和自由模态；冲突锁点明确拒绝。

这一联合优化范围是正权重 NURBS 求值/细化能力的子集；本版不提供一般有理多片、变化的非线性重参数化或任意多边角区的通用 G3 求解。

CPU 编译等式矩阵、依赖种子锁点的可行投影和零空间。GPU 从输入句柄取得当前控制网，执行联合投影，在零空间中搜索。每个候选同时检查采样规则性、有限数值范围和**相对输入控制网的累计位移预算**。输入对象不修改；失败不产生可见结果。

冗余锁点可能只对特定种子相容。CPU 额外编译相容性方向，GPU 用当前显存种子核查：完全锁定且已满足接缝的对象可以通过，完全锁定但接缝冲突的对象明确拒绝，不必读回种子，也不把矩阵冗余直接判成不可行。

目标是以下几何量的采样积分/面积平均，按声明长度尺度 L 和位移预算 B 归一：

```text
w_curvature * L^4 * mean_area(|∇S H|²)
+ w_reflection * L² * mean_area(|∇S R|²)
+ w_deviation * mean_area(|S - S_seed|²) / B²
```

它使用真实曲面导数和度量。反射场采用固定视线；这里优化反射场梯度，并非任意透视环境下反射图像的美学评分。默认权重可由模型在工具调用中选择，用户另行声明的交付要求始终固定。

优化采用有预算的 GPU 坐标搜索，不保证全局最优。每一步的正负候选和当前结果并行求值、在 GPU 选择；整次迭代结束读回 18 个 double 指标和结果句柄元数据，**中间控制网读回为零**。固定正基函数下，控制点最大位移同时给出同参数位置的曲面偏差上界；它不限制相邻面间曲率差，也不构成无自交证明。

Demo 将三片五次曲面联合约束为 G3，256 步优化目标由 **7.1495 降至 1.0880**；独立 33×33 网格的逐片平均曲率梯度平方之和下降约 **88.60%**。最大控制点位移 **4.4261 mm**，低于调用方 5 mm 交付上限；G3 jets 最大残差 **4.58e-11 mm**。STEP 双向采样往返误差约 **9.20e-12 mm**。这些是此组合样例的实测结果。

## 4. 模型工具与交付检查

[ResidentEngineeringRuntime](scripts/resident_runtime.py) 复用已有 `ModelAdaptor` 和 `ModelSession`，新工具可通过 observation 发现：

| 工具 | 结果 |
|---|---|
| `nurbs.create` / `surface.store` | 不可变显存曲面句柄 |
| `surface.refine` | 保形节点插入后的新句柄 |
| `surface.analyze` / `surface.adaptive` | 摘要或自适应诊断 |
| `surface.network_optimize` | 多片结果句柄与联合优化报告 |
| `surface.network_patch` | 选择结果中的一片 |
| `surface.network_check` | 明确读取几何，独立 CPU 接缝复核 |
| `topology.face` / `topology.trim` | 必要时读取几何后交给 CPU OCCT |

工具程序保持原子提交。失败程序新建但未引用的设备对象会释放；文档分支共用不可变对象，只有所有文档均不再引用时释放。模型主动调用优化工具，harness 不自动安排优化或因分数暂时变差而拒绝合法建模。

调用方可固定 network 的 STEP 交付规则。导出重新测量接缝、锁点和形状偏差，重新执行曲率/规则性诊断，并核对输出 B-rep 的实际支撑面、权重、节点和控制网。无关场景曲面不能代替实际交付对象通过检查。

Demo 是明确标记的反馈工具 fixture，**没有调用外部 LLM，也没有为新增工具训练 CuModel**。旧 CuModel 仍使用原 hood 驻留后端；本版不声称小模型已经学会通用 NURBS 建模。非线性设计意图和稳健拓扑仍使用 CPU。Class-A 工具和采样质量改善均不等于工业 Class-A 认证。

## 5. 验收范围

18 项新增验收覆盖：齐次细化保形、节点/权重负例、联合投影与冲突锁点、完全锁定种子的相容性、有理非均匀 NURBS 的原生 OCP D3 对照、精确圆柱曲率、曲率/反射梯度有限差分、自适应未知状态、64 字节摘要、预热零上传、句柄复用、PTX 复用、显存内细化、G3 与四片角点、CUDA Graph/约束计划复用、预算失败保留输入、原子资源清理、交付支撑面绑定及 STEP 往返。GPU NURBS jets 与原生 OCP D3 最大差异约 **4.43e-12 mm**。

另回归原 20 项工程验收；16 项 Rust 测试执行、Clippy 和 Python 编译检查通过。当前真实 SolidWorks 导入、完整特征依赖图、通用拓扑命名、任意 NURBS 多片优化和全局 Class-A 证明仍在本版范围之外。
