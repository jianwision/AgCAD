# Rust + CUDA 几何建模 demo

新增 [GPU 常驻曲面与多片联合优化](RESIDENT_SURFACES.md)：正权重 NURBS、显存内节点插入、
全四边 G1/G2/G3 约束和角点耦合、CUDA Graph 几何质量优化。运行 `./run.ps1 resident`；
输出三面开孔 G3 STEP，18 项新增验收通过。独立密网格曲率梯度指标改善约 89%，具体数学范围见文档。

已有 [Class-A / 设计意图 / 拓扑容差支持](ENGINEERING_SUPPORT.md)：三至七次 Bézier、
G1/G2/G3-compatible 匹配、GPU 曲率/反射诊断和平顺、带单位的非线性约束、
容差预算下的裁剪/布尔/分割/缝合与修复。新增工程文档后端通过同一个 ModelAdaptor
调用类型化工具程序。运行 `./run.ps1 engineering` 和 `./run.ps1 test-engineering`；
交付 `output/engineering/demo/class_a_panel.step`，20 项新增验收通过。
限定曲面工具支持不构成整车 Class-A 认证；非线性求解和 robust B-rep 拓扑明确使用 CPU。

当前架构 review 已重写为 [ModelAdaptor 与约束执行设计](SELF_IMPROVEMENT_REVIEW_V2.md)：
`Llm` 与 `CuModel` 实现同一个 `ModelAdaptor`，harness 执行用户/几何/资源约束，
质量评分作为反馈，允许合法探索。持久 `ModelSession`、Rust/CUDA worker、
草稿修复、分支与模型切换已实现，详见 [MODEL_SESSIONS.md](MODEL_SESSIONS.md)。
LLM 可由回调或 HTTP JSON bridge 接入；文件回放单独标记为 `ReplayModel`。

```powershell
.\run.ps1 session --require-goal # 常驻 GPU 会话、修复/切换与 STEP
.\run.ps1 test-session          # 实际 CUDA 与本地 HTTP fixture 验收
.\run.ps1 train-proposals  # 新提案模型，明确使用 CPU Candle 训练
.\run.ps1 diagnostics     # CuModel GPU 推理/提案/事务，D0-D4 与 STEP 复核
.\run.ps1 diagnostics --adaptor replay --proposals configs/llm-proposal-example.json
```

两种执行路径的独立报告保留在 `output/diagnostics/runs/`。
持久会话已支持九项几何/事务操作和 inspect/fork/select/drop。
当前 CuModel 只学习六个探针增量参数；134 项几何能力库尚未全部接入会话执行器。
默认 demo 的前两次决策为记录程序，真实外部 LLM 尚未调用。

新增 **GPU harness 初版**：共享仿射模板、32 个候选、8 轮设备端操作/检查/提交与回退、
不可变目标、预算/版本控制、轨迹和逐位重放。运行 `./run.ps1 harness`，
交付 `output/harness/harness_hood.step` 和离线轨迹查看器；详见 [HARNESS.md](HARNESS.md)。
当前控制器为确定性规则，尚未接入 Candle 反馈策略；固定四曲面范围。

模型与 harness 的后续递归改进路线、独立晋级评测和首轮预算见
[SELF_IMPROVEMENT_PLAN.md](SELF_IMPROVEMENT_PLAN.md)。该文件是计划，尚未启动自动迭代。
最新责任边界与递归改进修订见 [SELF_IMPROVEMENT_REVIEW_V2.md](SELF_IMPROVEMENT_REVIEW_V2.md)；
`python scripts/audit_improvement_assumptions.py` 可复现当前模板的坐标和最优值诊断。

新增 **GPU 多曲面约束求解**：变量/约束图、锁定/重合/对称/附着、C0/C1/C2 接缝、
截面拟合和平顺目标，GPU 批量求解 8 个四曲面候选。入口为 `./run.ps1 constraints`，
交付 `output/constraints/constraint_hood.step`；协议、验证和限制见 [CONSTRAINTS.md](CONSTRAINTS.md)。
新增求解器独立于下面的 134 项库验收计数。

已实现并集中验收 **134 项有明确输入范围的建模能力**。本轮补齐 44 项：
曲线编辑、高级扫掠/放样、曲面匹配与平顺、实体特征、网格编辑、修复、
对象选择和历史重建；另加入吸附、merge、沿曲线/网格/曲面移动 5 项。
134/134 通过，15 组原语 STEP 加 1 个组合示例均经
OCCT 重导入校验。完整入口、后端和限制见 [CAPABILITIES_V4.md](CAPABILITIES_V4.md)。

```powershell
.\run.ps1 suite       # GPU 批处理 → 独立检查 → STEP 导出与重读
python scripts/composed_panel.py  # GPU 曲面 → 缝合加厚 → 打两孔 → STEP
.\run.ps1 test
.\run.ps1 lint
```

| 交付 | 位置 |
|---|---|
| 曲面原语及 CAD 组合 | `output/primitives/surface_primitives.step` |
| 曲线原语及投影/求交/连接结果 | `output/primitives/curve_primitives.step` |
| 三面 G2 过渡组合 | `output/primitives/g2_transition.step` |
| 13 项多边形结果（分面 B-rep） | `output/primitives/polygon_primitives.step` |
| 新增曲线编辑 / 高级曲面 | `curve_editing.step`、`advanced_surfaces.step`，位于 `output/primitives/` |
| 圆角 / 变圆角 / 倒角 / 加厚 / 抽壳 / 拔模 / 肋板 | `output/primitives/solid_features.step` |
| 新增网格编辑 / 修复结果（分面 B-rep） | `output/primitives/advanced_meshes.step` |
| 双片 C2 曲面组合、加厚并打两孔 | `output/primitives/composed_panel.step` |
| 独立验收与 STEP 重读结果 | `output/primitives/validation.json` |
| 参数化调用入口、限制及实际结果 | `primitive_catalog.json` |
| 全部 STEP / 原语检索 / 预览入口 | `output/primitives/index.html` |

可组合接口和容量协议见 [PRIMITIVE_API.md](PRIMITIVE_API.md)，验收范围见
[SURFACE_BASELINE.md](SURFACE_BASELINE.md)。几何数学与网格在 CUDA 上运行；
稳健 B-rep 求交、裁剪、缝合、实体特征和 STEP 使用 CPU CAD 内核。部分
新增网格拓扑编辑也有明确的 CPU 后端。GPU 几何批次只做一次最终读回，
无中间读回；不宣称每个算子已有 CUDA 实现。当前神经网络仍只驱动下面
的四个 MVP 指令槽；134 项是可组合的库能力，并非已训练的模型动作空间。

新增小模型驱动 GPU 建模 MVP：Rust Candle 训练 769 参数网络，CUDA 推理
直接向显存写入四槽建模指令；GPU 生成 NURBS 车身、镜像截面和旋转面轮胎，
最后一次回传后导出精确曲面 STEP。

```powershell
.\run.ps1 train
.\run.ps1 mvp
```

运行生成 `output/car/car.step`、`viewer.html`、控制网、权重快照和检查报告。
四个原语、显存通信协议和复现说明见 [GPU_POLICY_MVP.md](GPU_POLICY_MVP.md)；
实际验证和限制见 [VALIDATION.md](VALIDATION.md)。两套流程的 STEP 内核安装到本目录
`.python-deps`：`python -m pip install --target .python-deps cadquery-ocp-novtk==7.9.3.1.1`。

以下是最初的隐式建模例子，可独立运行：

Rust 读取可替换的 CUDA 几何代码，经 NVRTC 在运行时编译为 PTX，
再在 GPU 上采样隐式场、提取三角网格并渲染预览。
默认模型是半径 30 mm 的球，沿 Z 轴切去半径 8 mm 的贯通圆柱。

## 运行

在此目录运行：

```powershell
.\run.ps1 run --check
.\run.ps1 run --radius 24 --hole 6 --grid 96 --width 512 --out output/variant --check
```

已在 Windows、Rust 1.93、CUDA 12.6、RTX 2000 Ada Laptop 上运行。
需要 NVIDIA 驱动、CUDA Toolkit 的 NVRTC、Rust MSVC 工具链和 C++ Build Tools。
`CUDA_PATH` 指向 Toolkit 安装目录。脚本初始化 Visual Studio C++ 环境，
将 Cargo 缓存放在项目的 `.cargo-home` 中；首次构建需下载依赖。

`--grid` 是每轴单元数，范围 24–256，默认 128；`--width` 是正方形预览像素数，
范围 128–2048，默认 960。`--radius` 和 `--hole` 都是半径，单位 mm。

输出位于 `output/`：

| 文件 | 内容 |
|---|---|
| `viewer.html` | 内嵌网格，可离线拖动旋转和缩放 |
| `preview.png` | GPU 对隐式函数做 sphere tracing 的预览 |
| `model.stl` | GPU 提取的二进制 STL；按 mm 解释坐标 |
| `report.json` | 几何检查、误差、传输字节数及 CUDA event 时间 |
| `model.ptx` | 本次实际编译并执行的 PTX |
| `model-source.cu` | 本次执行的 CUDA 源码快照 |

STL 本身不编码单位。HTML 显示导出的网格，PNG 直接渲染隐式函数；
验证报告检查的是网格。离线 HTML 里的下载链接需要与 STL 保持在同一目录。

## LLM 生成代码的接入点

编辑 `cuda/model.cuh`，或用 `--model` 指定外部文件。模型函数返回负值表示内部，
零值表示表面；可以使用基础函数 `v/add/sub/mul/dot/cross/len/norm`。

```cpp
__device__ float model(V3 p, float radius, float hole_radius) {
    float sphere = len(p) - radius;
    float cylinder = sqrtf(p.x*p.x+p.y*p.y) - hole_radius;
    return fmaxf(sphere, -cylinder);
}
```

球减圆柱的 min/max 组合保持正确的符号和零集，但不一定处处都是真正的距离。
预览要求函数值是保守步长估计，任意标量函数需要另外控制步长。
随附的环面可以直接运行，无需重建 Rust 宿主：

```powershell
.\target\release\rust-cuda-modeling-demo.exe --model cuda/torus.cuh --grid 96 --width 512 --out output/torus
```

`--check` 的解析尺寸/体积预期只适用于球减圆柱，自定义模型省略该选项。
通用的非有限值、边界、网格连接和朝向检查仍然执行。
本 demo 没有接入在线 LLM API；模型文件是 LLM 输出代码的接口。

## 执行与传输

```text
Rust: 读取模型 → NVRTC 编译 → 分配显存与锁页主存
GPU: 场采样 → marching tetrahedra 提取 → 预览渲染 → 收集校验样本
GPU → CPU: 一次打包 DtoH
Rust: 独立几何检查 → STL / PNG / HTML / JSON 导出
```

同一 stream 排队所有 kernel，中间不等待 host。完整场常驻显存；只返回 64 个样本、
错误计数、网格和图像。使用一个 `u32` arena 和一次 `memcpy_dtoh`，最后统一等待，
再查询已经完成的 CUDA events。

为避免先读回三角形数量，arena 为 `32 * grid²` 个三角形预留容量，空闲槽位也会传回。
报告列出这部分字节；超容量返回错误，不会交付被截断的网格。
“一次”指一次应用层 DtoH API 调用，不代表底层只有一个 DMA 事务。
详细取舍及后续设计见 [TRANSFER_STRATEGIES.md](TRANSFER_STRATEGIES.md)。

每个立方单元一致地分成六个四面体，共享边采用统一插值顺序。
连接检查使用逐位相同的坐标，不依赖容差焊接。没有 CPU 建模回退路径。
CPU 负责调度、检查和导出。输出不含 CAD B-rep 或 SolidWorks 特征树。
检查不能证明任意模型不存在自交，也不代表制造公差验收。
薄壁/小孔需有足够网格分辨率；模型零集必须位于每轴 `[-1.2R, 1.2R]` 内部。

## 验证

```powershell
.\run.ps1 test
.\run.ps1 lint
.\run.ps1 run --check
```

单元测试覆盖独立四面体的体积/拓扑、缺面、翻面、非有限值和参数边界。
真实 GPU 检查覆盖场值有限、采样域边界、每条边恰好有两个相反方向面、
无退化面和正有向体积。

`--check` 另检查 Euler 特征为 0、体积与独立 napkin-ring 公式
`4π/3 * (R²-a²)^(3/2)` 的相对误差不超过 2%、包围范围误差不超过两个步长，
以及 64 个 GPU 场样本与 CPU f64 参考的一致性。
X/Y 范围应为 `2R`，Z 范围应为 `2sqrt(R²-a²)`。
实测见 [VALIDATION.md](VALIDATION.md)。

报告区分 NVRTC 编译、GPU events、CPU 验证与导出耗时。未宣称 CPU/GPU 加速比。
原始 SDF CLI 未实现跨编辑常驻状态；新增 `session` 已保持同一进程中的 GPU 场景，
`constraints/harness` 使用捕获执行图。跨会话恢复、通用 PTX 缓存与图形互操作尚未实现。

## GDS 与原始参考

NVIDIA GPUDirect Storage 不支持 Windows，所以本 demo 经主存导出文件。
大型点云/体素数据可在受支持的 Linux/GPU/存储配置中另外接入 cuFile。

- [NVIDIA NVRTC 12.6](https://docs.nvidia.com/cuda/archive/12.6.2/nvrtc/index.html)
- [cudarc 0.19.10](https://docs.rs/cudarc/0.19.10/cudarc/driver/index.html)
- [NVIDIA GDS 支持范围](https://developer.nvidia.com/gpudirect-storage/)
