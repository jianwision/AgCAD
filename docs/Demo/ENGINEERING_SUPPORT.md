# Class-A 工具、设计意图与拓扑容差支持

本扩展增加可复用的曲面构造/诊断、非线性设计意图和受预算约束的 B-rep 操作，并通过已有 `ModelAdaptor.decide` / `ModelSession` 暴露为类型化工具程序。

本文件记录工程 v1。后续已新增 [GPU 常驻曲面与多片联合优化](RESIDENT_SURFACES.md)，提供独立的 `resident-engineering-v2` 后端、NURBS 和显存内细化；v1 的一次性批处理行为继续保留。

验证范围是明确的数学和工程子集。G3-compatible 接缝、曲率和反射条纹工具为 Class-A 工作流提供支持；当前没有整车 Class-A 认证。组合 demo 为两片五次概念面板和一个多边形 UV 开孔，不是量产翼子板。

## 运行和交付

```powershell
.\run.ps1 test-engineering   # 构建 Rust/CUDA，执行 20 项独立验收
.\run.ps1 engineering        # 参数求解 → 曲面匹配/平顺 → 裁剪/缝合 → STEP
python scripts/engineering_preview.py
.\run.ps1 test
.\run.ps1 lint
python scripts/engineering_delivery.py --package
```

复用项目 Rust/CUDA、NumPy、SciPy 和 `.python-deps` 中的 OCP 环境。质量预览另需 Matplotlib。运行入口初始化本机 MSVC/CUDA 路径，不要求重新训练已有 CuModel。

| 产物 | 内容 |
|---|---|
| [class_a_panel.step](output/engineering/demo/class_a_panel.step) | 两片五次曲面的开孔缝合曲面，2 面、0 实体 |
| [validation.json](output/engineering/demo/validation.json) | 用户约束、接缝、质量和 STEP 回读证据及源文件摘要 |
| [质量预览](output/engineering/demo/quality-preview.png) | 实际 GPU 网格的反射条纹和主曲率图；左图已标明 Z 显示夸大 |
| [acceptance.json](output/engineering/acceptance.json) | 20 项测试逐项结果与实现摘要 |
| [native-validation.json](output/engineering/class-a/native-validation.json) | GPU 三阶导数与原生 OCP D3 的独立对照 |
| `output/engineering/*.step` | 带孔曲面、实体分割、闭壳成实体和求交曲线的 STEP 往返样例 |
| `output/engineering/demo/model-events.json` | 同一模型适配接口的逐次工具决策与反馈 |
| `output/engineering/demo/runtime-report.json` | 操作预算、原子提交和 GPU 批次计量 |

新源码/结果包为 `output/delivery/rust-cuda-engineering-support-v1.zip`，附逐文件 SHA-256 清单；不包含工具链、Python 依赖和构建缓存。既有 v6 和 model-sessions-v1 ZIP 保留。

## 1. 曲面构造与 Class-A 诊断

入口：[class_a.py](scripts/class_a.py)、[CUDA 内核](cuda/class_a.cu)、[Rust 批执行器](src/bin/class_a_suite.rs)。

- `BezierPatch(poles)`：两个方向分别为三至七次的非有理张量 Bézier 曲面，4–8 × 4–8 控制点，单位 mm。
- `jet(u,v,du,dv)`：位置和总阶数不超过三的混合导数，参数域 `[0,1]²`。
- `match_boundary(A,B,order=1|2|3,scale,shear,reverse)`：把 B 的 U=0 边界与 A 的 U=1 边界匹配，修改所需的前 `order+1` 行控制点，保留其余行。
- `seam_report`：独立采样边界位置/导数、法向夹角和横向法曲率差。
- `gpu_batch`：最多 32 片曲面；每片最多 65×65 采样。一个批次上传、一个内核、一次最终结果读回，无中间读回。
- GPU 输出：三阶 jets、法向、Jacobian、平均/高斯/主曲率、固定视线方向的反射向量和条纹值、采样规则性。
- 显式 GPU fairing：控制网二阶差分能量的投影梯度迭代，最多 512 步；支持锁定任意控制点和每点累计位移预算。该能量是参数化控制网能量，不能当作物理曲率积分。

匹配采用明确的局部参数映射：

```text
B(u,v) 的边界 jets = A(1 + scale*u, phi(v) + shear*u) 的边界 jets
phi(v) = v 或 1-v；scale > 0
```

三阶 jet 按链式法则匹配。在这个可逆仿射参数映射下，可构造相应 G1/G2/G3 接缝。它覆盖非单位比例、剪切和反向边界，未实现任意变化的重参数化、不同形状边界自动寻配或复杂多面角区的通用 G3 构造。

`assess_quality(report, criteria)` 支持采样规则性、最大绝对主曲率和最大控制网平顺能量要求。它始终标记采样证据，不返回全局 Class-A 认证。反射条纹采用固定视线和解析条纹场；不是完整透视环境光反射系统。

## 2. 约束与设计意图

入口：[design_intent.py](scripts/design_intent.py)。

`DesignIntent` 包含带单位的固定/可变参数、边界、公式依赖、点、静态参考曲线/曲面及具名约束。支持 mm/cm/m、rad/deg 和无量纲值；内部归一为 mm/rad。公式支持安全 AST 算术、整数次幂、sqrt、abs、sin/cos，检查量纲、未知引用和依赖环，不执行任意 Python。

| 约束 | 支持范围 |
|---|---|
| expression | 公式两侧等式、≤、≥，量纲匹配 |
| distance / coincident | 三维点间距离和重合 |
| angle / parallel / perpendicular | 非退化方向间角度、平行和垂直 |
| symmetry | 关于固定平面的点对称 |
| on_curve | 静态非有理 Bézier 曲线，求解自由 t |
| on_surface | 静态 Bézier 曲面，求解自由 U/V |

```python
from design_intent import DesignIntent
from engineering_fixtures import intent_fixture

intent = DesignIntent(intent_fixture())
solution = intent.solve()              # 半径 10 cm、角度 30 deg
# x≈86.60254 mm, y≈50 mm；diameter 和 rise 由公式更新
changed = intent.with_parameters({'radius': 12.})  # 调用方显式修改设计契约
new_solution = changed.solve()
```

模型只能提出可变参数初值，不能覆盖固定设计参数。公式和目标随用户契约序列化并计算摘要；调用方修改返回新契约实例。求解使用显式 CPU SciPy bounded least-squares，最多 64 参数、128 公式、128 约束、2000 次残差函数评估预算。预算包含有限差分调用，耗尽时返回已见最优候选并明确报告；最终独立约束核查另计一次。

求解器优先寻找硬约束的数值候选，再逐项按声明容差独立评估；报告失败约束 ID、局部 Jacobian 秩和剩余自由度。软约束提供测量，本版求解器不优化软约束。数值未达标不构成全局不可行证明；局部秩也不是全局自由度证明。

当前公式自动传播发生在参数图中。几何通过工具程序构建，并在交付时绑定到设计参数；本扩展还没有完整 CAD 特征图的自动下游重建。

## 3. 复杂拓扑与容差

入口：[tolerance_topology.py](scripts/tolerance_topology.py)。这些操作明确使用 CPU OCCT。

`TolerancePolicy` 由调用方固定：绝对线性容差、相对尺度系数、原生容差上限、双向采样形变上限、短边诊断阈值。有效线性容差为 `max(absolute_mm, relative*bbox_diagonal_mm)`，超过原生容差预算直接拒绝，不自动放宽策略。

新增受策略控制的操作：

- **trim**：曲面参数域中的简单多边形外边界和最多 8 个孔；每环 3–128 点。预检自交、塌边、相交/嵌套孔和越界，构建原生 pcurve/3D 边。
- **Boolean union/cut/common**：明确 fuzzy tolerance、复制输入、原生有效性和结果实体要求。
- **split / section**：实体/曲面分割及交线构造；分割/布尔记录本次输入面的 Modified/Generated/Deleted 历史。
- **sew**：要求指定容差下形成一个连通壳，或显式允许断开集合；可把闭合可定向壳构造成正体积实体。
- **heal**：在复制对象上执行 ShapeFix；遵守原生容差上限，并执行双向采样形变检查。
- **inspect**：原生有效性、自由边、每壳非流形边、周期接缝、退化边、短边、原生最大容差、面积/实体体积及尺度。
- **export_step_checked**：STEP 输出后重读，检查有效性、原生容差、每实体面数、实体数、面积/体积和双向几何采样。

容差控制有两种不同证据：原生拓扑容差上限直接核查，几何变形上限通过双向采样核查。后者不是连续域 Hausdorff 上界。`BRepCheck_Analyzer` 也不构成任意曲面全局无自交证明。

分割后的多个有效实体可以共享边/面；非流形检查按每个壳执行。STEP 可能复制跨实体共享拓扑，因此报告唯一面/边数量变化，同时核对每实体面数、体积和几何；不承诺跨格式保留原生对象身份。本次 Modified/Generated/Deleted 历史也不是跨重建的永久拓扑命名。

## 4. 接入 ModelAdaptor

[EngineeringRuntime](scripts/engineering_runtime.py) 与已有 [ModelSession](scripts/model_session.py) 配合使用。文档内包含具名、类型化的不可变结果：Bézier 曲面、B-rep、设计意图解和诊断报告。文档使用已有 session/slot/generation/revision 句柄；fork 保留独立文档版本。

```json
{
  "schema": "cad-decision-v1",
  "kind": "tools",
  "object_id": "来自 observation.object.id",
  "scene_revision": 0,
  "calls": [
    {"op": "intent.solve", "args": {}, "out": "dimensions"}
  ]
}
```

观察中给出工具的名称、参数、结果类型、执行后端、对象目录、用户契约和剩余预算。曲面引用通过 `left/right/patch`，B-rep 引用通过 `shape/a/b/shapes/tools` 指定。完整注册表为 `engineering_runtime.TOOLS`。

每个工具程序最多 32 调用、128 文档结果；程序写入原子提交，任一失败丢弃整段暂存结果并返回具体错误，已预留的预算不返还。模型可根据反馈重新构造；质量变差不会自动触发拒绝。最多 1024 个显式后端请求，动作预算在文档分支间共享。

交付规则由调用方固定，可声明设计意图达标、曲面坐标与设计参数绑定、接缝阶次/jet/法向/法曲率容差、采样规则性和曲率阈值。导出时重新计算，并确认被检查的曲面控制网确实是导出 B-rep 的支撑面；不能用无关场景对象或缓存诊断代替交付检查。

本版工具程序没有持久私有草稿，使用程序级原子提交。原 `cuda-hood-v1` 后端继续保留其驻留 GPU 草稿和 CuModel rollout。新工程后端在 CPU 保存对象，按需调用 GPU 批次并读回结果；尚未把新工具变成通用常驻 GPU DAG。当前 CuModel 未训练这些新增工具，demo 使用明确标记的确定性反馈回调，不宣称真实外部 LLM 已推理。

## 5. 验收证据

20 项新增测试覆盖原生 OCP 三阶导数对照、解析曲率、反射向量、锁点/位移预算/能量降低、G1/G2/G3 参数映射和反向、奇异采样、单位/公式/冲突/自由 UV、包含有限差分的求解预算、拓扑组合/容差及工具程序保护。另回归 12 项旧会话测试、6 项适配器测试和 15 项 Rust 测试；Clippy 与 Python 编译检查通过。

实测 GPU jets 与 OCP D3 最大差异约 **8.56e-12 mm**；四曲面质量批次 **1 次最终读回、0 次中间读回**。组合 demo G3 jet 最大残差约 **7.91e-13 mm**，STEP 双向采样误差约 **8.25e-12 mm**。这些小误差说明限定计算/转换的一致性，不能代替工业造型质量和制造可行性验收。

库中的 134 项旧能力计数保持原定义；本扩展按服务和 20 项新增测试单独记录，不把包装同一个底层操作重复算成新建模能力。没有宣称任意拓扑修复、任意 G3、整车 Class-A 或相对 CATIA 的性能优势。
