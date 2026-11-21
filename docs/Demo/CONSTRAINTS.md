# GPU 多曲面约束求解 v1

在已有 134 项有界建模能力之外，增加固定拓扑的仿射约束图和批量 GPU 求解器。
四片双三次 Bezier 曲面在固定外边界、镜像对称、C0/C1/C2 接缝条件下，联合拟合截面目标并平顺。
一次处理 8 个候选，输出其中方案 6 的四面缝合曲面 STEP。长度单位为 mm。

```powershell
.\run.ps1 constraints
python scripts/constraint_delivery.py
# 可选：生成含完整源码、原有示例和新增结果的 v5 包
python scripts/constraint_delivery.py --package
```

## 可组合接口

接口位于 `scripts/constraint_graph.py`；所有约束最终降低为具名仿射行。
JSON 协议保存点变量、稀疏行系数、目标、硬/软标记、公差和权重，可重读编译。

| 接口 | 实际含义 |
|---|---|
| `point` / `row` | 稳定名称的 XYZ 变量 / 通用线性等式 |
| `lock` | 锁定点的指定坐标分量，或将其设为软目标 |
| `link` | 标量关系 a = scale*b + offset |
| `coincident` | 两个点 XYZ 重合 |
| `on_plane` | 点在固定平面上，法向自动归一化 |
| `symmetry` | 两点相对于指定固定平面对称 |
| `on_line` | 点在固定直线上，沿直线的位置仍可自由求解 |
| `on_curve` | 点附着在可编辑三次 Bezier 曲线的固定 t 上 |
| `on_patch` | 点附着在可编辑双三次 Bezier 曲面的固定 UV 上 |
| `patch_target` | 曲面固定 UV 处的指定 XYZ 分量拟合目标 |
| `seam` | 同向、等参数跨度的相邻双三次曲面 C0/C1/C2 连接 |
| `fair_patch` | 控制网二阶差分平方目标；不是物理曲率能量积分 |
| `compile` | 硬约束一致性、秩、冗余、剩余/未锚定自由度分析 |
| `write_batch` | 按相同维度分桶，生成 Rust/CUDA 输入协议 |

```python
from constraint_graph import ConstraintGraph
g = ConstraintGraph()
a = g.point('a', (0, 0, 0))
b = g.point('b', (1, 1, 1))
g.lock('anchor', a)
g.on_line('guide', b, origin=(0, 0, 0), direction=(1, 0, 0))
g.lock('desired_position', b, (20, 0, 0), hard=False, tolerance=.1)
compiled = g.compile()
```

## 执行边界

1. CPU 编译具名约束。对硬等式 E*x=f 做 SVD，得到 x=x0+Z*y；报告冗余、欠约束及疑似冲突行。冲突报告不是最小冲突集证明。
2. 软等式按 sqrt(weight)/tolerance 归一化。约束权重只影响软目标；硬约束不会被大惩罚权重替代。
3. 上传 B=A_weighted*Z 和 d=b_weighted-A_weighted*x0、重建算子及原始残差行。当前降低结果是稠密矩阵；并未宣称稀疏大规模求解器。
4. Rust 捕获一个 CUDA Graph。GPU 使用对角预条件 CG 求解 BᵀB*y=Bᵀd，分别并行执行行乘、转置乘、向量更新及归约，不显式存储 BᵀB。
5. 收敛、数值失败和迭代容量耗尽由 GPU 标记。已经收敛的候选跳过后续迭代工作，但执行图仍含固定数量的节点；这不是设备端新建执行图。
6. GPU 重建全部控制点，计算每条原始残差和最终真实梯度，并一次读回。CPU 检查硬约束及独立参考后，使用 OCCT 缝合、导出并重读 STEP。

每个批次：1 次输入上传、1 次最终下载、0 次中间读回、1 次图启动。
不同维度的验收桶各自独立运行，不能把整条含多个验收桶的命令说成总共只有一次下载。
容量：每桶 1..64 个候选，1..1024 个标量变量、最多 8192 行、0..512 剩余自由度；打包输入合计不超过 256 MiB。
全锁定、无硬约束、零初始残差分别有实际验收。默认迭代容量为 256。

PCG 使用正规方程，会放大条件数；适用范围是数值可解的有界仿射问题。未收敛结果被验收器拒绝。
CPU SVD、编译、NVRTC、图创建及 STEP 的开销都不包含在 `gpu_graph_ms` 内。尚无匹配 CPU 性能基线，不作加速结论。

## 验收

`output/constraints/validation.json` 包含 8 个四曲面候选、4 个独立随机系统、
1 个附着/滑动系统，以及全锁定和零残差系统，共 15 个成功 GPU 案例。
另有 10 项非法输入/冲突/欠约束检查及 4 个迭代容量耗尽案例。

- 全坐标 KKT 最小二乘参考独立于零空间 PCG；逐项检查硬约束及 GPU 原始残差。
- 检查 JVP 有限差分、VJP、伴随恒等式、候选参数线性响应和解析附着位置。
- 四条曲面接缝各检查 101 个位置及一/二阶导数；572 个曲面点与原生 CAD 求值比较。
- STEP 重导入检查有效性、4 面数量、面积及 196 个原曲面点到重读 B-rep 的距离。交付为曲面壳，非封闭实体；未在真实 SolidWorks/CATIA 中导入验证。

## 后续能力

本轮实现 C1/C2 作为兼容参数化下的连接条件；正则处可得到相应几何连续性。
一般重参数化 G1/G2、可变 t/UV 滑动、欧氏距离/角度等非线性约束、硬不等式、
最小厚度/间隙、全局碰撞和自交、优先级层次求解及稀疏大规模后端均尚未实现。
现有 769 参数小模型的四槽动作空间保持原有范围，新增约束图尚未训练进策略。
