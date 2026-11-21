# 可组合原语接口 v4

这是一套实际运行过的汽车曲面/多边形库，共 134 个验收项。逐项入口、
执行后端、输入限制保存在 `primitive_definitions.json` 和 `primitive_extensions.json`、
`primitive_operations.json`、`primitive_topology.json`、`primitive_advanced.json`、`primitive_motion.json`；`primitive_catalog.json`
在验收时加入实际结果。容量有界，调用者必须遵守输入前提；数学辅助函数
不承担通用 CAD 输入修复。它不是完整的生产 CAD 内核。

## 源码与执行边界

| 文件 | 内容 |
|---|---|
| `cuda/surface_primitives.cuh` | 点/向量、Bezier、NURBS 二阶导数、法向与偏移求值 |
| `cuda/curve_ops.cuh` | 曲线、参数裁剪、任意轴变换、曲率与连续性度量 |
| `cuda/surface_records.cuh` | 曲面记录布局、控制点写入 |
| `cuda/surface_builders.cuh` | 参数化曲面控制网、放样、Coons、G1/G2 连接带 |
| `cuda/surface_suite.cu` | 曲面/曲线配方、记录求值及验收 kernels |
| `cuda/polygon_ops.cuh` | 参数化基础网格、三角化、挤出、内插、细分、焊点、法线、截角及批处理 kernels |
| `src/bin/primitive_suite.rs` | NVRTC、显存 arena、kernel 调度、一次打包读回与独立网格检查 |
| `scripts/cad_ops.py` | 可复用的 CPU 原生 CAD 投影、求交、裁剪、分割、连接、填充、扫掠、放样、缝合 |
| `scripts/validate_all.py` | 独立 OCCT/解析参考、组合验证、四组 STEP 重读 |

NVRTC 按表中前六个 CUDA 文件的顺序拼接。曲面构造器可以单独使用前四个
文件；直接调用 `nurbs_surface` 求值，偏移使用 `offset_point`。原语间传递
显存里的记录或 `GMesh`，不需把控制网回传到 CPU。依赖分阶段 kernels
通过同一 stream 排序；单个 primitive 的局部循环目前串行，不宣称吞吐加速。

当前模型头只有 `NURBS_PATCH / MIRROR_Y / REVOLVE_Y / TESSELLATE` 四槽。
批处理测试用显式配方调用扩展库；当前模型尚未学习任意 134 项动作的规划。

## 扩展入口与组合

新增完整矩阵见 [CAPABILITIES_V4.md](CAPABILITIES_V4.md)，每项均含后端、
输入限制和验收证据。原始 46 项协议保留在本文后续部分。

| 源码 | 可复用能力 |
|---|---|
| `cuda/extended_curves.cuh` | 圆锥曲线、螺旋、Hermite、插值、长度/重采样/最近点 |
| `cuda/extended_solids.cuh` | 10 类基本实体定义和网格 |
| `cuda/modeling_operations.cuh` | bend/twist/taper、RMF sweep、非均匀 loft、Catmull-Clark/Loop、重采样 |
| `cuda/topology_operations.cuh` | revolve、删三角面、平面孔填充、Laplacian/Taubin |
| `cuda/advanced_operations.cuh` | 导轨/比例/变截面/闭环 sweep、Bezier 边界 G1/G2、联合平顺 |
| `scripts/cad_curve_extensions.py`、`cad_solids.py` | GPU 定义到原生 CAD 曲线/实体 |
| `scripts/cad_topology_operations.py` | 原生 Boolean/delete/cap/revolve |
| `scripts/cad_advanced.py` | 曲线编辑/拟合、兼容化、曲面编辑、实体特征、容差/小特征修复 |
| `scripts/mesh_advanced.py` | 网格减面/自适应细化/翻边/桥接/区域挤出/折痕细分/修复 |
| `scripts/feature_graph.py` | 类型句柄、几何选择、DAG、undo/redo、JSON 保存与重建 |
| `cuda/motion_operations.cuh`、`scripts/placement_ops.py` | 吸附、网格合并、按弧长移动、跨面走步、UV 贴附和实例放置 |

GPU 曲面输出仍使用下述 4192-double 记录。批次总 arena 由 Rust 计算，
新增算子继续在同一 stream 中传递显存记录，只在结束时下载一次。局部
拓扑循环有界且通常串行，尚无相对 CPU 的吞吐加速声明。

例如原生特征 DAG 可以组合现有 CPU 后端，注册函数返回的新实体可以继续
参与后续 Boolean、圆角或 STEP 输出：

```python
# 从 scripts/ 运行，或先将 scripts 加入 sys.path
from feature_graph import FeatureGraph
from cad_ops import valid
from cad_topology_operations import boolean_union
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
registry = {
    'box': ((), 'solid', lambda x,y,z,ox=0:
        valid(BRepPrimAPI_MakeBox(gp_Pnt(ox,0,0),gp_Pnt(ox+x,y,z)).Shape())),
    'union': (('solid','solid'), 'solid', boolean_union),
}
g = FeatureGraph(registry)
a = g.add('box', x=2,y=2,z=2)
b = g.add('box', x=2,y=2,z=2,ox=1)
body = g.add('union', [a,b])
shape = g.evaluate(body)
selection = g.select(a, 'face', (0,1,1))
g.update(a, z=3)       # 只失效 a 及其后代；b 的缓存保留
g.undo(); g.redo()
saved = g.dumps()
fresh = FeatureGraph.loads(saved, registry)
rebuilt = fresh.rebuild(body)
```

对象 id 跨参数编辑、历史和保存重建保持稳定。子形体选择属于特定结果
revision；修改或重建后旧选择会报错，需重新查询。查询多解也会报错。
此协议不承诺任意拓扑修改后仍自动保持原来的边/面命名。

完整的 GPU 曲面 → 缝合 → 加厚 → Boolean 打孔 → STEP 示例在
`scripts/composed_panel.py`。此例复用 GPU 输出的双片 C2 控制网，
不从 STL 拟合曲面；最终网格预览来自 CPU CAD 三角化。

## 曲面记录与组合

每个记录 `SURF_WORDS=4192` 个 double；当前批处理容量 14 个记录。

| 槽位 | 含义 |
|---|---|
| 0..3 | U/V 控制点数与次数 |
| 4..7 | U/V 有效参数域 |
| 8..11 | 类型（0 NURBS、1 offset）、base id、offset distance、配方状态 |
| 16..55 | 完整 U knot vector，包括重复节点 |
| 56..95 | 完整 V knot vector，包括重复节点 |
| 96..4191 | U 行优先的控制点 `[x,y,z,w]`，XYZ 未乘权重 |

每轴控制点最多 32，次数 1..5；每个参数向量有效长度为 `count+degree+1`。
权重必须有限且正，节点合法、非递减，有效域非空。只在正则点解释法向和
曲率；Jacobian 为 0 时曲率无效。`make_nurbs_patch` 检查大小、有限值、
节点单调性和权重；合法节点重数及其他构造器的曲线兼容性由调用者保证。
bool 构造器失败后应停止使用该记录；调用者写状态，后续 kernel 跳过失败项。

例如在同一 kernel 中创建曲面与偏移定义，后续 kernel 再采样：

```cpp
// arena is zero-initialized; poles contains 16 finite P3 control points.
double* base = surf_record(arena, 0);
if (!make_bezier_patch(base, poles, 4, 4)) { base[11] = 1; return; }
double* off = surf_record(arena, 1);
if (!make_offset(off, 0, 0.07)) { off[11] = 1; return; }
// Later, without a host readback:
Jet jet = nurbs_surface(base + 96, 4, 4, 3, 3, base + 16, base + 56, u, v);
P3 point = offset_point(jet, off[10]);
```

旋转面使用精确有理圆弧、绕 Y 轴；不是把等角采样点当控制点。GPU 扫掠
目前为精确平移曲面 `A(u)+B(v)`。GPU 放样对 2..11 条兼容非有理截面做
自然三次插值；G1/G2 连接带要求调用者提供边界与横向一/二阶导数。
一般路径扫掠、多边界约束填充及稳健拓扑使用 `cad_ops.py`。一般曲线的
v4 增加了 CPU 曲线仿射重参数化、GPU 双片约束平顺；任意 trimmed-NURBS
相交仍使用 CPU CAD 内核。双片平顺不等于全局 Class-A 曲面优化。

## 多边形记录与组合

`GMesh` 引用一个清零记录，`MV=MT=4096`，`MESH_WORDS=36872` doubles。
槽 0/1/2 为顶点数、三角形数、状态；8 开始放位置，`8+3*MV` 放顶点法线，
`8+6*MV` 放三角形索引。计数/索引用 double 存储但必须为精确整数。
失败记录不可继续消费；初始网格与输出记录必须满足各算子的容量前提。

典型依赖：`make_box` → `subdivide_triangles` → `weld_vertices` →
`mesh_normals`。需要独立输出的函数用清零且不重叠的目标记录。`weld_vertices`
采用首个代表点、半径容差，可能使过大的焊点容差产生退化面；调用者应重新
做拓扑检查。`subdivide_triangles` 是 1→4 线性细分，不是平滑细分。

三角化允许平面凹多边形，最多 64 个点、无孔洞；内插限凸多边形，按顶点
重心比例缩小；顶点截角限闭合凸三价多面体。网格构造器主要是独立对象
构造，不能直接向已有网格追加而假定索引自动重定位。几何容差是绝对值，
默认以 mm 解释，极小/极大尺度需要调用者归一化和重新验收。

旋转、平移、非均匀缩放、镜像可逐控制点/顶点调用；对完整网格做反射时
须修正绕序并重算法线。`mesh_affine` 是附带的整网格辅助函数，尚未作为
独立验收项，46 项变换验收针对点变换及独立 CAD 参考。

## 复现与交付

```powershell
.\run.ps1 suite       # 固定验收集，输出 output/primitives
.\run.ps1 primitives --out output/another  # 只生成另一目录的 GPU 数据
```

`suite` 使用固定输出目录，逐项验证，不因部分失败提前跳过后续项，最终失败
返回非零退出码。CUDA 接口及 CLI 的致命错误会立即失败。`validation.json`
是本次证据，不能用旧的通过报告替代新修改的重新验收。

GPU 批处理构造/求值后只执行一次 DtoH API 调用，5,225,232 字节，零中间
DtoH；CPU 随后进行独立验证及 CAD 拓扑/文件导出。生成的 STEP 单位 mm，
AP214，曲面保留解析/有理表示；多边形 STEP 是实际三角面构成的分面 B-rep，
不会恢复成光顺曲面。它可供 CAD 后续导入，不包含原生特征历史。
STEP 已用 OCCT 重读验证；未在本机 SolidWorks/CATIA 中验证导入。

## GPU 约束图扩展

`ConstraintGraph` 的变量/硬等式/软目标协议及可组合构造器见 [CONSTRAINTS.md](CONSTRAINTS.md)。
运行 `./run.ps1 constraints`；GPU 结果、独立参考和 STEP 位于 `output/constraints/`。
此扩展与原有 134 个建模验收项单独计数。

## GPU harness 初版

共享模板、操作事务、版本/预算、独立检查和设备轨迹接口见 [HARNESS.md](HARNESS.md)。
运行 `./run.ps1 harness`；初版固定四曲面，AUTO/SET/EDIT/NOOP/STOP，
未把 134 个库能力全部接入 GPU 事务，也尚未训练新的反馈策略。
