# GPU 建模 harness 初版

`./run.ps1 harness` 运行共享模板编译、32 候选设备端事务、独立检查、故障注入和 STEP 交付。
结果在 `output/harness/`；`viewer.html` 可离线查看最终控制网和每轮轨迹。
此流程与 134 项原语库、四槽 Candle 小模型及 PCG 约束求解器单独验收。

## 执行路径

```text
CPU: 约束图 → SVD 硬约束消元 → 仿射最优解映射 → 一次打包上传
GPU: 初始化/检查 → [提出操作 → 构造试验 → 独立检查 → 决定 → 提交 → 记录] × 8
GPU: 选择有效结果 → 显存快照 → 完整重放 → 逐位比较 → 一次最终读回
CPU: 全坐标 KKT 独立参考 → 原生 NURBS → 缝合 → STEP 导出/重读/距离检查
```

控制器在 GPU 中根据当前参数、不可变目标、有效性和预算生成 SET 或 STOP。
它是确定性控制器；已有 Candle 网络尚未接入此反馈接口，也没有训练新策略。
接入模型时，应替换 `harness_propose` 的 AUTO 分支，模型只产生候选操作，
继续由独立检查和提交路径决定是否采用。

## 初版问题及缓存

示例为 3000 × 1500 mm 的四片双三次发动机盖曲面：192 个坐标变量，400 条硬等式，
484 条软目标，硬约束秩 186，实际剩余 6 自由度。保持 XY、外边界和兼容 C2 接缝，
调整截面拟合与平顺。所有候选都按用户固定的 270 mm 冠高拟合目标评分；
它不是曲面最终实测最高点，也不能被操作指令改写。

算子固定时，目标关于归一化冠高 `t=(theta-240)/50` 为仿射：

```text
x = x0 + Z y                 # 硬等式的可行子空间
d(theta) = d0 + t d1
ymap = lstsq(B, [d0 d1])     # CPU QR/SVD，未显式求逆
y(theta) = ymap[:,0] + t ymap[:,1]
```

因此不需要循环 PCG 或学习迭代来解这个固定线性问题。设备重构控制网后仍独立计算
原始约束残差、固定用户目标分数、真实降低梯度及局部曲面检查，不以映射自报成功。
改变模板、系数、权重或硬约束需要 CPU 重新编译；版本不匹配返回 NEED_RECOMPILE。
本例硬约束右端不随 theta 变化，不支持参数化硬等式右端或任意新图直接套用该映射。

一个 1,435,136 字节模板供 32 个候选共享；另有请求与操作记录，总上传 1,444,096 字节。
与每候选复制完整模板相比，避免 44,485,248 字节模板副本，此数是布局计算值。
`template.npz` 使用图、编译器源码及 NumPy 版本构成缓存键，并校验文件 SHA-256；
验收实际再次载入并比较全部数组。每次启动仍上传一次；尚无跨进程显存缓存服务。

## 操作与事务

| 操作码 | 行为 |
|---|---|
| 0 AUTO | 设备规则提出固定目标 SET；已达到目标且解有效时 STOP |
| 1 SET_PARAMETER | 在 [200,320] mm 内重构缓存的精确仿射最优解 |
| 2 EDIT_CONTROL_DELTA | 修改试验控制网一个坐标；用于验证独立检查，也受全部提交条件约束 |
| 3 NOOP | 复用已经检查的当前结果，记录 CACHE_HIT |
| 4 STOP | 验证版本后停止该候选 |

当前与试验控制网各有独立缓冲区。提交要求：参数/版本有效、硬约束比值 ≤1、
每片 5×5 采样的 XY 投影 Jacobian 大于阈值、SET 的相对降低梯度 ≤1e-9，
且按不可变目标评分严格改善。改善阈值为 `1e-9 + 1e-12*max(abs(current_score),1)`。
失败保留当前控制点、自由度、参数、修订、目标分数及有效性；预算与拒绝计数会更新。

同参数命中要求当前具有求解有效位；默认缓存参数容差为 0。NOOP 只复用当前已检查状态。
每个活动的非 STOP 操作消耗一个预算；非法参数和过期操作也消耗预算，防止无限重试。
STOP 不消耗预算，但必须满足修订、模板、系数及目标版本。固定 8 轮捕获图中，
已停止/预算耗尽的候选由设备掩码跳过。初始结果无效的候选不可参与最终选择。
选择按固定分数升序、候选编号作确定性平局处理。

错误码按顺序为 ACCEPTED、CACHE_HIT、STOPPED、INACTIVE、BUDGET_EXHAUSTED、
STALE_REVISION、NEED_RECOMPILE、STALE_GOAL、INVALID_OPCODE、INVALID_PARAMETER、
NONFINITE_PARAMETER、HARD_CONSTRAINT_FAILED、GEOMETRY_CHECK_FAILED、
NUMERICAL_CHECK_FAILED、NO_IMPROVEMENT、INVALID_BASELINE（0–15）。

局部 Jacobian 检查适合本例固定 XY 图形曲面，不能证明全局无自交、碰撞、
曲率等级、制造可行性或任意方向曲面正则性。C2 验证限兼容多项式参数化，
未实现通用非线性 G1/G2、尺寸/接触约束或拓扑变化事务。

## 二进制 ABI 与容量

全部为 little-endian；浮点为 IEEE754 f64。Rust 在 CUDA 启动前检查尺寸、完整字节长度、
共享数据有限性、容差、掩码、权重、目标范围与预算。命令的非法值由设备检查。

| 记录 | 大小/顺序 |
|---|---|
| Header | 128 B：`GPHARN01`；12 个 u32；9 个 f64 |
| Header.u32 | version=1,n,m,k,r,candidates,rounds,template_rev,coefficient_rev,goal_rev,trace_words=24,reserved |
| Header.f64 | lower,upper,gradient_tol,improvement,score_slack,cache_epsilon,jac_min,center,scale |
| 共享数组 | Z[n,k],x0[n],ymap[k,2],A[r,n],bmap[r,2],tol[r],hard[r],weight[r],B[m,k],dmap[m,2]；row-major |
| Request | 24 B：goal:f64,initial:f64,budget:u32,goal_rev:u32 |
| Command | 32 B：opcode,expected_revision,template_rev,coefficient_rev,goal_rev,index 六个 u32；value:f64 |

请求按候选排列，命令按轮次优先排列。`weight` 是软目标 `weight/tolerance²`，硬行设 0。
容量：n 固定 192，k 为 1–192，m 为 1–8192，m≤r≤8192，候选 1–64，轮次 1–32，
每候选预算 0–256；输入不超过 256 MiB。支持此 ABI 不意味着任意模板已通过几何验收。

输出每候选 `2*n+2*k+32+rounds*24` 个 f64，顺序为 current_x、trial_x、current_y、trial_y、
32 槽状态、每轮 24 槽轨迹。状态与轨迹字段的权威顺序在 `scripts/harness_demo.py`
的 `STATE_NAMES`、`TRACE_NAMES`；CUDA 对应枚举。末尾 8 槽为选中编号、分数、
是否存在有效结果、候选数、重放差异 word 数、三个保留槽。计数与枚举在 f64 中精确表示。

所有设备操作在一条独立 stream 上。第一次图输出在显存复制为快照，重置后第二次执行
同一图，GPU 逐位比较全部输出，最后只调用一次应用层 DtoH（主案例 158,784 字节）。
图内无主机读回。HtoD/DtoH 次数指应用 API，不是物理 DMA 次数。
逐位一致结论限同设备、同输入、同次运行的相同二进制，不保证跨驱动或跨 GPU 一致。

## 验收与交付

32 个主候选 × 8 轮覆盖提交、复用、停止、预算、过期版本、非法参数、硬约束回退和无改善。
另有三组双候选故障注入：局部翻折、错误的仿射映射、无效初始曲面；对照候选停止在
相同初始状态，直接逐位比较回退后的全部当前控制点和有效性指标。
几何注入刻意放松受影响的硬行以到达几何拒绝路径，不是新的合格车身方案。
还有十个畸形输入在 Rust 主机解析阶段被拒绝。

主案例每个最终状态对照独立全坐标 KKT，每个实际求值试验的分数独立重算；
检查目标不可改写、评分单调、预算、修订和重放。选出的 GPU 控制网直接构造 4 片
原生 NURBS，缝合并导出 `harness_hood.step`，不是从 STL 拟合。
OCCT 重导入验证 4 面、0 实体及面积；196 个距离探针和 C0/C1/C2 接缝检查通过。
STEP 可作为后续 CAD 交换输入，无原生特征历史；本机未做 SolidWorks/CATIA 导入验收。

源码入口：`src/bin/gpu_harness.rs`、`cuda/gpu_harness.cu`、`scripts/harness_demo.py`。
`gpu_report.json`、`validation.json`、`trace.json`、`program.json`、`input.bin`、`result.bin`、
实际 CUDA 源码/PTX 快照和 STEP 构成本次可复核结果。执行时间是单次 GPU event 记录，
CPU 编译、NVRTC 和 STEP 导出单列，无 CPU/GPU 加速比结论。
