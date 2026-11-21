# 小模型在 GPU 上驱动 GPU 建模

运行链路：

```text
Rust/Candle 训练 → 导出 769 参数权重
                         ↓ 初始化时上传一次
GPU: 5→24(tanh)→25 策略网络
  → 四槽指令缓冲 → 参数/操作码验证
  → NURBS_PATCH → MIRROR_Y → REVOLVE_Y/NOOP → TESSELLATE
  → 检查结果、控制网、网格打包 → 最后一次 DtoH
CPU: 独立验证 → 精确 NURBS + B-rep 拓扑 → STEP → 重读验证
```

四个建模原语加一个 NOOP；其余 Bezier、Coons、导数等是数学辅助函数，
不是额外的模型指令。

| 操作码 | 原语 | 当前行为 |
|---|---|---|
| 1 | NURBS_PATCH | 生成 U 向夹持、V 向闭合的三次车身控制网 |
| 2 | MIRROR_Y | 镜像半截面控制点，组成对称封闭截面 |
| 3 | REVOLVE_Y | 旋转圆形轮廓，生成四个环面轮胎 |
| 4 | TESSELLATE | 对曲面采样、构造封口和定长三角形输出 |
| 0 | NOOP | 模型根据输入关闭轮胎生成 |

输入：归一化长度、宽度、高度、溜背程度、是否带轮胎。输出：四组五类
操作码 logits，加五个连续参数。网络只有 769 个参数，FP64 导出权重为
6,152 字节。训练采用有明确规则的合成配方，四个指令槽中，车身、镜像和
细分槽固定；轮胎槽在 REVOLVE_Y/NOOP 之间选择，连续参数也由模型预测。
这证明 GPU 输出可以直接影响后续几何，并不代表已经学会开放式 CAD 规划。

## 复现

```powershell
.\run.ps1 train       # Rust Candle，明确使用 CPU 训练
.\run.ps1 train-gpu   # 可选：明确使用 CUDA 训练（本机也已验证）
.\run.ps1 mvp         # CUDA 推理、曲面生成、STEP 导出及重读检查
.\run.ps1 car --no-wheels --length 5100 --width 2100 --height 1200 --style 0.9 --out output/car-variant
python scripts/export_step.py
```

CUDA 训练单独通过 `.\run.ps1 train-gpu` 请求；请求失败会报错，不会悄悄
切换 CPU。实际测试状态记录在 VALIDATION.md 和 models/training_report.json。
首次训练构建需下载 Candle；GPU 训练还要编译 Candle 的 CUDA kernels。

STEP 导出依赖 `cadquery-ocp-novtk==7.9.3.1.1`，安装到 `.python-deps`。
导出器用 GPU 控制点构造同一张有理 B 样条曲面；1,200 个 CUDA 样本与
独立 CAD 求值比较，再检查实体有效性和 STEP 重读后的数量、体积。
STL 只作预览/网格用途。

## 中间不回 CPU 的含义

Rust 预先把固定的 kernel 序列排入同一 CUDA stream；推理输出从不在
CPU 上解释。后续 kernel 从 GPU 指令缓冲读取操作码与参数，错误则跳过
几何写入。GPU 内没有动态分配或任意代码执行，容量和拓扑上限预先确定。
此 MVP 没有实现设备动态启动任意 kernel、CUDA Device Graph Launch，
也没有常驻 token 解码循环。固定任务先验证数据依赖即可。

权重和输入在开头上传；模型推理到建模结束，中间 DtoH 调用为零，最终
打包回传调用为一次。CUDA 事件只在最终读回后查询；不声称一个 API 调用
等于一次物理 DMA。首次 NVRTC/PTX 编译和驱动 JIT 可能远慢于实际建模。

## 范围和下一步

车身为封闭轮罩概念外形，轮胎是独立环面体，可能与车身相交。封口为平面，
与车身连接允许尖边；仅闭合周期接缝检查 C0/C1/C2。没有车窗分件、轮拱
裁剪或全车 Class-A 认证。模型预测尺寸带有学习误差，输入是目标比例，
实际尺寸以 report.json 为准。

原语库现已扩展为 134 项有界汽车曲面/实体/多边形能力，含 GPU 截面放样、Coons、
G1/G2 连接带及 CPU CAD 裁剪/求交/缝合，见 PRIMITIVE_API.md。此四槽模型
保持原有输出协议，尚未学习完整 134 项库操作。模型侧后续需要扩大训练
覆盖、加入类型句柄、合法动作掩码和几何误差反馈；自由指令序列可再使用
小型 Transformer。全车 Class-A、多 patch 全局约束仍属于后续工作。

参考：[Candle 官方训练示例](https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training/main.rs)、
[NVIDIA 设备图调度说明](https://developer.nvidia.com/blog/enabling-dynamic-control-flow-in-cuda-graphs-with-device-graph-launch/)、
[Open CASCADE STEP 支持](https://occt3d.com/dev/doc/overview/html/occt_user_guides__step.html)。
