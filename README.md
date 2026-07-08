# RL-Matting-Agent
![Rust](https://img.shields.io/badge/Rust-Edition__2024-black?style=flat-square&logo=rust&logoColor=white)
![Tauri](https://img.shields.io/badge/Tauri-v2-24C8D8?style=flat-square&logo=tauri&logoColor=white)
![React](https://img.shields.io/badge/React-18%2B-61DAFB?style=flat-square&logo=react&logoColor=black)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.0%2B-38BDF8?style=flat-square&logo=tailwind-css&logoColor=white)
![License: MPL 2.0](https://img.shields.io/badge/License-GPL__3.0-brightgreen)


<img width="2048" height="1280" alt="UI-1演示" src="https://github.com/user-attachments/assets/81dd8e33-48ab-47c4-9260-f81c0ba536c4" />

https://github.com/user-attachments/assets/fedb6c5e-db2e-41e3-b4cb-9c794858ef6b

TauriApp(Rust+React).An image subject extraction tool that integrates traditional image processing methods, deep learning models (custom-trained), and semantic segmentation models (existing U2-Net), while supporting customizable processing workflows and parameters.

一個整合了傳統影像處理方法（Rust標準圖形處理庫）、深度學習模型（自己訓練的）、語意分割模型（現有U2模型），且支援自訂處理工序與參數的影像主體擷取工具。

⚠️Vibe Coding产物，整个项目只有UI和ml_engine.rs是自己写的，模型是自己训练的，其余均由Trae完成。这是我的课设来着，仅供娱乐，切勿用于商业用途，因为实在没什么商业价值（）。

# Deployment Guide

    Clone the repository:
    git clone https://github.com/micromimo/RL-Matting-Agent.git

    Prepare the assets:
        Place the ONNX format models into /src-tauri/target/debug/up
        Place the U2Net series models into /src-tauri/target/debug/models

    Run the development server:
      Navigate to the project root directory and run the following commands in your terminal:
        cd RL-Matting-Agent
        npm install
        npm run tauri dev

# 部署方法
* **终端中执行**：
git clone https://github.com/micromimo/RL-Matting-Agent.git

* **资源准备**:
将OONX格式的模型放入/src-tauri/target/debug/\_up_，U2系列模型放在/src-tauri/target/debug/models

* **终端中执行**：

  cd {项目根目录}

  npm install

  npm run tauri dev





# 以下全都是技术文档，也就是废话🫣





# RL Matting Agent - 项目技术文档

## 1. 项目概述

**RL Matting Agent** 是一个基于强化学习（Reinforcement Learning）与传统图像处理技术的智能抠图桌面应用。项目以桌面端应用形式交付，集成了 **RL Agent 决策（基于弱监督序列补丁决策的图像主体提取）**、**U²-Net / 轻量级语义分割模型（Rembg）**、以及 **传统数字图像处理** 三套可自定义的抠图方案，通过可视化界面供用户进行参数调节、步骤观察和实时结果对比。

其中，我们自己主要实现的是\*\*基于弱监督序列补丁决策的图像主体提取，\*\*rembg和传统图像处理方案均是集成了现有的开源Rust Create。

***

## 2. 技术栈

### 2.1 前端 UI 层

| 技术                            | 版本    | 用途             |
| ----------------------------- | ----- | -------------- |
| **React**                     | 18.3+ | 前端 UI 框架，组件化开发 |
| **TypeScript**                | 5.4+  | 前端类型安全         |
| **Vite**                      | 5.3+  | 前端构建工具，支持 HMR  |
| **Tailwind CSS**              | 3.4+  | CSS 原子化样式工具    |
| **@tauri-apps/api**           | 2.0+  | Tauri 前端桥接 API |
| **@tauri-apps/plugin-dialog** | 2.0+  | 文件选择对话框        |

### 2.2 后端 / 核心逻辑层（Rust）

| 技术                      | 版本           | 用途                                                                                                    |
| ----------------------- | ------------ | ----------------------------------------------------------------------------------------------------- |
| **Tauri**               | 2.x          | 跨平台桌面应用框架（WebView + Rust）                                                                             |
| **Rust**                | Edition 2021 | 后端系统级编程语言                                                                                             |
| **ort (ONNX Runtime)**  | 2.0.0-rc.12  | ONNX 模型推理引擎（核心）                                                                                       |
| **image**               | 0.25         | 图像编解码与基础变换（resize, crop 等）                                                                            |
| **imageproc**           | 0.25         | 传统图像处理算子库（Canny、Morphology、AdaptiveThreshold、DistanceTransform、BilateralFilter、ConnectedComponents 等） |
| **ndarray**             | 0.17         | 多维数组运算（与 ONNX Runtime 配合）                                                                             |
| **rayon**               | 1.10         | 数据并行处理（多线程加速 crop 等操作）                                                                                |
| **base64**              | 0.22         | Base64 编码（图像数据传输）                                                                                     |
| **serde / serde\_json** | 1.0          | 序列化 / 反序列化（前后端通信）                                                                                     |

### 2.3 参考/集成的开源项目（核心实现基础）

本项目实现了三方案融合的抠图系统，其中两个子方案基于开源生态：

1. **imgly/background-removal-rs (方案 B 的实现参考)**
   - 项目地址：`https://github.com/imgly/background-removal-rs`
   - 作用：Rust 版的 AI 背景移除库，集成了多种轻量级语义分割模型（U²-Net, ISNet, Silueta 等）
   - 集成方式：因版本兼容性问题（与 `ort 2.0` 协同困难），项目未直接引用该 crate，而是**参考其架构，自行基于** **`ort`** **实现了等价的 Rembg Processor**。在代码和 UI 中仍使用 "Rembg" 命名以指代这一 AI 背景移除方案。
2. **imageproc (方案 C 的依赖)**
   - 项目地址：`https://github.com/image-rs/imageproc`
   - 作用：`image` crate 生态的经典图像处理扩展库
   - 集成方式：直接作为核心依赖使用，方案 C 的所有传统图像处理算子均来自此库。

### 2.3 模型格式与运行时

| 模型                            | 格式      | 输入尺寸                                     | 用途         |
| ----------------------------- | ------- | ---------------------------------------- | ---------- |
| **Policy Network** (RL Agent) | `.onnx` | `[1, 3, 224, 224]` + bbox state `[1, 4]` | 强化学习决策网络   |
| **U²-Net**                    | `.onnx` | `[1, 3, 320, 320]`                       | 高质量显著性目标分割 |
| **U²-Net Human Seg**          | `.onnx` | `[1, 3, 320, 320]`                       | 人像分割专用     |
| **Silueta**                   | `.onnx` | `[1, 3, 320, 320]`                       | 轻量级人像分割    |

### 2.4 运行时特性

- **MPS / CoreML 加速**：`ort` 依赖启用 `coreml` feature，自动使用 Metal Performance Shaders 进行 GPU 加速推理
- **多线程推理**：ONNX Session 配置 `intra_threads = 4`

***

## 3. 核心技术架构

### 3.1 整体处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Image (RGB)                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │   Config Check & Early Return │  ← 根据开关组合选择路径
              └───────────────┬───────────────┘
                              │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐
   │  RL Loop    │   │ Rembg Only  │   │  Traditional   │
   │  (Step-by-  │   │ (Direct     │   │  (Direct       │
   │   Step Env) │   │  Segmentation)│  │   Processing)  │
   └──────┬──────┘   └──────┬──────┘   └────────┬────────┘
          │                  │                  │
          ▼                  ▼                  ▼
   ┌─────────────────────────────────────────────┐
   │          Mask Generation Pipeline            │
   │  (Score Map + Threshold + Morphology + ...) │
   └───────────────────┬─────────────────────────┘
                       │
                       ▼
           Final RGBA Mask (Base64 / PNG)
```

### 3.2 三种处理方案详解

#### 方案 A：强化学习（RL）抠图

这是本项目的**核心创新**部分。RL Agent 被设计成一个\*\*"观察者"\*\*，通过逐步放大和定位来完成抠图任务。

**Action Space**（7 个离散动作）：

```
0: Move Left     1: Move Right
2: Move Up       3: Move Down
4: Zoom In       5: Zoom Out
6: Trigger (结束 / 完成)
```

**State Representation**（两个输入分支）：

1. **Patch 分支**：将当前边界框（Bounding Box）裁剪并 Resize 到 `224 × 224` RGB，作为图像特征输入
2. **几何分支**：归一化的边界框状态 `[x, y, w, h]`，均为相对坐标（0\~1）

**Policy Network 输入 / 输出**：

- **Input 1**: `[1, 3, 224, 224]` —— 当前 BBox 的 RGB Patch
- **Input 2**: `[1, 4]` —— BBox 归一化状态
- **Output 1**: `Logits [7]` —— 每个动作的得分
- **Output 2**: `Confidence [1]` —— 动作置信度
- **Output 3**: `Feature Vector [N]` —— 中间层特征向量（用于后续 Mask 生成）

**推理循环**：

1. 从全屏 BBox 开始，每一步调用 Policy Network 预测动作
2. 依据 `max_steps`（默认 30）和 `min_steps_before_trigger`（15）约束终止
3. 前 15 步强制避免 Trigger 动作（防止过早结束）
4. 当 Agent 输出 Trigger 且步数 >= 15 时，调用中心特征提取生成 Mask

#### 方案 B：AI 背景移除（Rembg）

基于 **U²-Net** 系列 ONNX 模型的直接语义分割方案。

**处理流程**：

1. 将目标区域（或全屏）Resize 到模型输入尺寸（320×320）
2. RGB 归一化 `pixel / 255.0` → `[1, 3, H, W]` Tensor
3. 送入 ONNX Runtime Session 推理
4. 输出 shape `[1, 1, 320, 320]` 的单通道 Alpha Mask
5. Resize 回原始尺寸
6. 根据阈值（0\~1，默认 0.5）生成二值化 / 软 Alpha 通道

**支持模型**：

| 模型 Key            | 文件                     | 特点          |
| ----------------- | ---------------------- | ----------- |
| `u2net`           | `u2net.onnx`           | 通用目标分割，质量最好 |
| `u2net_human_seg` | `u2net_human_seg.onnx` | 人像专用分割      |
| `silueta`         | `silueta.onnx`         | 轻量级人像分割     |

**关键参数**：

- `rembg_threshold`：Alpha 阈值，控制前景判定严格度
- `rembg_binary_mode`：开启后输出二值化 Mask，关闭则输出软 Alpha

#### 方案 C：传统图像处理

纯图像处理方案，无需深度学习模型，基于 OpenCV 风格的经典算法流水线。

**处理步骤**：

1. **背景色估计**：从图像中心区域采样，估计背景 RGB 值
2. **Score Map 生成**：结合 HSV 色彩差异 + 欧氏距离 + 中心先验，计算每个像素的前景置信分数
3. **双边滤波（可选）**：`bilateral_filter` 保边降噪
4. **Canny 边缘检测**：提取边缘图
5. **自适应阈值**：大津法（Otsu）或自适应阈值化
6. **形态学操作**：闭运算 + 开运算，填补孔洞、去除噪点
7. **连通域分析**：保留最大连通域（`trad_min_component_ratio`）
8. **距离变换（可选）**：移除边界像素，精细化 Mask

### 3.3 混合模式与配置开关

用户可以在 UI 中自由组合三套方案：

| 组合                  | 行为                      |
| ------------------- | ----------------------- |
| 仅 RL                | 完整强化学习推理流程              |
| 仅 Rembg             | 直接语义分割（跳过 RL 循环）        |
| 仅 Traditional       | 直接传统处理（跳过 RL 循环）        |
| RL + Rembg          | RL 定位 + Rembg 精修        |
| Rembg + Traditional | Rembg 分割 + 传统后处理优化      |
| RL + Traditional    | RL 定位 + 传统处理优化          |
| 全开                  | RL 定位 → Rembg 精修 → 传统优化 |
| 全关                  | 错误提示（至少启用一种）            |

**注意**：当 RL 未启用时（`enable_rl_model = false`），系统自动跳过 RL 循环，直接根据剩余开关组合调用相应的 Mask 生成路径。

***

## 4. 模型训练与数据

### 4.1 Policy Network 训练

#### 4.1.0 核心训练范式解析

在详细介绍之前，先回答三个核心技术问题：

##### 问题 1：什么是"在线 RL 训练"？与离线训练有何区别？

**离线训练（Offline Training）**：

- 先用数据集训练好一个模型，部署后模型权重不再改变
- 例如：U²-Net 用 DUTS 数据集训练好后，推理时直接使用固定权重

**在线 RL 训练（Online Reinforcement Learning）**：

- 训练过程中，**每一步决策都由 Agent 在环境中实时交互产生**
- 数据是**在训练时生成的**，而非提前固定
- 本项目中的实现：
  ```
  每个 Episode 的流程：
  ┌─────────────────────────────────────────────┐
  │ 1. 从 Dataset 加载 32 张图像（Batch）        │
  │ 2. 每张图像作为一个"环境"                    │
  │ 3. Agent 从中心 BBox 开始                    │
  │ 4. 在每张图像上连续执行 15 步动作（Move/Zoom）│
  │ 5. 每一步都使用 Policy Network 做决策        │
  │ 6. 根据决策结果更新 BBox，计算 Reward        │
  │ 7. 收集整个 Episode 的轨迹（Trajectory）     │
  │ 8. 使用 REINFORCE 算法更新 Policy Network    │
  └─────────────────────────────────────────────┘
  ```
- 关键代码在 [training.py: run\_episode\_batch()](file:///Users/micromimo/VSCode_Projects/RL/finalWork/rl-matting-app/train/training.py#L274-L396)：
  ```python
  # Agent 在环境中逐步决策
  for step in range(self.max_steps):      # 最多 15 步
      logits = model.action_head(fused)   # Policy Network 输出
      actions = dist.sample()             # 采样动作
      cx, cy, bw, bh = apply_action(...) # 环境响应
      reward = compute_iou_reward(...)    # 计算奖励
  ```

##### 问题 2：什么是"弱监督序列补丁决策"？

"弱监督"体现在**训练数据的标注方式**上：

**传统强监督训练**：

- 数据需要精确的像素级 Mask 作为 Ground Truth
- 每个像素都有正确答案

**本项目的弱监督方式**：

- 使用 Open Images 数据集的**物体级标注**（而非像素级 Mask）
- 训练时加载的 `localization.txt` 包含：
  ```
  图像路径, 分割掩码路径, 忽略区域路径
  ```
- 当掩码存在时，使用 IoU Reward；**当掩码不存在时，使用启发式 Reward**
- 这种方式让模型可以在**没有像素级标注的数据上训练**

**序列补丁决策**的含义：

- "补丁"（Patch）：每次只关注 BBox 内的局部区域，而非整张图
- "序列"（Sequence）：Agent 做连续多步决策，逐步定位目标
- "决策"（Decision）：每一步选择 7 个动作中的一个（Move/Zoom/Trigger）

##### 问题 3：没有参考答案时，Reward 如何计算？

这是 RL 训练的关键设计。本项目使用了**混合 Reward 策略**：

**Case A：有像素级掩码时（IoU Reward）**

```python
def _compute_iou_reward(self, bbox, mask_tensor):
    # 计算 BBox 与 GT Mask 的 IoU
    pred_box = torch.zeros(mask_h, mask_w)
    pred_box[y1:y2, x1:x2] = 1.0
    
    intersection = (pred_box * mask).sum()
    union = pred_area + mask_area - intersection
    iou = intersection / union
    
    return iou  # 直接用 IoU 作为 Reward
```

- 当 BBox 越来越贴合目标时，IoU 增大，Reward 变高
- 这是最直接的"正确答案"反馈

**Case B：没有像素级掩码时（启发式 Reward）**

```python
if mask_batch is None or mask_batch[b] is None:
    if action == Trigger:
        reward = 0.5      # 提前结束给中等奖励
    else:
        reward = 0.05     # 每步小奖励
```

- 没有 Ground Truth 时，使用**固定规则的稀疏奖励**
- Trigger 动作给较高奖励（鼓励 Agent 尽快定位）
- 其他动作给小奖励（鼓励探索）

**完整的 Reward 合成公式**：

```
step_reward = base_reward(iou) + trigger_bonus

其中：
  base_reward(iou) = {
    iou:  if mask 存在           # 直接使用 IoU
    0.5:  if trigger and 无 mask # Trigger 给 0.5
    0.05: if 其他 and 无 mask    # 普通步骤给 0.05
  }
  
  trigger_bonus = 0.2 if action == Trigger else 0
```

**这种设计的核心优势**：

1. **渐进式学习**：即使没有精确标注，Agent 也能学会"何时停止"
2. **数据利用最大化**：有标注的数据用 IoU 训练，无标注的数据用启发式训练
3. **课程学习**：先在有标注数据上学习基本定位能力，再在无标注数据上泛化

#### 4.1.1 数据集选择：Google Open Images (OID)

**选择理由**：

- **多样性**：Open Images 涵盖 600 多个物体类别，涵盖从单细胞生物到人造物体的丰富视觉变化
- **定位标注**：OID 提供了 **Bounding Box 标注** 与 **像素级分割掩码（Segmentation Mask）**，非常适合"强化学习定位 + 精确分割"的训练需求
- **数据规模**：足够大的规模支持复杂 RL Agent 的稳定训练
- **社区维护**：持续更新，广泛应用于目标检测、分割等 CV 任务的训练

**数据集组成**：

由于项目体量限制，我们只从庞大的数据集中选取了6000张图像，按类别分层采样（每类最多 50 张），保证类别均衡。

| Split   | 图像数量  | 格式  | 标注类型                              |
| ------- | ----- | --- | --------------------------------- |
| `train` | 5,000 | JPG | 类别标签 (`class_labels.txt`)         |
| `val`   | —     | JPG | 类别标签                              |
| `test`  | 1,000 | JPG | 类别标签 + 像素级掩码 (`localization.txt`) |

**图像特点**：

- 彩色 RGB 图像，来源于 Google 图像搜索
- 尺寸不固定，训练时统一 Resize 到 224×224
- 背景复杂度高（室内/室外/特殊光照/运动模糊等）
- 物体在画面中的位置、尺度、遮挡程度变化极大，正好模拟 RL Agent 需要处理的复杂场景

**目录结构**：

```
dataset/OpenImages/
├── train/
│   ├── 015p6/         # 图像 ID 前缀目录
│   │   ├── 2d10e5e8e8f0764a.jpg
│   │   └── ...
│   └── ...
└── test/
    └── ...

metadata/OpenImages/
├── train/
│   ├── class_labels.txt    # 图像路径 -> 类别索引 映射
│   └── localization.txt    # 图像路径 -> 掩码路径 映射（部分存在）
└── test/
    ├── class_labels.txt
    └── localization.txt    # 7,631 条标注（像素级分割掩码）
```

**数据划分策略**：

- **Training Set**：使用 `train` split 的 5,000 张图像，随机打散后作为 RL Agent 的交互环境
- **Validation Set**：使用 `test` split 的 1,000 张图像的一部分用于训练中间评估
- **划分依据**：遵循 Open Images 官方划分方式，避免数据泄漏
- **数据增强**：仅使用 `Resize(224,224) + Normalize(ImageNet mean/std)` 简单处理，因为 RL Agent 需要看到的是原始视觉多样性

#### 4.1.2 训练逻辑与评估指标

##### 4.1.2.1 最核心评估指标：Validation Reward (val\_reward)

在本项目中，**`val_reward`（验证集平均奖励）是评估模型训练效果的最核心指标**，它直接决定了最佳模型（`policy_best.pt`）的保存。

**代码位置**：[training.py: evaluate\_model()](file:///Users/micromimo/VSCode_Projects/RL/finalWork/rl-matting-app/train/training.py#L454-L524) 与 [training.py: main()](file:///Users/micromimo/VSCode_Projects/RL/finalWork/rl-matting-app/train/training.py#L705-L717)

```python
# 关键代码：val_reward 是唯一用于保存 best model 的指标
if val_reward > best_reward:
    best_reward = val_reward
    save_path.parent / "policy_best.pt"  # 保存最佳模型
```

**val\_reward 的计算逻辑**：

```python
def evaluate_model(model, val_loader, device):
    total_reward = 0.0
    total_steps = 0
    total_triggers = 0
    num_batches = 0

    for batch_data in val_loader:
        # Agent 在验证集上执行最多 15 步推理
        for step in range(15):
            actions = torch.argmax(probs)  # 使用 argmax，确定性决策
            
            # 计算是否所有样本都 Trigger
            if trigger.all():
                break
        
        # 每个 Batch 完成后累加统计
        batch_reward += 1.0
        total_reward += batch_reward
        total_steps += step + 1
        total_triggers += trigger.sum().item()
        num_batches += 1

    # 返回三个核心指标
    avg_reward = total_reward / max(1, num_batches)     # ← 最核心
    avg_steps = total_steps / max(1, num_batches)       # 辅助指标
    avg_triggers = total_triggers / max(1, num_batches) # 辅助指标
```

**val\_reward 的含义**：

- 范围：`[0, 1]`（每个 Batch 贡献 1.0 的满分）
- 本质：**完成率** —— 验证集中有多少比例的样本 Agent 能成功完成定位任务
- 解读：
  - `val_reward = 1.0`：所有样本 Agent 都能成功定位
  - `val_reward = 0.5`：一半样本成功定位
  - 与训练时的 `avg_reward` 不同：训练时是实际 Reward 累加，验证时是二值化的任务完成情况

##### 4.1.2.2 辅助评估指标

除了 `val_reward`，还有两个重要的辅助指标用于诊断 Agent 行为：

| 指标             | 计算方式              | 含义                        | 理想值                 |
| -------------- | ----------------- | ------------------------- | ------------------- |
| `val_reward`   | 成功完成的 Batch 比例    | **核心指标** — Agent 定位成功率    | `↑` 越高越好（上限 1.0）    |
| `avg_steps`    | 平均每个 Episode 的步数  | Agent 效率 — 步数越少越快完成       | `↓` 越低越好（下限由任务难度决定） |
| `avg_triggers` | Trigger 动作出现的平均次数 | Agent 终结倾向 — 反映是否学会"何时停止" | 适中（过低→不会停止；过高→过早停止） |

**训练日志示例**：

```
[Epoch 20/5000] avg_loss=145.6408 avg_reward=2.095 time=7.7s
  [VAL] reward=0.875 avg_steps=11.2 triggers=3.4
  [NEW BEST] val_reward=0.875, saved to models/policy_best.pt
```

##### 4.1.2.3 评估指标的设计哲学

为什么本项目选择 `val_reward`（完成率）作为核心指标，而不是传统的 IoU Accuracy？

**原因 1：与训练目标对齐**

- 训练时 Agent 的目标是"成功定位目标"，而非"精确分割"
- RL 训练中的 Reward 本质上就是在优化"能否完成任务"

**原因 2：规避 GT Mask 缺失问题**

- 验证集中并非所有样本都有像素级 GT Mask
- `val_reward` 不需要像素级标注，只需判断 Agent 是否完成了定位流程
- 这使得评估可以在更广泛的数据集上进行

**原因 3：符合实际应用场景**

- 在实际 App 使用中，用户关心的是"能否抠出主体"（成功率）
- 而非"Mask 的 IoU 是多少"（精确但次要）
- 因此用成功率作为核心指标更贴合产品价值

**原因 4：稳定可比较**

- `val_reward` 是 `[0, 1]` 范围内的连续值，便于跨训练会话比较
- IoU 受图像内容影响大，不同数据集之间可比性差

##### 4.1.2.4 Reward 设计与训练配置

**算法**：Actor-Critic 强化学习（REINFORCE with Baseline）

**Reward 设计**：

- **基础正奖励**：每一步 Agent 都会收到 `+0.05` 的小奖励，鼓励有效探索
- **Trigger 奖励**：Agent 执行 Trigger 动作时额外 `+0.2`，鼓励快速定位
- **IoU 奖励**：使用 **Intersection over Union** 计算当前 BBox 与 GT Mask 的重叠程度
  ```
  reward_iou = intersection(pred_box, gt_mask) / union(pred_box, gt_mask)
  ```
- **Step 奖励**：基于 IoU 的连续奖励（而非稀疏奖励），提供更密集的学习信号

**训练 Pipeline**：

```
1. 图像预处理（ImageNet Normalization）
2. 预计算特征（MobileNetV3-Small Backbone）
3. RL 交互循环（最多 15 步 / Episode）
4. 计算 Per-Step Reward（IoU 基础）
5. 折扣回报计算（γ = 0.99）
6. Policy Loss + Value Loss 计算
7. AdamW 优化器更新
8. Learning Rate Cosine Annealing 调度
```

**训练配置**：

| 参数                  | 值                   | 说明                        |
| ------------------- | ------------------- | ------------------------- |
| Optimizer           | AdamW               | 权重衰减 1e-5                 |
| Learning Rate       | 3e-4                | 初始学习率                     |
| LR Scheduler        | CosineAnnealingLR   | T\_max=500, eta\_min=3e-6 |
| Batch Size          | 32                  | 每个 Episode 并行 32 张图像      |
| Max Steps / Episode | 8 \~ 15             | Agent 每个 Episode 的最大步数    |
| Total Params        | 1,142,184           | 可训练参数 493,720             |
| Epochs              | 5,000               | 总训练轮数                     |
| Device              | MPS (Apple Silicon) | Mac 上使用 MPS 加速            |
| Gradient Clipping   | 1.0                 | 防止梯度爆炸                    |

**训练特点**：

- **预训练 Backbone**：使用 MobileNetV3-Small 预训练权重（ImageNet），冻结前若干层
- **在线 RL 训练**：在训练过程中，Agent 与环境实时交互产生轨迹数据，而非离线固定数据集（详见 4.1.0）
- **BBox 状态编码**：使用连续的归一化坐标（cx, cy, w, h）而非离散网格
- **混合 Reward**：IoU Reward（有标注）+ 启发式 Reward（无标注），最大化数据利用效率

#### 4.1.3 模型格式转换

**PyTorch → ONNX 导出**：

```python
torch.onnx.export(
    wrapped_model,
    (dummy_patch, dummy_bbox),           # 双输入
    save_path,
    input_names=["patch", "bbox_state"],
    output_names=["action_logits", "confidence", "feature"],
    dynamic_axes={...},
    opset_version=17,
)
```

**ONNX 模型规格**：

| 项目    | 规格                                                                |
| ----- | ----------------------------------------------------------------- |
| 文件大小  | \~7 MB                                                            |
| 输入    | `patch [1, 3, 224, 224]` + `bbox_state [1, 4]`                    |
| 输出    | `action_logits [1, 7]` + `confidence [1, 1]` + `feature [1, 128]` |
| Opset | 17                                                                |
| 推理引擎  | ONNX Runtime (ort 2.0.0-rc.12)                                    |
| 优化级别  | `GraphOptimizationLevel::All`                                     |
| 线程配置  | 4 线程                                                              |

**可用 Checkpoint**：

```
models/
  policy_best.pt              # 最佳模型
  policy_checkpoint_epoch*.pt  # 每 10 epoch 保存的 Checkpoint
  policy_network.onnx         # 最终导出的 ONNX 模型
```

### 4.2 Rembg 预训练模型

项目使用公开的 **U²-Net** 系列预训练权重：

- 由 `rembg` (Python) / `rembg-rs` 社区维护
- 训练数据：DUTS-TR + DUT-TE + HKU-IS + PASCAL VOC 等显著性检测数据集
- 直接下载 ONNX 格式权重，**无需重新训练**

***

## 5. 推理侧逻辑详解

### 5.1 RL Engine（`ml_engine.rs`）

```rust
// 模型加载
RlEngine::load(model_path)
// → 初始化 ONNX Runtime Session
// → 配置 GraphOptimizationLevel::All + 4 线程

// 推理（双输入模式）
fn infer(&mut self, patch: &[f32], bbox_state: &[f32]) -> Result<(u32, f32)>
// Input:  patch [3*224*224]  →  [1, 3, 224, 224] Tensor
// Input:  bbox  [4]          →  [1, 4] Tensor
// Output: argmax(logits[7]), confidence

// 推理（带特征提取模式）
fn infer_with_feature(...) -> Result<(u32, f32, Vec<f32>)>
// 额外输出 Feature Vector，用于 Score Map 计算
```

### 5.2 Rembg Processor（`rembg_processor.rs`）

```rust
// 模型动态检测
detect_input_hw(session)
// → 从 ONNX Session 的 Input Shape 自动推断输入尺寸
// → 支持 320x320 / 256x256 等不同模型

// 单 BBox 推理
mask_for_bbox(image, config, bbox)
// 1. 裁剪 BBox 区域
// 2. Resize 到模型输入尺寸
// 3. RGB 归一化 [0,1]
// 4. ONNX 推理
// 5. 输出 Shape 解析（支持 [1,1,H,W] / [1,H,W] / [H,W]）
// 6. Resize 回原尺寸
// 7. 阈值化 → 生成二值 Mask
```

### 5.3 传统图像处理（`img_processor.rs`）

**Score Map 计算逻辑**：

```
score_map(x, y) = Σ(wi × si) / Σ(wi)

其中：
  s_color_hsv  = 1 - |ΔH|/60        (色调接近度)
  s_color_sat  = |ΔS|             (饱和度差异)
  s_color_val  = |ΔV|/255         (明度差异)
  s_bg_euclid  = √(Σ(Ri-Rbg)²)/441.67  (RGB 欧氏距离)
  s_center     = 1 - √(dx²+dy²)×2  (中心先验)
  w: [2.0, 3.0, 1.0, 0.3, 0.2]    (各分量权重)
```

**后处理流水线**：

```
Score Map
  → Normalize (min-max)
  → Bilateral Filter (可选)
  → Canny Edge Detection
  → Adaptive Threshold (Otsu / Block-based)
  → Morphology (闭运算 → 开运算)
  → Connected Components Analysis (保留最大连通域)
  → Distance Transform (可选，边缘精修)
  → Final Binary Mask
```

***

## 6. App 功能详解

### 6.1 核心功能模块

1. **图像加载与预览**
   - 支持 JPEG / PNG / BMP 等常见格式
   - 图像自适应缩放显示
   - 实时进度反馈
2. **RL 循环可视化**
   - 逐步展示 Agent 每一步的动作（Move/Zoom/Trigger）
   - 实时显示 BBox 位置与大小变化
   - 展示 Confidence 曲线
   - 步骤进度条
3. **参数控制面板（Processing Options）**
   - **强化学习模型**开关 + 参数
     - Max Steps（最大步数）
     - Confidence Threshold（置信度阈值）
   - **AI 背景移除**开关 + 参数
     - 模型选择（U²-Net / Human Seg / Silueta）
     - Threshold（分割阈值）
     - Binary Mode（二值化模式）
   - **传统图像处理**开关 + 参数
     - Canny Low / High（边缘检测阈值）
     - Morphology Radius（形态学半径）
     - Min Component Ratio（最小连通域比例）
     - Edge Weight（边缘增强权重）
     - Adaptive Threshold Block / C（自适应阈值参数）
     - Bilateral Filter（双边滤波开关及参数）
     - Distance Transform（距离变换开关及权重）
4. **结果输出**
   - 透明背景 PNG 导出（RGBA 格式）
   - Base64 传输与预览
   - 多方案结果对比

### 6.2 UI 技术特性

- **Tauri 窗口配置**：
  - `decorations: true`（原生标题栏）
  - `theme: auto`（自动深色模式跟随系统）
  - 支持 macOS / Windows / Linux 跨平台
- **事件通信机制**：
  ```rust
  // 后端 → 前端
  app.emit("rl-step-update", &update)
  app.emit("rl-pipeline-start", &stages)
  app.emit("rl-pipeline-stage", &stage)

  // 前端监听
  useRlStepListener() // React Hook
  ```
- **Pipeline 阶段追踪**：
  ```
  load_image → init_bbox → load_model → rl_inference → generate_mask → complete
  ```

***

## 7. 代码结构

```
rl-matting-app/
├── src/                          # React 前端
│   ├── App.tsx                   # 主应用
│   ├── main.tsx                  # 入口
│   ├── index.css                 # Tailwind + 自定义样式
│   ├── components/
│   │   ├── ControlPanel.tsx      # 参数控制面板
│   │   ├── ImageCanvas.tsx       # 图像预览 + BBox 可视化
│   │   ├── GlassCard.tsx         # UI 组件
│   │   └── MetricsChart.tsx      # 指标图表
│   └── hooks/
│       └── useRlStepListener.ts  # RL 步骤监听 Hook
├── src-tauri/                    # Rust 后端
│   ├── src/
│   │   ├── main.rs              # Tauri 入口
│   │   ├── lib.rs               # 主逻辑：RL Loop + 命令定义
│   │   ├── ml_engine.rs        # RL Policy Network 推理引擎
│   │   ├── rembg_processor.rs  # Rembg/U²-Net 推理处理器
│   │   └── img_processor.rs     # 传统图像处理 + Score Map
│   ├── models/                  # ONNX 模型
│   │   ├── u2net.onnx
│   │   ├── u2net_human_seg.onnx
│   │   └── silueta.onnx
│   ├── Cargo.toml               # Rust 依赖配置
│   └── tauri.conf.json          # Tauri 应用配置
└── models/                      # PyTorch 训练 Checkpoint
    ├── policy_best.pt
    └── policy_checkpoint_epoch*.pt
```

### 7.1 核心 Rust 模块

| 模块                   | 职责                         | 关键类型                                              |
| -------------------- | -------------------------- | ------------------------------------------------- |
| `lib.rs`             | 命令入口、RL 循环编排、前后端通信         | `ProcessingConfig`, `BoundingBox`, `RlStepUpdate` |
| `ml_engine.rs`       | Policy Network 加载与推理       | `RlEngine`                                        |
| `rembg_processor.rs` | U²-Net 系列模型加载与推理           | `RembgProcessor`                                  |
| `img_processor.rs`   | 图像加载、裁剪、Score Map、Mask 后处理 | `generate_mask()`, `generate_rembg_only_mask()`   |

### 7.2 关键数据结构

```rust
pub struct ProcessingConfig {
    enable_rl_model: bool,       // 启用强化学习
    enable_traditional: bool,    // 启用传统处理
    enable_rembg: bool,          // 启用 AI 背景移除
    rl_max_steps: u32,           // RL 最大步数
    rl_confidence_threshold: f64, // RL 置信度阈值
    rembg_model: String,         // Rembg 模型名
    rembg_threshold: f64,        // Rembg 阈值
    rembg_binary_mode: bool,     // Rembg 二值模式
    trad_canny_low/high: f64,    // Canny 阈值
    trad_morphology_radius: u32, // 形态学半径
    // ... 更多参数
}

pub struct BoundingBox {
    x: u32, y: u32,
    width: u32, height: u32,
}

pub struct RlStepUpdate {
    step: u32,
    action_taken: String,
    bbox: BoundingBox,
    confidence: f32,
    is_finished: bool,
    mask_base64: Option<String>,
}
```

***

## 8. 构建与运行

### 8.1 开发模式

```bash
# 安装前端依赖
cd rl-matting-app
npm install

# 启动 Tauri 开发模式（同时启动前端 + Rust）
npm run tauri dev
```

### 8.2 构建生产版本

```bash
npm run tauri build
# 产物位于 src-tauri/target/release/bundle/
```

### 8.3 环境要求

| 依赖      | 版本要求                    |
| ------- | ----------------------- |
| Node.js | 18+                     |
| npm     | 9+                      |
| Rust    | 1.70+                   |
| 操作系统    | macOS / Windows / Linux |

***

## 9. 已知限制与未来方向

1. **模型输入尺寸固定**：U²-Net 系列使用 320×320，大图会被缩放到该尺寸再推理，可能损失细节
2. **RL 训练数据依赖**：Policy Network 需要与实际使用场景匹配的训练数据，迁移性待验证
3. **多 GPU / 批量推理**：当前为单张顺序推理，未实现批量优化
4. **模型格式扩展**：可扩展支持 ISNet-General (1024×1024)、SAM (Segment Anything) 等更大模型
5. **训练可视化**：当前仅支持推理可视化，训练过程可视化可作为未来功能

***

## 10. 参考资源

- [imgly/background-removal-rs (本项目方案 B 的实现参考)](https://github.com/imgly/background-removal-rs)
- [U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09070)
- [U²-Net Official Implementation](https://github.com/xuebinqin/U-2-Net)
- [imageproc (本项目方案 C 的核心依赖)](https://github.com/image-rs/imageproc)
- [Rembg (Python)](https://github.com/danielgatis/rembg)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Tauri v2 Documentation](https://v2.tauri.app/)



