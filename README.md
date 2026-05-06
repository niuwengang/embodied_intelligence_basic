# embodied-intelligence-fundamentals

具身智能基础知识学习仓库，系统梳理具身智能所需的核心算法与工程模式，为理解和实现 ACT、Diffusion Policy、TD-MPC 等策略算法打基础。

---

## 目录结构

```
embodied-intelligence-fundamentals/
├── generative_models/          # 生成模型
│   ├── auto_encoder.py         # 标准自编码器（AE）
│   └── var_auto_encoder.py     # 变分自编码器（VAE）
└── design_pattern/             # 工程设计模式
    └── registry_demo.py        # 注册表模式
```

---

## 模块说明

### 生成模型 `generative_models/`

具身智能中的策略学习（如 ACT）广泛使用生成模型对动作序列建模，本模块从零实现两个核心模型。

#### 自编码器 AE（`auto_encoder.py`）

在 MNIST 上实现标准 Encoder-Decoder 结构，将 784 维图像压缩到 32 维隐向量后重建。

网络结构：`784 → 256 → 32 → 256 → 784`

- 损失函数：Binary Cross Entropy（`reduction='mean'`）
- 优化器：Adam（lr=1e-3），训练 10 个 epoch
- 训练完成后保存权重至 `checkpoints/ae_model.pth`，可视化结果保存至 `ae_reconstruction.png`

#### 变分自编码器 VAE（`var_auto_encoder.py`）

在 AE 基础上引入概率建模，编码器输出隐变量的分布参数 (μ, log σ²) 而非固定点，使隐空间连续可插值，支持随机生成新样本。

网络结构：`784 → 256 → (μ, log σ²)[32] → 重参数化 → 32 → 256 → 784`

核心改进：
- **重参数化技巧**：`z = μ + σ·ε`，将随机性从网络参数中分离，使梯度可回传
- **损失函数（ELBO）**：重建损失（BCE，`reduction='sum'`）+ β × KL 散度，KL 项约束隐分布接近标准正态 N(0, I)
- **β-VAE**：调大 β（默认 1.0）可增强隐空间的解耦性，但重建质量可能下降
- **随机生成**：训练后可直接从 N(0, I) 采样隐向量并解码，生成新样本

```
x → Encoder → (μ, log σ²) → 重参数化 → z → Decoder → x̂
```

KL 散度闭式解：`KL = -0.5 × Σ(1 + log σ² - μ² - σ²)`

- 优化器：Adam（lr=1e-3），训练 10 个 epoch
- 推理时使用 `weights_only=True` 安全加载权重
- 训练完成后保存权重至 `checkpoints/vae_model.pth`，可视化结果保存至 `vae_reconstruction.png`

---

### 设计模式 `design_pattern/`

#### 注册表模式（`registry_demo.py`）

具身智能框架（如 LeRobot）通过注册表模式管理多种策略配置，支持从配置文件的字符串名称动态实例化对应策略类，无需修改框架代码即可扩展新策略。

```python
@Registry.register("act")
@dataclass
class ACTConfig(Registry):
    chunk_size: int = 100
    use_vae: bool = True

@Registry.register("diffusion")
@dataclass
class DiffusionConfig(Registry):
    num_steps: int = 50
    beta_schedule: str = "cosine"

# 从配置文件读到 "type": "act" 后动态创建
cfg = Registry.create("act", chunk_size=50)
print(cfg.type)  # "act"
```

---

## 快速开始

```bash
# 训练标准自编码器
python generative_models/auto_encoder.py

# 训练变分自编码器
python generative_models/var_auto_encoder.py

# 运行注册表模式演示
python design_pattern/registry_demo.py
```

---

## 环境依赖

```bash
pip install torch torchvision loguru tqdm matplotlib
```

---

## 参考

- [ACT: Action Chunking with Transformers](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [TD-MPC2](https://arxiv.org/abs/2310.16828)
- [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl)
