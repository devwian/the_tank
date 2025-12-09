# 坦克大战 RL 完整使用指南

## 📋 快速开始

### 1️⃣ 运行演示（查看环境）

```bash
python main.py
```

### 2️⃣ 训练模型

#### 基础训练（推荐新手）
```bash
python train.py --mode basic --steps 500000
```

#### 带检查点的训练（推荐长期训练）
```bash
python train.py --mode checkpoint --steps 1000000 --checkpoint-freq 50000
```

**参数说明:**
- `--mode`: 训练模式
  - `basic`: 基础训练，最后保存一次
  - `checkpoint`: 每隔一段时间保存检查点，便于恢复
- `--steps`: 总训练步数（默认 1000000）
- `--checkpoint-freq`: 检查点保存频率（默认 20000）

**示例:**
```bash
# 短期测试
python train.py --mode basic --steps 100000

# 长期训练，每 30000 步保存一次
python train.py --mode checkpoint --steps 2000000 --checkpoint-freq 30000
```

### 3️⃣ 测试和演示模型

#### 测试已训练的模型
```bash
python test.py --mode test --model tank_ppo_model --episodes 10
```

#### 交互演示（可视化）
```bash
python test.py --mode play --model tank_ppo_model --episodes 3
```

#### 无渲染快速测试
```bash
python test.py --mode test --model tank_ppo_model --episodes 20 --no-render
```

**参数说明:**
- `--mode`: 模式选择
  - `test`: 测试模式，输出统计数据
  - `play`: 交互演示，显示画面
- `--model`: 模型路径（不需要 .zip 后缀）
- `--episodes`: 测试回合数
- `--no-render`: 不显示画面（仅测试模式）

---

## 🏗️ 项目结构

```
the_tank/
├── constants.py        # 参数配置
├── sprites.py          # 游戏对象
├── pathfinding.py      # 寻路算法
├── bot_ai.py           # AI 决策
├── environment.py      # RL 环境
├── main.py             # 演示脚本
├── train.py            # 训练脚本（已更新）
├── test.py             # 测试脚本（新增）
├── logs/               # 检查点保存目录
└── README.md           # 原始文档
```

---

## 🎮 训练工作流

### 完整流程示例

```bash
# 1. 查看环境演示
python main.py

# 2. 开始训练（500k 步快速测试）
python train.py --mode basic --steps 500000

# 3. 训练完成后，测试模型
python test.py --mode test --model tank_ppo_model --episodes 10

# 4. 如果效果好，进行长期训练
python train.py --mode checkpoint --steps 2000000 --checkpoint-freq 50000

# 5. 定期检查进度
python test.py --mode test --model ./logs/tank_model_50000 --episodes 5

# 6. 最终演示
python test.py --mode play --model ./logs/tank_model_final --episodes 5
```

---

## 📊 理解训练结果

### 训练日志示例

```
开始训练... 总步数: 500000
============================================================
| rollout/                 |
|   ep_len_mean           | 287
|   ep_rew_mean           | -2.3
| time/                   |
|   fps                   | 2500
|   iterations            | 244
|   time_elapsed          | 100
|   total_timesteps       | 500000
| train/                  |
|   approx_kl             | 0.005
|   clip_fraction         | 0.12
|   entropy_loss          | -0.5
|   learning_rate         | 0.0003
|   loss                  | 0.8
|   n_updates             | 2440
|   policy_gradient_loss  | -0.003
|   value_loss            | 1.2
============================================================
```

**关键指标:**
- `ep_rew_mean`: 平均每回合奖励，**越高越好**
- `ep_len_mean`: 平均回合长度
- `loss`: 损失函数值，**越低越好**
- `approx_kl`: KL 散度，**越小收敛越稳定**
- `clip_fraction`: 裁剪比例，应保持在 0.1-0.3

### 测试结果示例

```
📊 测试统计:
  总回合数: 10
  胜利次数: 7
  失败次数: 2
  平局次数: 1
  平均步数: 245.6
  平均奖励: 3.45
  胜率: 70.0%
```

---

## ⚙️ 高级配置

### 修改超参数

编辑 `train.py` 中的 PPO 参数：

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,      # 学习率（越小训练越慢但稳定）
    n_steps=2048,              # 每次更新采集的步数
    batch_size=64,             # 批处理大小
    n_epochs=10,               # 每次更新的循环次数
    gamma=0.99,                # 折扣因子
    gae_lambda=0.95,           # GAE 参数
    clip_range=0.2             # PPO 裁剪范围
)
```

### 修改环境参数

编辑 `constants.py`：

```python
TANK_SPEED = 4              # 坦克速度
ANGLE_TOLERANCE = 10        # AI 瞄准容差
PATHFINDING_UPDATE_FREQ = 10  # 寻路更新频率
DEBUG_RENDER_PATH = True    # 显示寻路路径
```

---

## 🐛 常见问题

### Q: 训练很慢怎么办？
**A:** 使用 `--no-render` 模式，或减少 `batch_size`

### Q: 模型性能不好怎么办？
**A:** 
- 增加训练步数（至少 1M 步）
- 调整学习率
- 检查观测值是否正常

### Q: 如何恢复中断的训练？
**A:** 从检查点继续训练
```python
from stable_baselines3 import PPO
from environment import TankTroubleEnv

env = TankTroubleEnv(render_mode=None)
model = PPO.load("./logs/tank_model_500000", env=env)
model.learn(total_timesteps=500000)
```

### Q: 找不到模型文件怎么办？
**A:** 检查模型是否已训练：
```bash
ls *.zip              # Windows: dir *.zip
ls ./logs/            # Windows: dir logs\
```

---

## 📈 性能优化建议

1. **快速迭代测试**（100k-500k 步）
   ```bash
   python train.py --mode basic --steps 100000
   ```

2. **标准训练**（1M 步）
   ```bash
   python train.py --mode basic --steps 1000000
   ```

3. **长期训练**（2M-5M 步，分段保存）
   ```bash
   python train.py --mode checkpoint --steps 5000000 --checkpoint-freq 100000
   ```

4. **定期评估**
   ```bash
   # 每训练 100k 步评估一次
   for model in ./logs/tank_model_*.zip; do
       python test.py --mode test --model "${model%.zip}" --episodes 5
   done
   ```

---

## 📝 脚本修改历程

### v1.0 (原始)
- 单个 `gyming.py` 文件
- 直接导入 `TankTroubleEnv`

### v2.0 (模块化)
- 拆分为 7 个模块
- 提高代码复用性和可维护性

### v2.1 (训练脚本更新)
- 更新 `train.py` 导入语句
- 新增命令行参数支持
- 优化 PPO 超参数

### v2.2 (测试脚本新增)
- 新增 `test.py` 用于模型评估
- 支持可视化演示
- 自动统计胜率等指标

---

## 🎓 学习资源

- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **PPO 算法**: https://openai.com/blog/openai-baselines-ppo/

---

## 🤝 贡献建议

欢迎改进！可以尝试：
- 更好的 AI 决策算法
- 新的地图设计
- 自定义奖励函数
- 多智能体对战

---

**祝你训练顺利！** 🚀
