# 坦克大战 RL 环境 - 模块化版本

> 将单个 1100+ 行的文件拆分为 7 个专用模块，大幅降低耦合度

## 📁 文件结构

```
the_tank/
├── constants.py          # 常量定义（集中管理）
├── sprites.py            # 游戏对象：Wall, Bullet, Tank
├── pathfinding.py        # 寻路算法：GridMap, BFSPathfinder
├── bot_ai.py             # 机器人 AI：BotAI 决策逻辑
├── environment.py        # RL 环境：TankTroubleEnv（Gymnasium 标准）
├── main.py               # 主程序入口和演示
└── README.md             # 本文件
```

## 🎯 模块职责

| 模块 | 职责 | 关键类/函数 |
|------|------|-----------|
| `constants.py` | 参数集中管理 | `SCREEN_WIDTH`, `TANK_SPEED`, `ANGLE_TOLERANCE` 等 |
| `sprites.py` | 游戏实体 | `Wall`, `Bullet`, `Tank` 类及其行为 |
| `pathfinding.py` | 寻路系统 | `GridMap`（网格管理）, `BFSPathfinder`（BFS 算法） |
| `bot_ai.py` | AI 决策 | `BotAI`（战斗/寻路模式切换） |
| `environment.py` | RL 环境 | `TankTroubleEnv`（Gymnasium 接口） |
| `main.py` | 程序入口 | `run_demo()` 演示函数 |

## 🚀 快速开始

```bash
# 方式 1：运行演示
python main.py

# 方式 2：在自己的训练脚本中使用
from environment import TankTroubleEnv

env = TankTroubleEnv(render_mode="human")
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## 🔧 主要优化

### 1. **参数集中化**
所有魔数集中在 `constants.py`，便于快速调整：
```python
# 修改难度、速度等只需改 constants.py
TANK_SPEED = 4
ANGLE_TOLERANCE = 10
PATHFINDING_UPDATE_FREQ = 10
```

### 2. **高内聚、低耦合**
- `sprites.py` 只负责游戏对象渲染和碰撞
- `pathfinding.py` 纯算法，与游戏逻辑无关
- `bot_ai.py` 只依赖寻路和几何计算，不直接修改游戏状态
- `environment.py` 协调各模块，实现 Gymnasium 接口

### 3. **易于扩展**
新增功能只需在对应模块中添加：
- 新敌人类型 → 继承 `Tank` 或创建新类
- 新寻路算法 → 在 `pathfinding.py` 添加
- 新 AI 策略 → 在 `bot_ai.py` 添加方法

### 4. **可重用性**
- `pathfinding.py` 可用于其他项目（网格寻路）
- `sprites.py` 可独立测试或用于其他 Pygame 项目
- `environment.py` 符合 Gymnasium 标准，兼容所有 RL 框架

## 📊 行数对比

| 版本 | 文件数 | 总行数 | 单文件最大 | 耦合度 |
|------|--------|--------|-----------|--------|
| 原始 | 1 | 1100+ | 1100+ | 高 |
| 模块化 | 6+1 | ~1100 | 300 左右 | 低 |

## 🎮 环境特性

- **6 个动作**：待命、前进、后退、顺时针旋转、逆时针旋转、射击
- **30 维观测**：位置、角度、速度、冷却、周围子弹
- **智能 Bot**：
  - 视线检测 + 预判瞄准
  - BFS 寻路避开墙壁
  - 缓冲区膨胀防止卡角
  - 动态模式切换（战斗/寻路）

## 💡 使用示例

### 例 1：集中修改参数
```python
# constants.py
TANK_SPEED = 6         # 提速
ANGLE_TOLERANCE = 15   # 放宽瞄准容差
GRID_SIZE = 20         # 粗糙路径（更快计算）
```

### 例 2：自定义 AI
```python
# 在 bot_ai.py 中添加新方法
class BotAI:
    def aggressive_mode(self, bot, target):
        """激进模式：更快射击"""
        ...
```

### 例 3：训练 RL 模型
```python
import stable_baselines3 as sb3

env = TankTroubleEnv(render_mode=None)
model = sb3.PPO("MlpPolicy", env)
model.learn(total_timesteps=100000)
```

## 🔍 调试选项

`constants.py` 中提供调试开关：
```python
DEBUG_RENDER_GRID = False   # 显示网格和缓冲区
DEBUG_RENDER_PATH = True    # 显示寻路路径
```

## ⚙️ 依赖

- `pygame`
- `gymnasium`
- `numpy`

## 📝 许可

可随意修改和扩展。

---

**改进建议欢迎！** 如发现 bug 或有优化想法，欢迎反馈。
