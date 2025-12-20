"""
常量定义模块
包含所有游戏和AI参数的集中管理
"""

# 屏幕参数
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

# 对象大小
TANK_SIZE = 30
TANK_HITBOX_SCALE = 0.8  # 碰撞箱缩放比例，使其比视觉略小，防止卡死
BULLET_SIZE = 6

# 速度参数
TANK_SPEED = 4
ROTATION_SPEED = 4
BULLET_SPEED = 5

# 游戏规则
MAX_BOUNCES = 7  # 子弹最大反弹次数
BULLET_COOLDOWN = 20
MAX_BULLETS_PER_TANK = 5  # 每个坦克最大子弹数
FPS = 60

# 网格寻路参数
GRID_SIZE = 20  # 网格大小（增大以减少过度敏感）
GRID_BUFFER_RADIUS = 0  # 墙壁缓冲区半径（设为0，让bot更容易通过）

# 寻路更新频率
PATHFINDING_UPDATE_FREQ = 10  # 每N帧更新一次路径

# AI 视线和决策参数
LINE_OF_SIGHT_CHECK = True
VISION_DISTANCE = 300  # 视线检测距离（像素）
ANGLE_TOLERANCE = 10  # 瞄准容差（度）
NODE_ARRIVAL_DISTANCE = 20  # 到达路点的距离阈值（增大以减少卡顿）

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (200, 200, 200)

# RL 环境参数
MAX_STEPS_PER_EPISODE = 1500
# 观察空间: 16 (Agent+Enemy+Relative) + 40 (子弹) + 8 (射线检测墙壁距离)
OBSERVATION_SIZE = 64


# 奖励参数（稀疏奖励，只在击毁时给奖励）
STEP_PENALTY = -0.002        # 稍微增加每步惩罚，鼓励尽快结束战斗
BULLET_HIT_AGENT_REWARD = -30.0  # 被击中惩罚
FRIENDLY_FIRE_PENALTY = -20.0   # 自杀额外惩罚
ENEMY_HIT_REWARD = 150.0        # 恢复击杀奖励，增加诱惑力
TIMEOUT_PENALTY = -100.0         # 大幅增加超时惩罚，严厉打击“摆烂”行为
COLLISION_PENALTY = -0.2        # 大幅增加撞墙惩罚，必须比待机惩罚更重
IDLE_PENALTY = -0.1             # 待机惩罚

# 辅助奖励
REWARD_ACCURATE_SHOT = 1.0     # 提高精准射击奖励
REWARD_SHOOT = -0.02            # 进一步降低射击成本
REWARD_FORWARD_MOVE = 0.001     # 极小的正向前进奖励，引导坦克优先选择前进而非倒退
REWARD_SURVIVAL = 0  # 存活奖励

# 调试模式
DEBUG_RENDER_GRID = False  # 是否绘制网格
DEBUG_RENDER_PATH = False   # 是否绘制寻路路径
