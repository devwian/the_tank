"""
常量定义模块
包含所有游戏和AI参数的集中管理
"""

# 屏幕参数
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

# 对象大小
TANK_SIZE = 30
BULLET_SIZE = 6

# 速度参数
TANK_SPEED = 4
ROTATION_SPEED = 4
BULLET_SPEED = 5

# 游戏规则
MAX_BOUNCES = 2  # 子弹最大反弹次数
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
OBSERVATION_SIZE = 30

# 奖励参数 (Aggressive 修改版)
STEP_PENALTY = -0.005          # 降低每步惩罚，让它敢于多活一会儿来尝试策略
BULLET_HIT_AGENT_REWARD = -10.0
FRIENDLY_FIRE_PENALTY = -2.0   # 降低误伤惩罚，鼓励它多开火尝试
ENEMY_HIT_REWARD = 30.0        # 提高击杀奖励 (从20提至30)，增加诱惑力

# 新增：进攻性引导奖励
REWARD_AIMING = 0.05           # 每帧瞄准敌人的奖励 (最大值)
REWARD_APPROACH = 0.002        # 每帧接近敌人的奖励
REWARD_SHOOT = -0.05           # 开火惩罚 (防止无脑乱射，鼓励有把握再开枪)
REWARD_SURVIVAL = 0.001        # 存活奖励 (鼓励活着)

# 调试模式
DEBUG_RENDER_GRID = False  # 是否绘制网格
DEBUG_RENDER_PATH = True   # 是否绘制寻路路径
