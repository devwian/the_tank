"""
RL 环境模块
实现 Gymnasium 环境接口
"""

import pygame
import math
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, TANK_SIZE, GRID_SIZE,
    WHITE, MAX_STEPS_PER_EPISODE, OBSERVATION_SIZE,
    STEP_PENALTY, BULLET_HIT_AGENT_REWARD, FRIENDLY_FIRE_PENALTY,
    ENEMY_HIT_REWARD, TIMEOUT_PENALTY, FPS, DEBUG_RENDER_PATH, DEBUG_RENDER_GRID,
    LIGHT_GRAY, REWARD_SHOOT, COLLISION_PENALTY, REWARD_ACCURATE_SHOT,
    VISION_DISTANCE, REWARD_FORWARD_MOVE, TANK_SPEED, BULLET_COOLDOWN, BULLET_SPEED,
    IDLE_PENALTY, REWARD_SURVIVAL
)
from sprites import Wall, Tank
from pathfinding import GridMap, AStarPathfinder
from bot_ai import BotAI


class TankTroubleEnv(gym.Env):
    """坦克大战 RL 环境"""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    # 动作名称映射
    ACTION_NAMES = {
        0: "待命",
        1: "前进",
        2: "后退",
        3: "顺时针",
        4: "逆时针",
        5: "射击"
    }
    
    def __init__(self, render_mode=None, debug_mode=False, difficulty=1):
        """
        初始化环境
        
        Args:
            render_mode: 渲染模式
            debug_mode: 调试模式
            difficulty: 难度级别 (1=无墙无Bot行动, 2=有墙Bot移动不攻击, 3=完整版)
        """
        super(TankTroubleEnv, self).__init__()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.debug_mode = debug_mode  # 调试模式
        self.difficulty = difficulty  # 难度级别
        self.screen = None
        self.clock = None
        
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Tank Trouble Hunter RL Environment")
            self.clock = pygame.time.Clock()
        
        # 游戏对象
        self.all_sprites = None
        self.walls = None
        self.bullets = None
        self.tanks = None
        self.agent = None
        self.enemy = None
        
        # 寻路系统（使用 A* 算法）
        self.grid_map = GridMap()
        self.pathfinder = AStarPathfinder(self.grid_map)
        self.bot_ai = BotAI(self.grid_map, self.pathfinder)
        
        # 游戏状态
        self.steps = 0
        self.max_steps = MAX_STEPS_PER_EPISODE
        
        # 动作历史记录（防止震荡）
        self.action_history = []
        self.max_history = 5

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.all_sprites = pygame.sprite.Group()
        # 难度 1: 无内部墙壁; 难度 2,3: 有内部墙壁
        self.walls = self._create_walls(no_internal_walls=(self.difficulty == 1))
        self.bullets = pygame.sprite.Group()
        self.tanks = pygame.sprite.Group()
        
        # 初始化网格地图
        self.grid_map.init_from_walls(self.walls)
        
        # 随机生成玩家位置
        self.agent = self._spawn_tank_random((200, 0, 0), tank_id=1)
        
        # 随机生成敌人位置（确保与玩家有足够距离）
        self.enemy = self._spawn_tank_random((0, 0, 200), tank_id=2, min_dist_from=self.agent, min_dist=150)
        
        self.all_sprites.add(self.walls)
        self.all_sprites.add(self.agent)
        self.all_sprites.add(self.enemy)
        self.tanks.add(self.agent)
        self.tanks.add(self.enemy)
        
        self.steps = 0
        self.bot_ai.current_path = []
        self.stuck_steps = 0
        
        # 重置动作历史
        self.action_history = []
        
        return self._get_obs(), {}
    
    def _spawn_tank_random(self, color, tank_id, min_dist_from=None, min_dist=100):
        """
        随机生成坦克位置
        color: 坦克颜色
        tank_id: 坦克ID
        min_dist_from: 需要与该坦克保持距离（可选）
        min_dist: 最小距离
        """
        margin = TANK_SIZE * 2  # 边缘留白
        max_attempts = 100
        
        for _ in range(max_attempts):
            # 随机位置（避开边缘）
            x = random.randint(margin, SCREEN_WIDTH - margin)
            y = random.randint(margin, SCREEN_HEIGHT - margin)
            
            # 创建临时坦克检测碰撞
            dummy = Tank(x, y, color, tank_id)
            
            # 检查墙壁碰撞
            if pygame.sprite.spritecollide(dummy, self.walls, False):
                continue
            
            # 检查网格是否可行走
            gx, gy = self.grid_map.pixel_to_grid(x, y)
            if not self.grid_map.is_walkable(gx, gy):
                continue
            
            # 检查与其他坦克的距离
            if min_dist_from is not None:
                dist = math.hypot(
                    x - min_dist_from.rect.centerx,
                    y - min_dist_from.rect.centery
                )
                if dist < min_dist:
                    continue
            
            # 随机初始角度
            dummy.angle = random.randint(0, 359)
            dummy.rotate()
            
            return dummy
        
        # 如果随机失败，使用默认位置
        fallback_x = margin if tank_id == 1 else SCREEN_WIDTH - margin
        fallback_y = margin if tank_id == 1 else SCREEN_HEIGHT - margin
        tank = Tank(fallback_x, fallback_y, color, tank_id)
        tank.angle = random.randint(0, 359)
        tank.rotate()
        return tank

    def step(self, action):
        """执行一步"""
        self.steps += 1
        reward = STEP_PENALTY  # 基础奖励（现在为0）
        terminated = False
        truncated = False
    
        # 记录行动前的距离（用于计算接近奖励）
        old_dist = math.hypot(
            self.agent.rect.centerx - self.enemy.rect.centerx,
            self.agent.rect.centery - self.enemy.rect.centery
        )
        
        # 玩家行动
        old_pos = (self.agent.rect.centerx, self.agent.rect.centery)
        self.agent.act(action, self.walls, self.bullets, self.all_sprites, other_tanks=self.enemy)
        self.agent.update_velocity()
        new_pos = (self.agent.rect.centerx, self.agent.rect.centery)
        
        # 检查是否撞墙（位置没变但尝试移动了）
        if action in [1, 2] and old_pos == new_pos:
            reward += COLLISION_PENALTY
        
        # 检查是否长时间卡住（位置几乎没变）- 简化逻辑
        dist_moved = math.hypot(new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
        if dist_moved < 0.5:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
            
        # 长时间不动给予轻微惩罚
        if self.stuck_steps > 30:
            reward -= 0.01
        
        # 简化动作历史记录
        action_int = int(action)
        self.action_history.append(action_int)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
        
        # 检测严重震荡（连续4步只有两种动作且交替出现）
        if len(self.action_history) >= 4:
            recent = self.action_history[-4:]
            if len(set(recent)) == 2 and (set(recent) == {3, 4} or set(recent) == {1, 2}):
                reward -= 0.1  # 大幅增加惩罚
        
        # 待机惩罚
        if action == 0:
            reward += IDLE_PENALTY
            
        # 计算接近敌人的奖励（轻微引导）
        new_dist = math.hypot(
            self.agent.rect.centerx - self.enemy.rect.centerx,
            self.agent.rect.centery - self.enemy.rect.centery
        )
        approach_reward = (old_dist - new_dist) * 0.01
        reward += approach_reward
        
        # 朝向敌人的奖励（鼓励瞄准）
        agent_pos = self.agent.rect.center
        enemy_pos = self.enemy.rect.center
        dx = enemy_pos[0] - agent_pos[0]
        dy = enemy_pos[1] - agent_pos[1]
        target_angle = math.degrees(math.atan2(-dy, dx))
        
        # 规范化角度到 [-180, 180]
        self.agent.angle = (self.agent.angle + 180) % 360 - 180
        self.enemy.angle = (self.enemy.angle + 180) % 360 - 180
        
        # 计算最小角度差
        angle_diff = (target_angle - self.agent.angle + 180) % 360 - 180
        angle_diff_abs = abs(angle_diff)
        
        # 取消持续朝向奖励，防止智能体只转不打
        # pointing_reward = (1.0 - (angle_diff_abs / 180.0)) * 0.002
        # reward += pointing_reward
            
        # 射击动作奖励
        if action == 5:
            reward += REWARD_SHOOT
            # 只在射击时给予瞄准奖励，鼓励精准射击
            has_los = not self._raycast_hit_wall(agent_pos, enemy_pos)
            if angle_diff_abs < 20 and has_los:
                reward += REWARD_ACCURATE_SHOT
        
        bot_action = 0  # 默认待命
        # Bot 行动（暂时关闭）
        """
        if self.difficulty == 1:
            # 难度1: Bot 完全不动
            bot_action = 0  # 待命
        elif self.difficulty == 2:
            # 难度2: Bot 只移动不攻击
            bot_action = self.bot_ai.decide_action(
                self.enemy, self.agent, self.walls, self.steps, self.bullets,
                can_attack=False
            )
        else:
            # 难度3: Bot 完整行为
            bot_action = self.bot_ai.decide_action(
                self.enemy, self.agent, self.walls, self.steps, self.bullets
            )
        """
        self.enemy.act(bot_action, self.walls, self.bullets, self.all_sprites, other_tanks=self.agent)
        self.enemy.update_velocity()
        
        # 调试日志：记录双方行动
        if self.debug_mode:
            agent_action_name = self.ACTION_NAMES.get(int(action), "未知")
            bot_action_name = self.ACTION_NAMES.get(int(bot_action), "未知")
            print(f"[Step {self.steps:4d}] Agent: {agent_action_name:4s} | Bot: {bot_action_name:4s} | "
                  f"Agent位置:({self.agent.rect.centerx:3d},{self.agent.rect.centery:3d}) | "
                  f"Bot位置:({self.enemy.rect.centerx:3d},{self.enemy.rect.centery:3d})|")
        
        # 更新子弹
        self.bullets.update(self.walls)
        
        # 结果状态: "win"=胜利, "lose"=失败, "timeout"=超时, None=未结束
        result = None
        
        # 碰撞检测
        for bullet in self.bullets:
            hit_tanks = pygame.sprite.spritecollide(bullet, self.tanks, False)
            for tank in hit_tanks:
                # 跳过安全帧内的发射者（防止刚发射就击中自己）
                if bullet.safe_frames > 0 and bullet.owner_id == tank.id:
                    continue
                    
                bullet.kill()
                if tank.id == self.agent.id:
                    # 玩家被击中 -> 失败
                    reward = BULLET_HIT_AGENT_REWARD
                    terminated = True
                    result = "lose"
                    if bullet.owner_id == self.agent.id:
                        reward += FRIENDLY_FIRE_PENALTY
                        if self.debug_mode:
                            print(f"\n💀 [Step {self.steps}] Agent 自杀！被自己的子弹击中")
                    else:
                        if self.debug_mode:
                            print(f"\n💀 [Step {self.steps}] Agent 被 Bot 的子弹击中！")
                                    
                elif tank.id == self.enemy.id:
                    # Bot被击中 -> 胜利
                    terminated = True
                    result = "win"
                    if bullet.owner_id == self.agent.id:
                        # 玩家击中Bot，玩家得分
                        reward = ENEMY_HIT_REWARD
                        if self.debug_mode:
                            print(f"\n🎯 [Step {self.steps}] Bot 被 Agent 的子弹击中！")
                    else:
                        # Bot自杀，玩家也得分
                        reward = ENEMY_HIT_REWARD
                        if self.debug_mode:
                            print(f"\n💀 [Step {self.steps}] Bot 自杀！被自己的子弹击中")
        
        # 检查终止条件
        if self.steps >= self.max_steps:
            truncated = True
            # 超时惩罚（只在未终止时追加）
            if not terminated:
                reward += TIMEOUT_PENALTY
                result = "timeout"

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {"result": result}

    def _get_obs(self):
        """获取观测值 (64维)"""
        def nx(x): return x / SCREEN_WIDTH
        def ny(y): return y / SCREEN_HEIGHT
        
        rad = math.radians(self.agent.angle)
        
        # 计算与敌人的相对信息
        agent_pos = self.agent.rect.center
        enemy_pos = self.enemy.rect.center
        dx = enemy_pos[0] - agent_pos[0]
        dy = enemy_pos[1] - agent_pos[1]
        dist = math.hypot(dx, dy)
        target_angle = math.degrees(math.atan2(-dy, dx))
        
        # 相对角度差 (归一化到 [-1, 1])
        angle_diff = (target_angle - self.agent.angle + 180) % 360 - 180
        rel_angle = angle_diff / 180.0
        
        # 是否有视线 (Line of Sight)
        has_los = 1.0 if not self._raycast_hit_wall(agent_pos, enemy_pos) else 0.0
        
        # 基础信息 (16维)
        obs = [
            # 1. 自身位置 (2)
            nx(self.agent.rect.centerx), ny(self.agent.rect.centery),
            # 2. 自身朝向 (2)
            math.sin(rad), math.cos(rad),
            # 3. 自身速度 (2)
            self.agent.vx / TANK_SPEED, self.agent.vy / TANK_SPEED,
            # 4. 自身冷却 (1)
            self.agent.cooldown / BULLET_COOLDOWN,
            
            # 5. 敌人位置 (2)
            nx(self.enemy.rect.centerx), ny(self.enemy.rect.centery),
            # 6. 敌人朝向 (2)
            math.sin(math.radians(self.enemy.angle)),
            math.cos(math.radians(self.enemy.angle)),
            # 7. 敌人速度 (2)
            self.enemy.vx / TANK_SPEED, self.enemy.vy / TANK_SPEED,
            
            # 8. 相对信息 (3)
            rel_angle,
            dist / math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT),
            has_los
        ]
        
        # 子弹信息 (40维)
        bullets = sorted(
            self.bullets,
            key=lambda b: math.hypot(
                b.rect.centerx - self.agent.rect.centerx, 
                b.rect.centery - self.agent.rect.centery
            )
        )
        
        max_bullets = 10
        for i in range(max_bullets):
            if i < len(bullets):
                b = bullets[i]
                obs.extend([
                    nx(b.rect.centerx), 
                    ny(b.rect.centery),
                    b.dx / BULLET_SPEED, 
                    b.dy / BULLET_SPEED
                ])
            else:
                obs.extend([0, 0, 0, 0])
        
        # 射线检测墙壁距离 (8维) - 8个方向，每45度一个
        # 方向: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        ray_distances = self._cast_rays()
        obs.extend(ray_distances)
        
        # 确保长度正确
        if len(obs) != OBSERVATION_SIZE:
            if len(obs) < OBSERVATION_SIZE:
                obs.extend([0] * (OBSERVATION_SIZE - len(obs)))
            else:
                obs = obs[:OBSERVATION_SIZE]
        
        return np.array(obs, dtype=np.float32)
    
    def _cast_rays(self):
        """发射射线检测墙壁距离"""
        cx = self.agent.rect.centerx
        cy = self.agent.rect.centery
        max_dist = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)  # 最大检测距离
        
        ray_distances = []
        # 8个方向: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        for angle_offset in range(0, 360, 45):
            angle = math.radians(angle_offset)
            dx = math.cos(angle)
            dy = -math.sin(angle)  # pygame的y轴向下
            
            # 沿射线方向检测墙壁
            min_dist = max_dist
            step = 5  # 检测步长
            for d in range(step, int(max_dist), step):
                x = int(cx + dx * d)
                y = int(cy + dy * d)
                
                # 检查是否出界或碰到墙壁
                if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
                    min_dist = d
                    break
                
                # 检查是否碰到墙壁
                for wall in self.walls:
                    if wall.rect.collidepoint(x, y):
                        min_dist = d
                        break
                else:
                    continue
                break
            
            # 归一化到[0, 1]
            ray_distances.append(min_dist / max_dist)
        
        return ray_distances

    def _create_walls(self, no_internal_walls=False):
        """创建随机墙壁（优化版，确保足够通行空间）
        
        Args:
            no_internal_walls: 如果为True，只创建边界墙，不创建内部墙壁
        """
        walls = pygame.sprite.Group()
        
        # 边界墙（必须保留）
        border_thickness = 10
        walls.add(Wall(0, 0, SCREEN_WIDTH, border_thickness))  # 上
        walls.add(Wall(0, SCREEN_HEIGHT - border_thickness, SCREEN_WIDTH, border_thickness))  # 下
        walls.add(Wall(0, 0, border_thickness, SCREEN_HEIGHT))  # 左
        walls.add(Wall(SCREEN_WIDTH - border_thickness, 0, border_thickness, SCREEN_HEIGHT))  # 右
        
        # 如果不生成内部墙壁，直接返回
        if True: # 暂时关闭所有内部墙体生成
            return walls
        
        # 固定内部墙壁（暂时取消随机生成）
        fixed_walls = [
            (150, 150, 15, 100),
            (435, 150, 15, 100),
            (150, 350, 15, 100),
            (435, 350, 15, 100),
            (250, 292, 100, 15)
        ]
        for x, y, w, h in fixed_walls:
            walls.add(Wall(x, y, w, h))
        
        return walls

    def _raycast_hit_wall(self, start, end):
        """简单的射线墙壁检测 - 检查start到end的直线是否被墙壁阻挡"""
        line = (start, end)
        for wall in self.walls:
            if wall.rect.clipline(line):
                return True
        return False

    def _render_frame(self):
        """渲染一帧"""
        if self.screen is None:
            return
        
        self.screen.fill(WHITE)
        self.all_sprites.draw(self.screen)
        
        # 调试：绘制路径
        if DEBUG_RENDER_PATH and self.bot_ai.current_path:
            pts = [self.grid_map.grid_to_pixel(*p) for p in self.bot_ai.current_path]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (0, 255, 0), False, pts, 2)
        
        # 调试：绘制网格缓冲区
        if DEBUG_RENDER_GRID and self.grid_map.grid_map is not None:
            for x in range(self.grid_map.grid_cols):
                for y in range(self.grid_map.grid_rows):
                    if self.grid_map.grid_map[x][y] == 1:
                        r = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                        pygame.draw.rect(self.screen, LIGHT_GRAY, r, 1)
        
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        """关闭环境"""
        if self.screen:
            pygame.quit()
