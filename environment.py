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
    ENEMY_HIT_REWARD, FPS, DEBUG_RENDER_PATH, DEBUG_RENDER_GRID,
    LIGHT_GRAY, REWARD_AIMING, REWARD_APPROACH, REWARD_SHOOT, REWARD_SURVIVAL,
    VISION_DISTANCE, ANGLE_TOLERANCE
)
from sprites import Wall, Tank
from pathfinding import GridMap, BFSPathfinder
from bot_ai import BotAI


class TankTroubleEnv(gym.Env):
    """坦克大战 RL 环境"""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode=None):
        super(TankTroubleEnv, self).__init__()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Tank RL: Pathfinding Bot")
            self.clock = pygame.time.Clock()
        
        # 游戏对象
        self.all_sprites = None
        self.walls = None
        self.bullets = None
        self.tanks = None
        self.agent = None
        self.enemy = None
        
        # 寻路系统
        self.grid_map = GridMap()
        self.pathfinder = BFSPathfinder(self.grid_map)
        self.bot_ai = BotAI(self.grid_map, self.pathfinder)
        
        # 游戏状态
        self.steps = 0
        self.max_steps = MAX_STEPS_PER_EPISODE

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.all_sprites = pygame.sprite.Group()
        self.walls = self._create_walls()
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
        reward = STEP_PENALTY + REWARD_SURVIVAL  # 基础奖励 + 存活奖励
        terminated = False
        truncated = False
        
        # 记录行动前的距离（用于计算接近奖励）
        prev_dist = math.hypot(
            self.enemy.rect.centerx - self.agent.rect.centerx,
            self.enemy.rect.centery - self.agent.rect.centery
        )
        
        # 玩家行动
        self.agent.act(action, self.walls, self.bullets, self.all_sprites)
        self.agent.update_velocity()
        
        # Bot 行动（传入子弹信息用于躲避）
        bot_action = self.bot_ai.decide_action(
            self.enemy, self.agent, self.walls, self.steps, self.bullets
        )
        self.enemy.act(bot_action, self.walls, self.bullets, self.all_sprites)
        self.enemy.update_velocity()
        
        # ========== 进攻性奖励计算 ==========
        # 1. 接近敌人奖励
        curr_dist = math.hypot(
            self.enemy.rect.centerx - self.agent.rect.centerx,
            self.enemy.rect.centery - self.agent.rect.centery
        )
        if curr_dist < prev_dist:
            reward += REWARD_APPROACH * (prev_dist - curr_dist)
        
        # 2. 瞄准敌人奖励（计算agent朝向与敌人方向的角度差）
        dx = self.enemy.rect.centerx - self.agent.rect.centerx
        dy = self.enemy.rect.centery - self.agent.rect.centery
        target_angle = math.degrees(math.atan2(-dy, dx))
        angle_diff = abs(target_angle - self.agent.angle)
        while angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 角度差越小，瞄准奖励越高
        if angle_diff < 45:  # 在45度范围内给予奖励
            aiming_reward = REWARD_AIMING * (1 - angle_diff / 45)
            reward += aiming_reward
        
        # 3. 射击奖励/惩罚
        if action == 5:  # 射击动作
            reward += REWARD_SHOOT
            # 如果瞄准较准且距离合适，给予额外奖励
            if angle_diff < ANGLE_TOLERANCE and curr_dist < VISION_DISTANCE:
                reward += 0.1  # 精准射击奖励
        
        # 更新子弹
        self.bullets.update(self.walls)
        
        # 碰撞检测
        for bullet in self.bullets:
            hit_tanks = pygame.sprite.spritecollide(bullet, self.tanks, False)
            for tank in hit_tanks:
                bullet.kill()
                if tank.id == self.agent.id:
                    reward = BULLET_HIT_AGENT_REWARD
                    terminated = True
                    if bullet.owner_id == self.agent.id:
                        reward += FRIENDLY_FIRE_PENALTY
                elif tank.id == self.enemy.id:
                    if bullet.owner_id == self.agent.id:
                        reward = ENEMY_HIT_REWARD
                        terminated = True
        
        # 检查终止条件
        if self.steps >= self.max_steps:
            truncated = True
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """获取观测值"""
        def nx(x): return x / SCREEN_WIDTH
        def ny(y): return y / SCREEN_HEIGHT
        
        rad = math.radians(self.agent.angle)
        obs = [
            nx(self.agent.rect.centerx), ny(self.agent.rect.centery),
            math.sin(rad), math.cos(rad),
            nx(self.enemy.rect.centerx), ny(self.enemy.rect.centery),
            math.sin(math.radians(self.enemy.angle)),
            math.cos(math.radians(self.enemy.angle)),
            self.agent.vx / TANK_SIZE, self.agent.vy / TANK_SIZE,
            self.agent.cooldown / 20
        ]
        
        # 最近的 5 发子弹
        bullets = sorted(
            self.bullets,
            key=lambda b: math.hypot(
                b.rect.x - self.agent.rect.x, b.rect.y - self.agent.rect.y
            )
        )
        for i in range(5):
            if i < len(bullets):
                b = bullets[i]
                obs.extend([nx(b.rect.centerx), ny(b.rect.centery),
                           b.dx / TANK_SIZE, b.dy / TANK_SIZE])
            else:
                obs.extend([0, 0, 0, 0])
        
        while len(obs) < OBSERVATION_SIZE:
            obs.append(0)
        
        return np.array(obs[:OBSERVATION_SIZE], dtype=np.float32)

    def _create_walls(self):
        """创建随机墙壁（优化版，确保足够通行空间）"""
        walls = pygame.sprite.Group()
        
        # 边界墙（必须保留）
        border_thickness = 10
        walls.add(Wall(0, 0, SCREEN_WIDTH, border_thickness))  # 上
        walls.add(Wall(0, SCREEN_HEIGHT - border_thickness, SCREEN_WIDTH, border_thickness))  # 下
        walls.add(Wall(0, 0, border_thickness, SCREEN_HEIGHT))  # 左
        walls.add(Wall(SCREEN_WIDTH - border_thickness, 0, border_thickness, SCREEN_HEIGHT))  # 右
        
        # 随机生成内部墙壁（减少数量，缩短长度）
        num_walls = random.randint(2, 5)  # 减少墙壁数量
        
        wall_rects = []  # 用于检测墙壁之间的间距
        
        for _ in range(num_walls):
            for attempt in range(10):  # 尝试多次找到合适位置
                # 随机决定墙壁方向（水平或垂直）
                if random.random() < 0.5:
                    # 水平墙（缩短长度）
                    width = random.randint(60, 150)
                    height = 15
                else:
                    # 垂直墙（缩短长度）
                    width = 15
                    height = random.randint(60, 150)
                
                # 随机位置（更大的边缘margin）
                margin = 80
                x = random.randint(margin, SCREEN_WIDTH - margin - width)
                y = random.randint(margin, SCREEN_HEIGHT - margin - height)
                
                new_rect = pygame.Rect(x, y, width, height)
                
                # 检查与现有墙壁的间距（至少50像素）
                too_close = False
                for existing in wall_rects:
                    # 扩展检测区域
                    expanded = existing.inflate(50, 50)
                    if expanded.colliderect(new_rect):
                        too_close = True
                        break
                
                if not too_close:
                    walls.add(Wall(x, y, width, height))
                    wall_rects.append(new_rect)
                    break
        
        return walls

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
