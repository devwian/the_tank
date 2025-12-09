import pygame
import math
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# --- 常量定义 ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
TANK_SIZE = 30
# 适当降低速度以配合精细操作，或者保持高速但增加寻路频率
TANK_SPEED = 4 
ROTATION_SPEED = 4
BULLET_SPEED = 5
MAX_BOUNCES = 5
BULLET_COOLDOWN = 20 
FPS = 60

# 网格设置
# GRID_SIZE 越小，路径越平滑，但计算量越大
# 坦克 30px，设为 15px 可以实现比较精细的控制
GRID_SIZE = 15 

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# --- 基础类保持不变 ---
class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(GRAY)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, owner_id):
        super().__init__()
        self.image = pygame.Surface([6, 6])
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.speed = BULLET_SPEED
        rad = math.radians(angle)
        self.dx = math.cos(rad) * self.speed
        self.dy = -math.sin(rad) * self.speed
        self.bounces = 0
        self.max_bounces = MAX_BOUNCES
        self.owner_id = owner_id

    def update(self, walls):
        self.rect.x += self.dx
        hit_list = pygame.sprite.spritecollide(self, walls, False)
        if hit_list:
            self.dx *= -1
            self.bounces += 1
            self.rect.x += self.dx 
        self.rect.y += self.dy
        hit_list = pygame.sprite.spritecollide(self, walls, False)
        if hit_list:
            self.dy *= -1
            self.bounces += 1
            self.rect.y += self.dy
        if (self.rect.x < 0 or self.rect.x > SCREEN_WIDTH or 
            self.rect.y < 0 or self.rect.y > SCREEN_HEIGHT or 
            self.bounces > self.max_bounces):
            self.kill()

class Tank(pygame.sprite.Sprite):
    def __init__(self, x, y, color, id):
        super().__init__()
        self.original_image = pygame.Surface([TANK_SIZE, TANK_SIZE], pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, color, (0, 0, TANK_SIZE, TANK_SIZE))
        # 炮塔指示
        pygame.draw.rect(self.original_image, BLACK, (TANK_SIZE - 8, TANK_SIZE//2 - 4, 8, 8))
        self.image = self.original_image
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.id = id
        self.angle = 0 
        self.speed = TANK_SPEED
        self.cooldown = 0
        self.old_rect = self.rect.copy()
        
        # 速度记录用于预判
        self.vx = 0
        self.vy = 0
        self.last_pos = (x, y)

    def update_velocity(self):
        self.vx = self.rect.centerx - self.last_pos[0]
        self.vy = self.rect.centery - self.last_pos[1]
        self.last_pos = (self.rect.centerx, self.rect.centery)

    def act(self, action, walls, bullets_group, all_sprites):
        self.old_rect = self.rect.copy()
        rotated = False
        moved = False
        
        if action == 3: self.angle += ROTATION_SPEED; rotated = True
        elif action == 4: self.angle -= ROTATION_SPEED; rotated = True
            
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.speed
        dy = -math.sin(rad) * self.speed
        
        if action == 1: self.rect.x += dx; self.rect.y += dy; moved = True
        elif action == 2: self.rect.x -= dx; self.rect.y -= dy; moved = True
            
        if rotated: self.rotate()
        if moved and pygame.sprite.spritecollide(self, walls, False): self.rect = self.old_rect
            
        if self.cooldown > 0: self.cooldown -= 1
        if action == 5 and self.cooldown == 0:
            self.shoot(bullets_group, all_sprites)
            return True
        return False

    def rotate(self):
        old_center = self.rect.center
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = old_center

    def shoot(self, bullets_group, all_sprites):
        rad = math.radians(self.angle)
        bx = self.rect.centerx + math.cos(rad) * (TANK_SIZE/1.5)
        by = self.rect.centery - math.sin(rad) * (TANK_SIZE/1.5)
        bullet = Bullet(bx, by, self.angle, self.id)
        bullets_group.add(bullet)
        all_sprites.add(bullet)
        self.cooldown = BULLET_COOLDOWN

# --- RL 环境 (核心修改部分) ---

class TankTroubleEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode=None):
        super(TankTroubleEnv, self).__init__()
        self.action_space = spaces.Discrete(6)
        # 增加观测: Agent(4) + AgentVel(2) + Enemy(4) + Bullets(5*4) = 30
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(30,), dtype=np.float32)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Tank RL: Pathfinding Bot (Safety Buffer)")
            self.clock = pygame.time.Clock()

        # 寻路变量
        self.grid_cols = SCREEN_WIDTH // GRID_SIZE
        self.grid_rows = SCREEN_HEIGHT // GRID_SIZE
        self.grid_map = None
        self.current_path = []
        self.last_path_calc = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.all_sprites = pygame.sprite.Group()
        self.walls = self._create_walls() # 创建墙壁
        self.bullets = pygame.sprite.Group()
        self.tanks = pygame.sprite.Group()
        
        # 初始化网格 (包含膨胀层)
        self._init_grid_map_with_buffer()
        
        self.agent = Tank(80, 80, RED, id=1)
        
        # 随机生成敌人位置，但要避开墙和缓冲区
        while True:
            ex, ey = random.randint(100, 500), random.randint(100, 500)
            # 简单的检查：是否与墙重叠
            dummy = Tank(ex, ey, BLUE, id=2)
            if not pygame.sprite.spritecollide(dummy, self.walls, False):
                # 更严格的检查：是否在安全区内 (避免Bot出生就卡住)
                gx, gy = int(ex // GRID_SIZE), int(ey // GRID_SIZE)
                if self.grid_map[gx][gy] == 0:
                    self.enemy = dummy
                    break

        self.all_sprites.add(self.walls)
        self.all_sprites.add(self.agent)
        self.all_sprites.add(self.enemy)
        self.tanks.add(self.agent)
        self.tanks.add(self.enemy)
        
        self.steps = 0
        self.max_steps = 1500
        self.current_path = []
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward = -0.01
        terminated = False
        truncated = False
        
        # Agent
        self.agent.act(action, self.walls, self.bullets, self.all_sprites)
        self.agent.update_velocity()
        
        # Bot (寻路 + 攻击)
        bot_action = self._smart_pathfinding_bot(self.enemy, self.agent)
        self.enemy.act(bot_action, self.walls, self.bullets, self.all_sprites)
        self.enemy.update_velocity()
        
        # Update Bullets
        self.bullets.update(self.walls)
        
        # Collision Logic
        for bullet in self.bullets:
            hit_tanks = pygame.sprite.spritecollide(bullet, self.tanks, False)
            for tank in hit_tanks:
                bullet.kill()
                if tank.id == self.agent.id:
                    reward = -10.0
                    terminated = True
                    if bullet.owner_id == self.agent.id: reward -= 5.0
                elif tank.id == self.enemy.id:
                    if bullet.owner_id == self.agent.id: reward = 10.0
                    terminated = True

        if self.steps >= self.max_steps: truncated = True
        if self.render_mode == "human": self._render_frame()
        return self._get_obs(), reward, terminated, truncated, {}

    # --- 核心：网格初始化与膨胀 ---
    def _init_grid_map_with_buffer(self):
        """
        1. 标记墙壁 (Val=1)
        2. 标记墙壁周围一圈 (Val=1, Padding)
        这样 Bot 寻路时就会自然避开墙壁边缘，不再卡角
        """
        self.grid_map = np.zeros((self.grid_cols, self.grid_rows), dtype=int)
        
        # 第一步：标记真实的墙壁
        temp_map = np.zeros_like(self.grid_map)
        for wall in self.walls:
            # 获取墙壁覆盖的 grid
            # 使用 inflate 稍微缩小一点 rect，避免刚好压线的判定
            # 但为了安全，我们通常宁可多判不可少判
            x_start = max(0, wall.rect.x // GRID_SIZE)
            x_end = min(self.grid_cols, (wall.rect.right) // GRID_SIZE + 1)
            y_start = max(0, wall.rect.y // GRID_SIZE)
            y_end = min(self.grid_rows, (wall.rect.bottom) // GRID_SIZE + 1)
            
            temp_map[x_start:x_end, y_start:y_end] = 1
            
        # 第二步：应用膨胀 (Padding)
        # 遍历所有格子，如果是墙，把它的 8 邻域都标记为不可走
        # 这样 Tank 中心点就不会靠近墙壁 1 个 Grid 的距离 (约15-20px)
        self.grid_map = temp_map.copy()
        for x in range(self.grid_cols):
            for y in range(self.grid_rows):
                if temp_map[x][y] == 1:
                    # 标记周围
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                                self.grid_map[nx][ny] = 1

    # --- 核心：BFS 寻路 ---
    def _bfs_path(self, start, end):
        # start, end 都是 grid 坐标 (gx, gy)
        # 如果终点在墙里(或者保护区里)，尝试找终点最近的可走点
        if self.grid_map[end[0]][end[1]] == 1:
            # 简单的螺旋搜索找到最近的空地
            found = False
            for r in range(1, 5): # 搜索半径
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        nx, ny = end[0]+dx, end[1]+dy
                        if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                            if self.grid_map[nx][ny] == 0:
                                end = (nx, ny)
                                found = True
                                break
                    if found: break
                if found: break
        
        # BFS
        queue = deque([start])
        came_from = {start: None}
        
        while queue:
            current = queue.popleft()
            if current == end:
                break
            
            cx, cy = current
            # 4方向搜索 (不建议8方向，容易穿墙角)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                    if self.grid_map[nx][ny] == 0 and (nx, ny) not in came_from:
                        queue.append((nx, ny))
                        came_from[(nx, ny)] = current
        
        # 重建路径
        if end not in came_from:
            return [] # 无路
        
        path = []
        curr = end
        while curr != start:
            path.append(curr)
            curr = came_from[curr]
            if curr is None: break # Should not happen
        path.reverse()
        return path

    # --- 核心：Bot 逻辑 ---
    def _smart_pathfinding_bot(self, bot, target):
        # 1. 基础计算
        target_pos = target.rect.center
        bot_pos = bot.rect.center
        
        # 计算预判位置 (简单的提前量)
        dist = math.hypot(target_pos[0] - bot_pos[0], target_pos[1] - bot_pos[1])
        time_hit = dist / BULLET_SPEED
        pred_x = target_pos[0] + target.vx * time_hit
        pred_y = target_pos[1] + target.vy * time_hit
        
        # 2. 视线检测 (Priority 1)
        # 如果能直接看到敌人(或预判点)，且距离适中，直接进入战斗模式，放弃寻路
        has_los = self._check_line_of_sight(bot_pos, (pred_x, pred_y), self.walls)
        
        angle_to_target = math.degrees(math.atan2(-(pred_y - bot_pos[1]), pred_x - bot_pos[0]))
        angle_diff = self._norm_angle(angle_to_target - bot.angle)
        
        if has_los:
            # 清空路径，专注于战斗
            self.current_path = []
            
            # 瞄准
            if abs(angle_diff) < 10:
                if bot.cooldown == 0: return 5 # Shoot
                # 冷却中，可以尝试稍微移动闪避，或者原地不动
                return 0
            else:
                return 3 if angle_diff > 0 else 4
        
        # 3. 寻路模式 (Priority 2)
        # 每 10 帧更新一次路径 (频率太高浪费性能，太低反应慢)
        gx_bot = int(bot_pos[0] // GRID_SIZE)
        gy_bot = int(bot_pos[1] // GRID_SIZE)
        gx_target = int(target_pos[0] // GRID_SIZE)
        gy_target = int(target_pos[1] // GRID_SIZE)
        
        if self.steps % 10 == 0 or not self.current_path:
            self.current_path = self._bfs_path((gx_bot, gy_bot), (gx_target, gy_target))
            
        # 沿路径移动
        if self.current_path:
            # 获取下一个路点
            # 如果离第一个点非常近，就去第二个点
            next_grid = self.current_path[0]
            next_pixel = (next_grid[0] * GRID_SIZE + GRID_SIZE//2, 
                          next_grid[1] * GRID_SIZE + GRID_SIZE//2)
            
            dist_to_node = math.hypot(next_pixel[0] - bot_pos[0], next_pixel[1] - bot_pos[1])
            
            if dist_to_node < 15: # 到达该节点
                self.current_path.pop(0)
                if self.current_path:
                    next_grid = self.current_path[0]
                    next_pixel = (next_grid[0] * GRID_SIZE + GRID_SIZE//2, 
                                  next_grid[1] * GRID_SIZE + GRID_SIZE//2)
                else:
                    return 0 # 路径走完了
            
            # 导航向 next_pixel
            move_angle = math.degrees(math.atan2(-(next_pixel[1] - bot_pos[1]), next_pixel[0] - bot_pos[0]))
            move_diff = self._norm_angle(move_angle - bot.angle)
            
            # 移动决策
            if abs(move_diff) > 15:
                return 3 if move_diff > 0 else 4
            else:
                return 1 # Forward
                
        return 0

    def _norm_angle(self, a):
        a = a % 360
        if a > 180: a -= 360
        return a

    def _check_line_of_sight(self, p1, p2, walls):
        return not any(w.rect.clipline(p1, p2) for w in walls)

    def _get_obs(self):
        # 保持原来的 Observation 结构
        # ... (省略部分代码以节省篇幅，逻辑同上一版) ...
        # 这里简单返回一个全零数组作为占位，实际请复制上一版 _get_obs 代码
        def nx(x): return x / SCREEN_WIDTH
        def ny(y): return y / SCREEN_HEIGHT
        rad = math.radians(self.agent.angle)
        obs = [
            nx(self.agent.rect.centerx), ny(self.agent.rect.centery),
            math.sin(rad), math.cos(rad),
            nx(self.enemy.rect.centerx), ny(self.enemy.rect.centery),
            math.sin(math.radians(self.enemy.angle)), math.cos(math.radians(self.enemy.angle)),
            self.agent.vx / TANK_SPEED, self.agent.vy / TANK_SPEED,
            self.agent.cooldown / BULLET_COOLDOWN
        ]
        # Bullets (5 closest)
        bullets = sorted(self.bullets, key=lambda b: math.hypot(b.rect.x-self.agent.rect.x, b.rect.y-self.agent.rect.y))
        for i in range(5):
            if i < len(bullets):
                b = bullets[i]
                obs.extend([nx(b.rect.centerx), ny(b.rect.centery), b.dx/BULLET_SPEED, b.dy/BULLET_SPEED])
            else:
                obs.extend([0,0,0,0])
        while len(obs) < 30: obs.append(0)
        return np.array(obs[:30], dtype=np.float32)

    def _create_walls(self):
        walls = pygame.sprite.Group()
        # 边界
        walls.add(Wall(0, 0, SCREEN_WIDTH, 10)) 
        walls.add(Wall(0, SCREEN_HEIGHT-10, SCREEN_WIDTH, 10)) 
        walls.add(Wall(0, 0, 10, SCREEN_HEIGHT)) 
        walls.add(Wall(SCREEN_WIDTH-10, 0, 10, SCREEN_HEIGHT))
        # 迷宫地形
        walls.add(Wall(100, 100, 20, 200))
        walls.add(Wall(100, 300, 200, 20))
        walls.add(Wall(400, 100, 20, 250))
        walls.add(Wall(300, 450, 150, 20))
        walls.add(Wall(150, 450, 20, 100))
        return walls

    def _render_frame(self):
        if self.screen is None: return
        self.screen.fill(WHITE)
        self.all_sprites.draw(self.screen)
        
        # 调试：绘制路径
        if self.current_path:
            pts = [(p[0]*GRID_SIZE+GRID_SIZE//2, p[1]*GRID_SIZE+GRID_SIZE//2) for p in self.current_path]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, GREEN, False, pts, 2)
                
        # 调试：绘制网格缓冲区（可选，用于查看安全区）
        # if self.grid_map is not None:
        #     for x in range(self.grid_cols):
        #         for y in range(self.grid_rows):
        #             if self.grid_map[x][y] == 1:
        #                 r = pygame.Rect(x*GRID_SIZE, y*GRID_SIZE, GRID_SIZE, GRID_SIZE)
        #                 pygame.draw.rect(self.screen, (200, 200, 200), r, 1)

        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])
    
    def close(self):
        if self.screen: pygame.quit()

# --- 测试代码 ---
if __name__ == "__main__":
    env = TankTroubleEnv(render_mode="human")
    print("Safety Buffer Bot Ready.")
    
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample() 
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            for event in pygame.event.get():
                if event.type == pygame.QUIT: env.close(); exit()
    env.close()