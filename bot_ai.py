"""
机器人 AI 模块 (A* + DWA版)
实现 Bot 的智能决策逻辑
使用 A* 进行全局路径规划，DWA 进行局部动态避障
"""

import math
import random
import pygame
from constants import (
    BULLET_SPEED, ANGLE_TOLERANCE, NODE_ARRIVAL_DISTANCE,
    VISION_DISTANCE, TANK_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT,
    TANK_SPEED, MAX_BOUNCES, ROTATION_SPEED, GRID_SIZE
)

# ============ AI 参数 ============
DODGE_RADIUS = 180           # 躲避检测半径
PREDICT_FRAMES = 15          # 射击预判帧数
STUCK_THRESHOLD = 5          # 判定卡死的像素阈值
STUCK_CHECK_FRAMES = 30      # 多少帧检查一次卡死

# ============ DWA 参数 ============
DWA_PREDICT_TIME = 1.0       # DWA 预测时间（秒）
DWA_TIME_STEP = 0.1          # DWA 仿真时间步长
DWA_HEADING_WEIGHT = 1.0     # 目标方向权重
DWA_DISTANCE_WEIGHT = 0.2    # 距离权重
DWA_VELOCITY_WEIGHT = 0.1    # 速度权重
DWA_OBSTACLE_WEIGHT = 2.0    # 障碍物惩罚权重
DWA_PATH_WEIGHT = 1.5        # A*路径跟随权重

# ============ 路径跟随参数 ============
PURE_PURSUIT_LOOKAHEAD = 60  # Pure Pursuit 前瞻距离（像素）
ANGLE_TOLERANCE_MOVE = 25    # 移动时的角度容差（度）- 增大以减少震荡
ANGLE_TOLERANCE_TURN = 8     # 纯转向时的角度容差（度）
SMOOTH_TURN_THRESHOLD = 45   # 小于此角度差时边走边转


class DWAPlanner:
    """
    Dynamic Window Approach (动态窗口法) 局部规划器
    适配坦克的离散动作空间
    """
    
    def __init__(self, grid_map):
        self.grid_map = grid_map
        # 坦克动作: 0=待命, 1=前进, 2=后退, 3=顺时针, 4=逆时针
        self.motion_primitives = self._generate_motion_primitives()
    
    def _generate_motion_primitives(self):
        """
        生成运动基元 - 坦克可能的动作序列组合
        每个基元是一个动作序列，模拟几帧后的状态
        增加了交替转向+前进的基元，模拟边走边转
        """
        primitives = []
        # 基本动作组合：(动作, 持续帧数)
        
        # === 直行系列 ===
        primitives.append([(1, 6)])  # 直行
        
        # === 边走边转系列（交替执行转向和前进）===
        # 小角度调整 - 适合轻微方向修正
        primitives.append([(3, 1), (1, 2), (3, 1), (1, 2)])  # 边走边右转（小）
        primitives.append([(4, 1), (1, 2), (4, 1), (1, 2)])  # 边走边左转（小）
        
        # 中等角度调整
        primitives.append([(3, 2), (1, 2), (3, 2), (1, 1)])  # 边走边右转（中）
        primitives.append([(4, 2), (1, 2), (4, 2), (1, 1)])  # 边走边左转（中）
        
        # === 先转后走系列 ===
        primitives.append([(3, 3), (1, 4)])  # 先右转再直行
        primitives.append([(4, 3), (1, 4)])  # 先左转再直行
        primitives.append([(3, 5), (1, 2)])  # 大右转+前进
        primitives.append([(4, 5), (1, 2)])  # 大左转+前进
        
        # === 纯转向系列 ===
        primitives.append([(3, 6)])  # 持续右转
        primitives.append([(4, 6)])  # 持续左转
        
        # === 后退系列 ===
        primitives.append([(2, 4)])  # 后退
        primitives.append([(2, 2), (3, 3)])  # 后退+右转
        primitives.append([(2, 2), (4, 3)])  # 后退+左转
        
        # === 待命 ===
        primitives.append([(0, 4)])  # 待命
        
        return primitives
    
    def simulate_motion(self, pos, angle, primitive, walls):
        """
        模拟执行一个运动基元后的轨迹
        返回: (最终位置, 最终角度, 轨迹点列表, 是否发生碰撞)
        """
        x, y = float(pos[0]), float(pos[1])
        current_angle = float(angle)
        trajectory = [(x, y)]
        collision = False
        
        for action, frames in primitive:
            for _ in range(frames):
                if action == 1:  # 前进
                    rad = math.radians(current_angle)
                    dx = math.cos(rad) * TANK_SPEED
                    dy = -math.sin(rad) * TANK_SPEED
                    new_x, new_y = x + dx, y + dy
                elif action == 2:  # 后退
                    rad = math.radians(current_angle)
                    dx = math.cos(rad) * TANK_SPEED
                    dy = -math.sin(rad) * TANK_SPEED
                    new_x, new_y = x - dx, y - dy
                elif action == 3:  # 顺时针（角度增加）
                    current_angle += ROTATION_SPEED
                    new_x, new_y = x, y
                elif action == 4:  # 逆时针（角度减少）
                    current_angle -= ROTATION_SPEED
                    new_x, new_y = x, y
                else:  # 待命
                    new_x, new_y = x, y
                
                # 碰撞检测
                if self._check_collision((new_x, new_y), walls):
                    collision = True
                    # 不更新位置，但继续模拟
                else:
                    x, y = new_x, new_y
                
                trajectory.append((x, y))
        
        # 归一化角度
        while current_angle > 180: current_angle -= 360
        while current_angle < -180: current_angle += 360
        
        return (x, y), current_angle, trajectory, collision
    
    def _check_collision(self, pos, walls):
        """检查位置是否碰撞墙壁"""
        half_size = TANK_SIZE // 2
        rect = pygame.Rect(pos[0] - half_size, pos[1] - half_size, TANK_SIZE, TANK_SIZE)
        for w in walls:
            if rect.colliderect(w.rect):
                return True
        # 边界检查
        if pos[0] < half_size or pos[0] > SCREEN_WIDTH - half_size:
            return True
        if pos[1] < half_size or pos[1] > SCREEN_HEIGHT - half_size:
            return True
        return False
    
    def evaluate_trajectory(self, end_pos, end_angle, trajectory, collision,
                           goal_pos, path_points, walls, bullets=None, bot_id=None):
        """
        评估轨迹的代价
        返回: 代价值（越小越好）
        """
        cost = 0.0
        
        # 1. 碰撞惩罚（最重要）
        if collision:
            cost += 1000.0
        
        # 2. 目标方向代价
        if goal_pos:
            target_angle = math.degrees(math.atan2(
                -(goal_pos[1] - end_pos[1]),
                goal_pos[0] - end_pos[0]
            ))
            angle_diff = abs(self._normalize_angle(target_angle - end_angle))
            cost += DWA_HEADING_WEIGHT * angle_diff
        
        # 3. A* 路径跟随代价
        if path_points and len(path_points) > 0:
            # 找到轨迹终点到路径的最近点距离
            min_dist = float('inf')
            for path_point in path_points[:5]:  # 只考虑前5个路径点
                px, py = path_point
                dist = math.hypot(px - end_pos[0], py - end_pos[1])
                min_dist = min(min_dist, dist)
            cost += DWA_PATH_WEIGHT * min_dist
        
        # 4. 障碍物接近代价
        min_obs_dist = self._get_min_obstacle_distance(trajectory, walls)
        if min_obs_dist < TANK_SIZE * 1.5:
            cost += DWA_OBSTACLE_WEIGHT * (TANK_SIZE * 1.5 - min_obs_dist)
        
        # 5. 子弹躲避代价
        if bullets and bot_id is not None:
            bullet_cost = self._evaluate_bullet_risk(trajectory, bullets, bot_id)
            cost += bullet_cost
        
        # 6. 距离目标代价
        if goal_pos:
            dist_to_goal = math.hypot(goal_pos[0] - end_pos[0], goal_pos[1] - end_pos[1])
            cost += DWA_DISTANCE_WEIGHT * dist_to_goal
        
        return cost
    
    def _evaluate_bullet_risk(self, trajectory, bullets, bot_id):
        """评估轨迹对子弹的风险"""
        risk = 0.0
        for bullet in bullets:
            if bullet.owner_id == bot_id:
                continue
            
            # 预测子弹轨迹
            bx, by = bullet.rect.centerx, bullet.rect.centery
            
            for traj_point in trajectory:
                tx, ty = traj_point
                # 计算轨迹点与子弹的距离
                dist = math.hypot(tx - bx, ty - by)
                if dist < DODGE_RADIUS:
                    risk += (DODGE_RADIUS - dist) * 5.0
                
                # 子弹移动一步
                bx += bullet.dx
                by += bullet.dy
        
        return risk
    
    def _get_min_obstacle_distance(self, trajectory, walls):
        """获取轨迹到最近障碍物的距离"""
        min_dist = float('inf')
        for point in trajectory:
            for w in walls:
                # 计算点到矩形的最短距离
                cx = max(w.rect.left, min(point[0], w.rect.right))
                cy = max(w.rect.top, min(point[1], w.rect.bottom))
                dist = math.hypot(point[0] - cx, point[1] - cy)
                min_dist = min(min_dist, dist)
        return min_dist
    
    def _normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle < -180: angle += 360
        return angle
    
    def select_best_action(self, bot_pos, bot_angle, goal_pos, path_points, 
                          walls, bullets=None, bot_id=None):
        """
        选择最佳动作
        返回: 最佳动作ID (0-4)
        """
        best_cost = float('inf')
        best_primitive_idx = 0
        
        for idx, primitive in enumerate(self.motion_primitives):
            end_pos, end_angle, trajectory, collision = self.simulate_motion(
                bot_pos, bot_angle, primitive, walls
            )
            
            cost = self.evaluate_trajectory(
                end_pos, end_angle, trajectory, collision,
                goal_pos, path_points, walls, bullets, bot_id
            )
            
            if cost < best_cost:
                best_cost = cost
                best_primitive_idx = idx
        
        # 返回选中基元的第一个动作
        return self.motion_primitives[best_primitive_idx][0][0]


class BotAI:
    """
    高级机器人 AI 控制器 (A* + DWA)
    """
    
    STATE_DODGE = "躲避"
    STATE_ATTACK = "攻击"
    STATE_CHASE = "追击"
    STATE_UNSTUCK = "脱困"
    
    def __init__(self, grid_map, pathfinder, debug_mode=False):
        self.grid_map = grid_map
        self.pathfinder = pathfinder  # 使用 A* 寻路器
        self.dwa_planner = DWAPlanner(grid_map)  # DWA 局部规划器
        self.debug_mode = debug_mode
        
        # 状态变量
        self.current_path = []  # A* 全局路径（网格坐标）
        self.current_path_pixels = []  # 像素坐标路径
        self.last_pos = (0, 0)
        self.last_pos_time = 0
        self.stuck_counter = 0
        self.unstuck_action = None
        self.unstuck_timer = 0
        
        # 路径更新计数
        self.path_update_counter = 0
        
        self.action_log = []
        self.last_log_step = -1

    def decide_action(self, bot, target, walls, steps, bullets=None, can_attack=True):
        """
        主决策函数
        返回: 0=待命, 1=前进, 2=后退, 3=顺时针, 4=逆时针, 5=射击
        
        Args:
            bot: Bot坦克对象
            target: 目标坦克对象
            walls: 墙壁对象组
            steps: 当前步数
            bullets: 子弹对象组
            can_attack: 是否允许攻击（False时只移动不攻击）
        """
        bot_pos = bot.rect.center
        target_pos = target.rect.center
        
        # 1. 检测卡死 (最高优先级，除了躲避)
        if steps % STUCK_CHECK_FRAMES == 0:
            dist_moved = math.hypot(bot_pos[0] - self.last_pos[0], bot_pos[1] - self.last_pos[1])
            if dist_moved < STUCK_THRESHOLD:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            self.last_pos = bot_pos

        # 如果处于脱困模式
        if self.unstuck_timer > 0:
            self.unstuck_timer -= 1
            return self.unstuck_action
        
        # 触发脱困
        if self.stuck_counter >= 2:
            self.stuck_counter = 0
            self.unstuck_action, self.unstuck_timer = self._calculate_unstuck_action(bot, walls)
            self._log(steps, "UNSTUCK", self.unstuck_action, "检测到卡死，智能脱困")
            return self.unstuck_action

        # 2. 躲避子弹 (高优先级) - 使用DWA进行智能躲避
        dangerous_bullet = self._get_most_dangerous_bullet(bot, bullets, walls)
        if dangerous_bullet:
            action = self._calculate_dodge_action_dwa(bot, dangerous_bullet, walls, bullets)
            self._log(steps, "DODGE", action, "检测到致命威胁")
            return action

        # 3. 攻击判定 (中优先级) - 仅在允许攻击时执行
        if can_attack:
            aim_action = self._calculate_combat_action(bot, target, walls)
            if aim_action is not None:
                self._log(steps, "ATTACK", aim_action, "锁定目标")
                return aim_action

        # 4. 追击/寻路 (低优先级) - 使用 A* + DWA
        chase_action = self._calculate_chase_action_astar_dwa(bot, target, walls, steps, bullets)
        self._log(steps, "CHASE", chase_action, "寻找目标")
        return chase_action

    # ============================================================
    #                       A* + DWA 追击系统
    # ============================================================

    def _calculate_chase_action_astar_dwa(self, bot, target, walls, steps, bullets):
        """
        使用 A* 进行全局规划，Pure Pursuit + DWA 进行局部控制
        """
        bot_pos = bot.rect.center
        target_pos = target.rect.center
        
        # 更新 A* 全局路径（每15帧或路径为空时）
        if steps % 15 == 0 or not self.current_path:
            self._update_global_path(bot_pos, target_pos)
        
        # 使用 Pure Pursuit 获取前瞻目标点
        goal_pos = self._get_pure_pursuit_goal(bot_pos, bot.angle)
        
        if goal_pos is None:
            # 没有路径，直接朝向目标
            goal_pos = target_pos
        
        # 计算到目标的角度差
        target_angle = math.degrees(math.atan2(
            -(goal_pos[1] - bot_pos[1]),
            goal_pos[0] - bot_pos[0]
        ))
        angle_diff = self._normalize_angle(target_angle - bot.angle)
        
        # 根据角度差选择行为模式
        if abs(angle_diff) < ANGLE_TOLERANCE_TURN:
            # 角度很小，直接前进
            return 1
        elif abs(angle_diff) < SMOOTH_TURN_THRESHOLD:
            # 中等角度差，使用 DWA 进行边走边转
            action = self.dwa_planner.select_best_action(
                bot_pos=bot_pos,
                bot_angle=bot.angle,
                goal_pos=goal_pos,
                path_points=self.current_path_pixels,
                walls=walls,
                bullets=bullets,
                bot_id=bot.id
            )
            return action
        else:
            # 大角度差，先原地转向
            # angle_diff > 0 表示需要增加角度，动作 3 是角度增加
            # angle_diff < 0 表示需要减少角度，动作 4 是角度减少
            return 3 if angle_diff > 0 else 4
    
    def _get_pure_pursuit_goal(self, bot_pos, bot_angle):
        """
        Pure Pursuit 算法：找到路径上距离坦克 lookahead 距离的目标点
        这样可以实现更平滑的路径跟随
        """
        if not self.current_path_pixels:
            return None
        
        # 移除已经到达的路径点
        while self.current_path_pixels:
            next_pixel = self.current_path_pixels[0]
            dist = math.hypot(next_pixel[0] - bot_pos[0], next_pixel[1] - bot_pos[1])
            
            if dist < NODE_ARRIVAL_DISTANCE:
                self.current_path_pixels.pop(0)
                if self.current_path:
                    self.current_path.pop(0)
            else:
                break
        
        if not self.current_path_pixels:
            return None
        
        # 找到 lookahead 距离处的目标点
        accumulated_dist = 0
        prev_point = bot_pos
        
        for i, point in enumerate(self.current_path_pixels):
            segment_dist = math.hypot(point[0] - prev_point[0], point[1] - prev_point[1])
            accumulated_dist += segment_dist
            
            if accumulated_dist >= PURE_PURSUIT_LOOKAHEAD:
                # 在这段路径上插值找到精确的 lookahead 点
                overshoot = accumulated_dist - PURE_PURSUIT_LOOKAHEAD
                if segment_dist > 0:
                    ratio = overshoot / segment_dist
                    goal_x = point[0] - ratio * (point[0] - prev_point[0])
                    goal_y = point[1] - ratio * (point[1] - prev_point[1])
                    return (goal_x, goal_y)
                return point
            
            prev_point = point
        
        # 路径不够长，返回最后一个点
        return self.current_path_pixels[-1]
    
    def _update_global_path(self, bot_pos, target_pos):
        """更新 A* 全局路径"""
        start_grid = self.grid_map.pixel_to_grid(*bot_pos)
        end_grid = self.grid_map.pixel_to_grid(*target_pos)
        
        self.current_path = self.pathfinder.find_path(start_grid, end_grid)
        
        # 转换为像素坐标
        self.current_path_pixels = [
            self.grid_map.grid_to_pixel(*grid_pos) 
            for grid_pos in self.current_path
        ]
    
    def _get_current_goal(self, bot_pos):
        """获取当前应该追踪的路径点（保留用于兼容性）"""
        # 移除已经到达的路径点
        while self.current_path_pixels:
            next_pixel = self.current_path_pixels[0]
            dist = math.hypot(next_pixel[0] - bot_pos[0], next_pixel[1] - bot_pos[1])
            
            if dist < NODE_ARRIVAL_DISTANCE:
                self.current_path_pixels.pop(0)
                if self.current_path:
                    self.current_path.pop(0)
            else:
                break
        
        if self.current_path_pixels:
            # 返回前瞻点而不是最近点
            return self._get_lookahead_point(bot_pos, PURE_PURSUIT_LOOKAHEAD)
        return None
    
    def _get_lookahead_point(self, bot_pos, lookahead_dist):
        """
        获取路径上指定前瞻距离的点
        如果路径不够长，返回路径终点
        """
        if not self.current_path_pixels:
            return None
        
        accumulated_dist = 0
        prev_point = bot_pos
        
        for point in self.current_path_pixels:
            segment_dist = math.hypot(point[0] - prev_point[0], point[1] - prev_point[1])
            accumulated_dist += segment_dist
            
            if accumulated_dist >= lookahead_dist:
                return point
            
            prev_point = point
        
        # 返回最后一个点
        return self.current_path_pixels[-1]

    # ============================================================
    #                       躲避系统 (DWA增强)
    # ============================================================

    def _get_most_dangerous_bullet(self, bot, bullets, walls):
        """找到最近的、且没有墙壁阻挡的威胁子弹"""
        if not bullets:
            return None
            
        bot_pos = bot.rect.center
        min_dist = DODGE_RADIUS
        danger_bullet = None
        
        for b in bullets:
            if b.owner_id == bot.id: continue
            
            dist = math.hypot(b.rect.centerx - bot_pos[0], b.rect.centery - bot_pos[1])
            if dist > DODGE_RADIUS: continue
            
            # 方向检测：子弹是否在靠近
            to_bot_x = bot_pos[0] - b.rect.centerx
            to_bot_y = bot_pos[1] - b.rect.centery
            dot_prod = b.dx * to_bot_x + b.dy * to_bot_y
            
            if dot_prod <= 0: continue
            
            # 射线检测
            if self._raycast_hit_wall(b.rect.center, bot_pos, walls):
                continue
            
            # 预测最近点距离
            b_speed = math.hypot(b.dx, b.dy)
            if b_speed == 0: continue
            cross_prod = abs(b.dx * to_bot_y - b.dy * to_bot_x)
            perp_dist = cross_prod / b_speed
            
            if perp_dist > (TANK_SIZE / 2) + 10:
                continue
                
            if dist < min_dist:
                min_dist = dist
                danger_bullet = b
                
        return danger_bullet

    def _calculate_dodge_action_dwa(self, bot, bullet, walls, bullets):
        """使用 DWA 计算最佳躲避动作"""
        bot_pos = bot.rect.center
        
        # 计算子弹方向角度（注意：子弹 dy 已经是 pygame 坐标系，需要取负转换为数学坐标系）
        # bullet.dy 是 pygame 坐标系（向下为正），需要转换
        bullet_angle = math.atan2(-bullet.dy, bullet.dx)  # 转换为数学坐标系角度
        
        # 垂直于子弹方向的两个躲避方向
        dodge_angle1 = bullet_angle + math.pi / 2
        dodge_angle2 = bullet_angle - math.pi / 2
        
        # 选择更远离墙壁的方向（使用 pygame 坐标系计算位置）
        dodge_dist = TANK_SIZE * 3
        pos1 = (bot_pos[0] + math.cos(dodge_angle1) * dodge_dist,
                bot_pos[1] - math.sin(dodge_angle1) * dodge_dist)  # 转换回 pygame 坐标系
        pos2 = (bot_pos[0] + math.cos(dodge_angle2) * dodge_dist,
                bot_pos[1] - math.sin(dodge_angle2) * dodge_dist)  # 转换回 pygame 坐标系
        
        # 检查哪个方向更安全
        safe1 = not self._check_collision(pos1, walls)
        safe2 = not self._check_collision(pos2, walls)
        
        if safe1 and not safe2:
            goal_pos = pos1
        elif safe2 and not safe1:
            goal_pos = pos2
        elif safe1 and safe2:
            # 两个都安全，选择离当前朝向更近的
            angle1_diff = abs(self._normalize_angle(math.degrees(dodge_angle1) - bot.angle))
            angle2_diff = abs(self._normalize_angle(math.degrees(dodge_angle2) - bot.angle))
            goal_pos = pos1 if angle1_diff < angle2_diff else pos2
        else:
            # 两个都不安全，尝试后退
            bot_rad = math.radians(bot.angle)
            goal_pos = (bot_pos[0] - math.cos(bot_rad) * dodge_dist,
                       bot_pos[1] + math.sin(bot_rad) * dodge_dist)  # 后退（pygame 坐标系）
        
        # 使用 DWA 选择最佳动作
        return self.dwa_planner.select_best_action(
            bot_pos=bot_pos,
            bot_angle=bot.angle,
            goal_pos=goal_pos,
            path_points=[],
            walls=walls,
            bullets=bullets,
            bot_id=bot.id
        )

    # ============================================================
    #                       战斗系统 (Bounce & Predict)
    # ============================================================

    def _calculate_combat_action(self, bot, target, walls):
        """
        计算攻击动作
        """
        bot_pos = bot.rect.center
        target_pos = target.rect.center
        
        dist = math.hypot(bot_pos[0] - target_pos[0], bot_pos[1] - target_pos[1])
        
        # 致命一击检测
        if bot.cooldown == 0:
            will_hit = self._simulate_shot(bot_pos, bot.angle, target, walls, bot.rect)
            if will_hit:
                return 5

        # 瞄准逻辑
        if dist < VISION_DISTANCE:
            has_los = not self._raycast_hit_wall(bot_pos, target_pos, walls)
            
            if has_los:
                # 预判瞄准
                lead_x = target_pos[0] + (target.vx * (dist / BULLET_SPEED))
                lead_y = target_pos[1] + (target.vy * (dist / BULLET_SPEED))
                
                target_angle = math.degrees(math.atan2(-(lead_y - bot_pos[1]), lead_x - bot_pos[0]))
                diff = self._normalize_angle(target_angle - bot.angle)
                
                if abs(diff) < ANGLE_TOLERANCE:
                    return 0 
                elif diff > 0:
                    return 3
                else:
                    return 4
        
        return None

    def _simulate_shot(self, start_pos, angle, target, walls, bot_rect=None):
        """
        物理引擎模拟：判断给定角度发射子弹是否会命中目标
        """
        x, y = float(start_pos[0]), float(start_pos[1])
        rad = math.radians(angle)
        dx = math.cos(rad) * BULLET_SPEED
        dy = -math.sin(rad) * BULLET_SPEED
        
        target_rect = target.rect.inflate(-5, -5)
        
        safe_dist = TANK_SIZE + 10
        bounces = 0
        max_bounces = MAX_BOUNCES
        
        for step in range(300):
            x += dx
            y += dy
            
            rect = pygame.Rect(x - 3, y - 3, 6, 6)
            
            for w in walls:
                if rect.colliderect(w.rect):
                    bounces += 1
                    if bounces > max_bounces:
                        return False
                    
                    overlap_x = min(rect.right, w.rect.right) - max(rect.left, w.rect.left)
                    overlap_y = min(rect.bottom, w.rect.bottom) - max(rect.top, w.rect.top)
                    
                    if overlap_x < overlap_y:
                        dx *= -1
                        x += dx * 2
                    else:
                        dy *= -1
                        y += dy * 2
                    break
            
            if bot_rect:
                dist_from_start = math.hypot(x - start_pos[0], y - start_pos[1])
                if dist_from_start > safe_dist:
                    if rect.colliderect(bot_rect):
                        return False
            
            if rect.colliderect(target_rect):
                return True
            
            if not (0 <= x <= SCREEN_WIDTH and 0 <= y <= SCREEN_HEIGHT):
                return False
        
        return False

    # ============================================================
    #                       工具函数
    # ============================================================

    def _calculate_unstuck_action(self, bot, walls):
        """智能脱困"""
        bot_pos = bot.rect.center
        check_dist = TANK_SIZE * 2.5
        
        wall_directions = []
        for angle_offset in range(0, 360, 15):
            rad = math.radians(angle_offset)
            check_x = bot_pos[0] + math.cos(rad) * check_dist
            check_y = bot_pos[1] - math.sin(rad) * check_dist
            
            if self._check_collision((check_x, check_y), walls):
                wall_directions.append(angle_offset)
        
        if not wall_directions:
            return (random.choice([1, 2, 3, 4]), 20)
        
        max_density = 0
        best_angle = 0
        
        for start_angle in range(0, 360, 15):
            density = 0
            for wall_angle in wall_directions:
                diff = abs(wall_angle - start_angle)
                diff = min(diff, 360 - diff)
                if diff <= 30:
                    density += 1
            
            if density > max_density:
                max_density = density
                best_angle = start_angle
        
        escape_angle = (best_angle + 180) % 360
        
        current_rad = math.radians(bot.angle)
        escape_rad = math.radians(escape_angle)
        angle_diff = math.degrees(self._normalize_angle_rad(escape_rad - current_rad))
        
        if abs(angle_diff) > 45:
            return (3 if angle_diff > 0 else 4, 8)
        elif abs(angle_diff) > 15:
            return (3 if angle_diff > 0 else 4, 12)
        else:
            return (1, 35)

    def _raycast_hit_wall(self, start, end, walls):
        """简单的射线墙壁检测"""
        line = (start, end)
        for w in walls:
            if w.rect.clipline(line):
                return True
        return False

    def _check_collision(self, pos, walls):
        """检查点是否在墙内"""
        r = pygame.Rect(pos[0]-10, pos[1]-10, 20, 20)
        for w in walls:
            if r.colliderect(w.rect):
                return True
        return False

    def _normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle < -180: angle += 360
        return angle

    def _normalize_angle_rad(self, angle_rad):
        """归一化弧度到 [-π, π]"""
        while angle_rad > math.pi: angle_rad -= 2 * math.pi
        while angle_rad < -math.pi: angle_rad += 2 * math.pi
        return angle_rad

    def _log(self, step, state, action, msg):
        if not self.debug_mode or step == self.last_log_step:
            return
        self.last_log_step = step
        print(f"[Bot] {state} | Act:{action} | {msg}")
        
    def clear_action_log(self):
        pass