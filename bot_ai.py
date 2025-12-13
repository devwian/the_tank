"""
机器人 AI 模块 (强化版)
实现 Bot 的智能决策逻辑
改进点：引入射线检测、反弹预测、更平滑的移动和防卡死机制
"""

import math
import random
import pygame
from constants import (
    BULLET_SPEED, ANGLE_TOLERANCE, NODE_ARRIVAL_DISTANCE,
    VISION_DISTANCE, TANK_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT,
    TANK_SPEED, MAX_BOUNCES, ROTATION_SPEED
)

# ============ AI 参数 ============
DODGE_RADIUS = 180           # 躲避检测半径
PREDICT_FRAMES = 15          # 射击预判帧数
STUCK_THRESHOLD = 5          # 判定卡死的像素阈值
STUCK_CHECK_FRAMES = 30      # 多少帧检查一次卡死

class BotAI:
    """
    高级机器人 AI 控制器
    """
    
    STATE_DODGE = "躲避"
    STATE_ATTACK = "攻击"
    STATE_CHASE = "追击"
    STATE_UNSTUCK = "脱困"
    
    def __init__(self, grid_map, pathfinder, debug_mode=False):
        self.grid_map = grid_map
        self.pathfinder = pathfinder
        self.debug_mode = debug_mode
        
        # 状态变量
        self.current_path = []
        self.last_pos = (0, 0)
        self.last_pos_time = 0
        self.stuck_counter = 0
        self.unstuck_action = None
        self.unstuck_timer = 0
        
        self.action_log = []
        self.last_log_step = -1

    def decide_action(self, bot, target, walls, steps, bullets=None):
        """
        主决策函数
        返回: 0=待命, 1=前进, 2=后退, 3=顺时针, 4=逆时针, 5=射击
        """
        bot_pos = bot.rect.center
        target_pos = target.rect.center
        
        # 1. 检测卡死 (最高优先级，除了躲避)
        # 每隔一段时间检查位置变化
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
            # 智能脱困：检测墙体位置决定脱困方向
            self.unstuck_action, self.unstuck_timer = self._calculate_unstuck_action(bot, walls)
            self._log(steps, "UNSTUCK", self.unstuck_action, "检测到卡死，智能脱困")
            return self.unstuck_action

        # 2. 躲避子弹 (高优先级)
        dangerous_bullet = self._get_most_dangerous_bullet(bot, bullets, walls)
        if dangerous_bullet:
            action = self._calculate_dodge_action(bot, dangerous_bullet, walls)
            self._log(steps, "DODGE", action, "检测到致命威胁")
            return action

        # 3. 攻击判定 (中优先级)
        # 检查是否能直接射击或通过反弹射击
        aim_action = self._calculate_combat_action(bot, target, walls)
        if aim_action is not None:
            self._log(steps, "ATTACK", aim_action, "锁定目标")
            return aim_action

        # 4. 追击/寻路 (低优先级)
        chase_action = self._calculate_chase_action(bot, target, walls, steps)
        self._log(steps, "CHASE", chase_action, "寻找目标")
        return chase_action

    # ============================================================
    #                       躲避系统 (Smart Dodge)
    # ============================================================

    def _get_most_dangerous_bullet(self, bot, bullets, walls):
        """找到最近的、且没有墙壁阻挡的威胁子弹"""
        if not bullets:
            return None
            
        bot_pos = bot.rect.center
        min_dist = DODGE_RADIUS
        danger_bullet = None
        
        for b in bullets:
            if b.owner_id == bot.id: continue # 忽略自己的子弹
            
            # 距离检测
            dist = math.hypot(b.rect.centerx - bot_pos[0], b.rect.centery - bot_pos[1])
            if dist > DODGE_RADIUS: continue
            
            # 方向检测：子弹是否在靠近？
            # 向量点积 < 0 表示夹角大于90度，即正在靠近
            to_bot_x = bot_pos[0] - b.rect.centerx
            to_bot_y = bot_pos[1] - b.rect.centery
            dot_prod = b.dx * to_bot_x + b.dy * to_bot_y
            
            if dot_prod <= 0: continue # 子弹正在远离
            
            # 射线检测：子弹和坦克之间是否有墙？
            # 只有当子弹真的能打到我时才躲避
            if self._raycast_hit_wall(b.rect.center, bot_pos, walls):
                continue
                
            # 预测最近点距离
            b_speed = math.hypot(b.dx, b.dy)
            if b_speed == 0: continue
            # 叉积计算点到直线距离
            cross_prod = abs(b.dx * to_bot_y - b.dy * to_bot_x)
            perp_dist = cross_prod / b_speed
            
            # 如果弹道偏离超过坦克半径+安全余量，忽略
            if perp_dist > (TANK_SIZE / 2) + 10:
                continue
                
            if dist < min_dist:
                min_dist = dist
                danger_bullet = b
                
        return danger_bullet

    def _calculate_dodge_action(self, bot, bullet, walls):
        """计算最佳躲避动作"""
        # 子弹角度
        bullet_angle = math.atan2(bullet.dy, bullet.dx)
        # 两个垂直方向：+90度 和 -90度
        angle_opts = [bullet_angle + math.pi/2, bullet_angle - math.pi/2]
        
        best_action = 0
        max_safety = -999
        
        # 简单的评估：尝试前/后移动，看哪个离垂直方向更近且不撞墙
        bot_rad = math.radians(bot.angle)
        
        # 模拟 1: 前进
        fx = bot.rect.centerx + math.cos(bot_rad) * TANK_SPEED * 5
        fy = bot.rect.centery - math.sin(bot_rad) * TANK_SPEED * 5
        
        # 模拟 2: 后退
        bx = bot.rect.centerx - math.cos(bot_rad) * TANK_SPEED * 5
        by = bot.rect.centery + math.sin(bot_rad) * TANK_SPEED * 5
        
        # 检查墙壁碰撞
        f_safe = not self._check_collision((fx, fy), walls)
        b_safe = not self._check_collision((bx, by), walls)
        
        if f_safe: return 1
        if b_safe: return 2
        return 3 # 如果前后都撞墙，就转圈听天由命

    # ============================================================
    #                       战斗系统 (Bounce & Predict)
    # ============================================================

    def _calculate_combat_action(self, bot, target, walls):
        """
        计算攻击动作：
        1. 检查当前角度是否能命中（直接或反弹）-> 射击
        2. 检查直视视线 -> 转向瞄准
        """
        bot_pos = bot.rect.center
        target_pos = target.rect.center
        
        dist = math.hypot(bot_pos[0] - target_pos[0], bot_pos[1] - target_pos[1])
        
        # --- 1. 致命一击检测 (Raycast Check) ---
        # 模拟当前角度发射子弹，看是否会命中敌人
        # 这是一个昂贵的操作，只在冷却好时检测
        if bot.cooldown == 0:
            # 传入 bot.rect 用于检测自杀风险
            will_hit = self._simulate_shot(bot_pos, bot.angle, target, walls, bot.rect)
            if will_hit:
                return 5 # 开火！

        # --- 2. 瞄准逻辑 ---
        if dist < VISION_DISTANCE:
            # 简单的直线视线检查
            has_los = not self._raycast_hit_wall(bot_pos, target_pos, walls)
            
            if has_los:
                # 预判瞄准：加上敌人的速度向量
                lead_x = target_pos[0] + (target.vx * (dist / BULLET_SPEED))
                lead_y = target_pos[1] + (target.vy * (dist / BULLET_SPEED))
                
                target_angle = math.degrees(math.atan2(-(lead_y - bot_pos[1]), lead_x - bot_pos[0]))
                diff = self._normalize_angle(target_angle - bot.angle)
                
                if abs(diff) < ANGLE_TOLERANCE:
                    # 瞄准了，但不要直接开火，依赖上面的 _simulate_shot 进行安全检查
                    # 如果冷却好了但上面没返回5，说明射击不安全或无法命中
                    return 0 
                elif diff > 0:
                    return 3 # 顺时针
                else:
                    return 4 # 逆时针
        
        return None

    def _simulate_shot(self, start_pos, angle, target, walls, bot_rect=None):
        """
        物理引擎模拟：判断给定角度发射子弹是否会命中目标
        支持 MAX_BOUNCES 次反弹，同时检测自杀风险
        """
        x, y = float(start_pos[0]), float(start_pos[1])
        rad = math.radians(angle)
        dx = math.cos(rad) * BULLET_SPEED
        dy = -math.sin(rad) * BULLET_SPEED
        
        target_rect = target.rect.inflate(-5, -5)
        
        # 自杀检测相关
        safe_dist = TANK_SIZE + 10  # 安全距离（更大的安全区）
        bounces = 0
        max_bounces = MAX_BOUNCES
        
        for step in range(300):
            # 移动
            x += dx
            y += dy
            
            rect = pygame.Rect(x - 3, y - 3, 6, 6)
            
            # 墙壁碰撞检测
            for w in walls:
                if rect.colliderect(w.rect):
                    bounces += 1
                    if bounces > max_bounces:
                        return False  # 超过反弹次数，子弹消失
                    
                    # 简单反弹：根据碰撞位置决定反弹方向
                    overlap_x = min(rect.right, w.rect.right) - max(rect.left, w.rect.left)
                    overlap_y = min(rect.bottom, w.rect.bottom) - max(rect.top, w.rect.top)
                    
                    if overlap_x < overlap_y:
                        dx *= -1
                        x += dx * 2
                    else:
                        dy *= -1
                        y += dy * 2
                    break
            
            # 自杀检测：子弹飞出安全距离后检测是否会回来击中自己
            if bot_rect:
                dist_from_start = math.hypot(x - start_pos[0], y - start_pos[1])
                if dist_from_start > safe_dist:
                    if rect.colliderect(bot_rect):
                        return False  # 会打到自己！
            
            # 命中敌人判定
            if rect.colliderect(target_rect):
                return True
            
            # 出界判定
            if not (0 <= x <= SCREEN_WIDTH and 0 <= y <= SCREEN_HEIGHT):
                return False
        
        return False

    # ============================================================
    #                       追击系统 (Smooth Pathing)
    # ============================================================

    def _calculate_chase_action(self, bot, target, walls, steps):
        bot_pos = bot.rect.center
        target_pos = target.rect.center
        
        # 只有在需要更新路径时才重新寻路 (CPU优化)
        if steps % 15 == 0 or not self.current_path:
            gx_start = self.grid_map.pixel_to_grid(*bot_pos)
            gx_end = self.grid_map.pixel_to_grid(*target_pos)
            self.current_path = self.pathfinder.find_path(gx_start, gx_end)
        
        if not self.current_path:
            # 没路了？或者已经在同一个格子里
            # 尝试直接转向目标
            target_angle = math.degrees(math.atan2(-(target_pos[1] - bot_pos[1]), target_pos[0] - bot_pos[0]))
            diff = self._normalize_angle(target_angle - bot.angle)
            if abs(diff) > 10:
                return 3 if diff > 0 else 4
            return 1 # 冲！

        # 获取下一个路径点
        next_grid = self.current_path[0]
        next_pixel = self.grid_map.grid_to_pixel(*next_grid)
        
        dist_to_node = math.hypot(next_pixel[0] - bot_pos[0], next_pixel[1] - bot_pos[1])
        
        # 到达节点，移除
        if dist_to_node < NODE_ARRIVAL_DISTANCE:
            self.current_path.pop(0)
            if not self.current_path:
                return 0
            next_pixel = self.grid_map.grid_to_pixel(*self.current_path[0])
            
        # 导航逻辑
        target_angle = math.degrees(math.atan2(-(next_pixel[1] - bot_pos[1]), next_pixel[0] - bot_pos[0]))
        angle_diff = self._normalize_angle(target_angle - bot.angle)
        
        # 智能移动：如果角度偏差不大，可以一边走一边转
        if abs(angle_diff) > 45:
            # 偏差太大，原地旋转
            return 3 if angle_diff > 0 else 4
        elif abs(angle_diff) > 10:
            # 偏差中等，由于坦克不能斜着走，这里根据具体情况切换
            # 这里的简单逻辑是：优先修正角度
            return 3 if angle_diff > 0 else 4
        else:
            # 角度正确，前进
            return 1

    # ============================================================
    #                       工具函数
    # ============================================================

    def _calculate_unstuck_action(self, bot, walls):
        """
        智能脱困：检测四周墙体位置，选择最佳脱困方向
        返回: (action, duration)
        """
        bot_pos = bot.rect.center
        bot_rad = math.radians(bot.angle)
        check_dist = TANK_SIZE * 2  # 检测距离
        
        # 计算四个方向的检测点
        # 前方
        front_x = bot_pos[0] + math.cos(bot_rad) * check_dist
        front_y = bot_pos[1] - math.sin(bot_rad) * check_dist
        front_blocked = self._check_collision((front_x, front_y), walls)
        
        # 后方
        back_x = bot_pos[0] - math.cos(bot_rad) * check_dist
        back_y = bot_pos[1] + math.sin(bot_rad) * check_dist
        back_blocked = self._check_collision((back_x, back_y), walls)
        
        # 左侧 (逆时针90度)
        left_rad = bot_rad + math.pi / 2
        left_x = bot_pos[0] + math.cos(left_rad) * check_dist
        left_y = bot_pos[1] - math.sin(left_rad) * check_dist
        left_blocked = self._check_collision((left_x, left_y), walls)
        
        # 右侧 (顺时针90度)
        right_rad = bot_rad - math.pi / 2
        right_x = bot_pos[0] + math.cos(right_rad) * check_dist
        right_y = bot_pos[1] - math.sin(right_rad) * check_dist
        right_blocked = self._check_collision((right_x, right_y), walls)
        
        # 决策逻辑
        # 优先级：找到空旷方向移动
        
        # 情况1：前方有墙，后方空旷 -> 后退
        if front_blocked and not back_blocked:
            return (2, 25)  # 后退
        
        # 情况2：后方有墙，前方空旷 -> 前进
        if back_blocked and not front_blocked:
            return (1, 20)  # 前进
        
        # 情况3：前后都有墙，检查左右
        if front_blocked and back_blocked:
            if not left_blocked:
                return (4, 15)  # 逆时针转向左侧
            elif not right_blocked:
                return (3, 15)  # 顺时针转向右侧
            else:
                # 四面楚歌，随机挣扎
                return (random.choice([3, 4]), 20)
        
        # 情况4：前后都空旷，但可能是侧面卡住
        if left_blocked and not right_blocked:
            # 左边有墙，顺时针转然后前进
            return (3, 10)
        elif right_blocked and not left_blocked:
            # 右边有墙，逆时针转然后前进
            return (4, 10)
        
        # 情况5：默认后退+旋转组合
        # 先后退一点，再随机转向
        return (2, 15)

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

    def _log(self, step, state, action, msg):
        if not self.debug_mode or step == self.last_log_step:
            return
        self.last_log_step = step
        print(f"[Bot] {state} | Act:{action} | {msg}") 
        # 解开注释以查看详细调试信息
        
    def clear_action_log(self):
        pass