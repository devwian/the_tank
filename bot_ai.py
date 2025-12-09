"""
机器人 AI 模块
实现 Bot 的智能决策逻辑（完全重写版）
核心思路：状态机 + 简单有效的躲避算法
"""

import math
import random
from constants import (
    BULLET_SPEED, ANGLE_TOLERANCE, NODE_ARRIVAL_DISTANCE,
    VISION_DISTANCE, LINE_OF_SIGHT_CHECK, PATHFINDING_UPDATE_FREQ,
    TANK_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT
)

# ============ AI 参数 ============
DODGE_RADIUS = 150           # 躲避检测半径
DANGER_RADIUS = 60           # 紧急躲避半径
HIT_PREDICT_FRAMES = 20      # 预测未来多少帧


class BotAI:
    """
    机器人 AI 控制器
    
    状态机：
    - DODGE: 躲避状态（最高优先级）
    - COMBAT: 战斗状态（有视线时瞄准射击）
    - CHASE: 追击状态（寻路接近目标）
    """
    
    STATE_DODGE = "dodge"
    STATE_COMBAT = "combat"
    STATE_CHASE = "chase"
    
    def __init__(self, grid_map, pathfinder):
        self.grid_map = grid_map
        self.pathfinder = pathfinder
        self.current_path = []
        self.steps = 0
        self.state = self.STATE_CHASE
        self.dodge_direction = None  # 躲避方向: 'left', 'right', 'forward', 'backward'
    
    def decide_action(self, bot, target, walls, steps, bullets=None):
        """
        主决策函数
        返回: 0=待命, 1=前进, 2=后退, 3=顺时针, 4=逆时针, 5=射击
        """
        self.steps = steps
        
        bot_pos = bot.rect.center
        target_pos = target.rect.center
        dist_to_target = math.hypot(
            target_pos[0] - bot_pos[0], 
            target_pos[1] - bot_pos[1]
        )
        
        # ========== 1. 检测是否需要躲避 ==========
        if bullets:
            danger = self._find_danger_bullet(bot, bullets)
            if danger:
                self.state = self.STATE_DODGE
                return self._do_dodge(bot, danger)
        
        # ========== 2. 检测是否可以战斗 ==========
        if dist_to_target < VISION_DISTANCE:
            # 计算预判位置
            pred_pos = self._predict_position(target, dist_to_target)
            
            # 检查视线
            if self._has_line_of_sight(bot_pos, pred_pos, walls):
                self.state = self.STATE_COMBAT
                return self._do_combat(bot, pred_pos)
        
        # ========== 3. 追击模式 ==========
        self.state = self.STATE_CHASE
        return self._do_chase(bot, target_pos)
    
    # ============================================================
    #                       躲避系统
    # ============================================================
    
    def _find_danger_bullet(self, bot, bullets):
        """
        找到最危险的子弹
        返回: (bullet, time_to_hit, closest_distance) 或 None
        """
        bot_x, bot_y = bot.rect.center
        bot_radius = TANK_SIZE / 2
        
        most_dangerous = None
        min_time = float('inf')
        
        for bullet in bullets:
            # 跳过自己的子弹
            if bullet.owner_id == bot.id:
                continue
            
            bx, by = bullet.rect.center
            dx, dy = bullet.dx, bullet.dy
            
            # 当前距离
            dist = math.hypot(bx - bot_x, by - bot_y)
            if dist > DODGE_RADIUS:
                continue
            
            # 计算子弹是否朝向 bot
            # 向量：子弹 -> bot
            to_bot_x = bot_x - bx
            to_bot_y = bot_y - by
            
            # 子弹速度的模
            bullet_speed = math.hypot(dx, dy)
            if bullet_speed < 0.001:
                continue
            
            # 点积判断方向
            dot = (to_bot_x * dx + to_bot_y * dy)
            if dot <= 0:
                # 子弹正在远离
                continue
            
            # 计算最近通过距离（垂直距离）
            # 叉积 / 速度模 = 垂直距离
            cross = abs(to_bot_x * dy - to_bot_y * dx)
            closest_dist = cross / bullet_speed
            
            # 如果最近距离大于坦克半径，不会被击中
            if closest_dist > bot_radius + TANK_SIZE * 0.5:
                continue
            
            # 计算到达时间
            time_to_hit = dot / (bullet_speed * bullet_speed)
            
            # 只关注即将到来的子弹
            if time_to_hit < HIT_PREDICT_FRAMES and time_to_hit < min_time:
                min_time = time_to_hit
                most_dangerous = (bullet, time_to_hit, closest_dist)
        
        return most_dangerous
    
    def _do_dodge(self, bot, danger_info):
        """
        执行躲避动作
        核心思路：垂直于子弹方向移动
        """
        bullet, time_to_hit, _ = danger_info
        bot_x, bot_y = bot.rect.center
        bx, by = bullet.rect.center
        
        # 子弹飞行方向
        bullet_angle = math.atan2(bullet.dy, bullet.dx)
        
        # 两个垂直方向（左和右）
        perp_left = bullet_angle + math.pi / 2
        perp_right = bullet_angle - math.pi / 2
        
        # 坦克当前朝向
        tank_angle = math.radians(bot.angle)
        
        # 计算坦克前进方向与两个垂直方向的夹角
        # 选择与前进方向更接近的那个
        diff_left = abs(self._angle_diff(tank_angle, perp_left))
        diff_right = abs(self._angle_diff(tank_angle, perp_right))
        diff_left_back = abs(self._angle_diff(tank_angle + math.pi, perp_left))
        diff_right_back = abs(self._angle_diff(tank_angle + math.pi, perp_right))
        
        # 找最小角度差，决定动作
        options = [
            (diff_left, 1, "forward_left"),      # 前进接近左垂直
            (diff_right, 1, "forward_right"),    # 前进接近右垂直
            (diff_left_back, 2, "back_left"),    # 后退接近左垂直
            (diff_right_back, 2, "back_right"),  # 后退接近右垂直
        ]
        
        best = min(options, key=lambda x: x[0])
        angle_diff, move_action, _ = best
        
        # 如果角度差太大，需要先转向
        if angle_diff > math.radians(60):
            # 紧急情况直接移动
            if time_to_hit < 5:
                return random.choice([1, 2])  # 随便动
            
            # 有时间就转向
            if diff_left < diff_right:
                target_angle = perp_left
            else:
                target_angle = perp_right
            
            turn_diff = self._angle_diff(tank_angle, target_angle)
            return 3 if turn_diff > 0 else 4
        
        return move_action
    
    def _angle_diff(self, a1, a2):
        """计算两个角度的差值，结果在 [-pi, pi]"""
        diff = a2 - a1
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    # ============================================================
    #                       战斗系统
    # ============================================================
    
    def _predict_position(self, target, dist):
        """预测目标位置"""
        tx, ty = target.rect.center
        
        if dist < 1:
            return (tx, ty)
        
        # 简单线性预测
        time_to_hit = dist / BULLET_SPEED
        pred_x = tx + target.vx * time_to_hit * 1.2
        pred_y = ty + target.vy * time_to_hit * 1.2
        
        # 边界约束
        pred_x = max(TANK_SIZE, min(SCREEN_WIDTH - TANK_SIZE, pred_x))
        pred_y = max(TANK_SIZE, min(SCREEN_HEIGHT - TANK_SIZE, pred_y))
        
        return (pred_x, pred_y)
    
    def _do_combat(self, bot, target_pos):
        """
        战斗模式：瞄准并射击
        """
        bot_x, bot_y = bot.rect.center
        tx, ty = target_pos
        
        # 计算目标角度
        target_angle = math.degrees(math.atan2(-(ty - bot_y), tx - bot_x))
        
        # 当前角度差
        angle_diff = self._normalize_angle(target_angle - bot.angle)
        
        # 清空寻路路径
        self.current_path = []
        
        # 瞄准
        if abs(angle_diff) <= ANGLE_TOLERANCE:
            # 已瞄准，射击
            if bot.cooldown == 0:
                return 5
            # 等待冷却，可以稍微移动
            return 0
        
        # 需要转向
        return 3 if angle_diff > 0 else 4
    
    def _has_line_of_sight(self, p1, p2, walls):
        """检查两点之间是否有直线视线"""
        for wall in walls:
            if wall.rect.clipline(p1, p2):
                return False
        return True
    
    # ============================================================
    #                       追击系统
    # ============================================================
    
    def _do_chase(self, bot, target_pos):
        """
        追击模式：寻路接近目标
        """
        bot_x, bot_y = bot.rect.center
        
        # 转换为网格坐标
        gx_bot, gy_bot = self.grid_map.pixel_to_grid(bot_x, bot_y)
        gx_target, gy_target = self.grid_map.pixel_to_grid(*target_pos)
        
        # 定期更新路径
        if self.steps % PATHFINDING_UPDATE_FREQ == 0 or not self.current_path:
            self.current_path = self.pathfinder.find_path(
                (gx_bot, gy_bot), (gx_target, gy_target)
            )
        
        # 沿路径移动
        if self.current_path:
            return self._follow_path(bot)
        
        return 0
    
    def _follow_path(self, bot):
        """沿路径移动"""
        if not self.current_path:
            return 0
        
        bot_x, bot_y = bot.rect.center
        
        # 获取下一个路点
        next_grid = self.current_path[0]
        next_x, next_y = self.grid_map.grid_to_pixel(*next_grid)
        
        # 到达检测
        dist = math.hypot(next_x - bot_x, next_y - bot_y)
        if dist < NODE_ARRIVAL_DISTANCE:
            self.current_path.pop(0)
            if not self.current_path:
                return 0
            next_grid = self.current_path[0]
            next_x, next_y = self.grid_map.grid_to_pixel(*next_grid)
        
        # 计算目标角度
        target_angle = math.degrees(math.atan2(-(next_y - bot_y), next_x - bot_x))
        angle_diff = self._normalize_angle(target_angle - bot.angle)
        
        # 转向或前进
        if abs(angle_diff) > 20:
            return 3 if angle_diff > 0 else 4
        return 1
    
    def _normalize_angle(self, angle):
        """归一化角度到 [-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
