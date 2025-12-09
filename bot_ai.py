"""
机器人 AI 模块
实现 Bot 的智能决策逻辑（增强版）
"""

import math
import random
from constants import (
    BULLET_SPEED, ANGLE_TOLERANCE, NODE_ARRIVAL_DISTANCE,
    VISION_DISTANCE, LINE_OF_SIGHT_CHECK, PATHFINDING_UPDATE_FREQ,
    TANK_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT
)

# 增强 AI 参数
DODGE_DISTANCE = 80          # 躲避检测距离
DODGE_PRIORITY = True        # 是否优先躲避
STRAFE_WHILE_SHOOTING = True # 射击时是否移动
FLANK_PROBABILITY = 0.3      # 包抄概率
PREDICTION_FACTOR = 1.2      # 预判系数（越大预判越远）


class BotAI:
    """机器人 AI 控制器（增强版）"""
    
    def __init__(self, grid_map, pathfinder):
        """
        初始化 Bot AI
        grid_map: GridMap 实例
        pathfinder: BFSPathfinder 实例
        """
        self.grid_map = grid_map
        self.pathfinder = pathfinder
        self.current_path = []
        self.steps = 0
        self.bullets_ref = None  # 子弹引用，用于躲避
        self.last_dodge_dir = 0  # 上次躲避方向
        self.flank_target = None # 包抄目标点
        self.aggression = 0.7    # 进攻性（0-1）
    
    def decide_action(self, bot, target, walls, steps, bullets=None):
        """
        决定 Bot 的下一步行动
        bot: 机器人对象
        target: 目标（通常是玩家坦克）
        walls: 墙壁 Group
        steps: 当前步数
        bullets: 子弹 Group（用于躲避）
        
        返回: 0-5 的动作值
        """
        self.steps = steps
        self.bullets_ref = bullets
        
        # 1. 基础计算
        target_pos = target.rect.center
        bot_pos = bot.rect.center
        dist = math.hypot(target_pos[0] - bot_pos[0], target_pos[1] - bot_pos[1])
        
        # 2. 优先级1：躲避来袭子弹
        if DODGE_PRIORITY and bullets:
            dodge_action = self._check_dodge(bot, bullets)
            if dodge_action is not None:
                return dodge_action
        
        # 3. 计算增强预判位置
        pred_pos = self._predict_target(target, bot_pos, dist)
        
        # 4. 优先级2：战斗模式（有视线）
        if LINE_OF_SIGHT_CHECK and dist < VISION_DISTANCE:
            has_los = self._check_line_of_sight(bot_pos, pred_pos, walls)
            if has_los:
                return self._combat_mode(bot, pred_pos, dist)
        
        # 5. 优先级3：战术寻路（包抄或直接追击）
        return self._tactical_pathfinding(bot, target_pos, walls)
    
    def _predict_target(self, target, bot_pos, dist):
        """增强的目标预判"""
        target_pos = target.rect.center
        
        if dist <= 0:
            return target_pos
        
        # 计算子弹飞行时间
        time_hit = (dist / BULLET_SPEED) * PREDICTION_FACTOR
        
        # 基于目标速度预判
        pred_x = target_pos[0] + target.vx * time_hit
        pred_y = target_pos[1] + target.vy * time_hit
        
        # 边界限制
        pred_x = max(TANK_SIZE, min(SCREEN_WIDTH - TANK_SIZE, pred_x))
        pred_y = max(TANK_SIZE, min(SCREEN_HEIGHT - TANK_SIZE, pred_y))
        
        return (pred_x, pred_y)
    
    def _check_dodge(self, bot, bullets):
        """检测并躲避来袭子弹"""
        bot_pos = bot.rect.center
        
        for bullet in bullets:
            # 跳过自己的子弹
            if bullet.owner_id == bot.id:
                continue
            
            bullet_pos = bullet.rect.center
            dist_to_bullet = math.hypot(
                bullet_pos[0] - bot_pos[0],
                bullet_pos[1] - bot_pos[1]
            )
            
            if dist_to_bullet > DODGE_DISTANCE:
                continue
            
            # 计算子弹轨迹是否会命中
            if self._will_bullet_hit(bot_pos, bullet):
                # 选择躲避方向（垂直于子弹方向）
                bullet_angle = math.atan2(bullet.dy, bullet.dx)
                
                # 交替左右躲避，避免来回摇摆
                if self.last_dodge_dir == 0:
                    self.last_dodge_dir = 1 if random.random() > 0.5 else -1
                
                dodge_angle = bullet_angle + (math.pi / 2) * self.last_dodge_dir
                bot_angle_rad = math.radians(bot.angle)
                
                # 计算需要的动作
                angle_diff = self._normalize_angle(
                    math.degrees(dodge_angle) - bot.angle
                )
                
                # 优先后退躲避（更快）
                if abs(angle_diff) > 90:
                    return 2  # 后退
                elif abs(angle_diff) < 45:
                    return 1  # 前进
                else:
                    return 3 if angle_diff > 0 else 4  # 旋转
        
        self.last_dodge_dir = 0  # 重置躲避方向
        return None
    
    def _will_bullet_hit(self, bot_pos, bullet):
        """预测子弹是否会命中"""
        # 简化检测：计算子弹未来位置与 bot 的最近距离
        bullet_pos = bullet.rect.center
        
        # 计算子弹方向向量
        bullet_dir = math.sqrt(bullet.dx**2 + bullet.dy**2)
        if bullet_dir == 0:
            return False
        
        # 子弹到 bot 的向量
        to_bot = (bot_pos[0] - bullet_pos[0], bot_pos[1] - bullet_pos[1])
        
        # 点积判断子弹是否朝向 bot
        dot = (to_bot[0] * bullet.dx + to_bot[1] * bullet.dy) / bullet_dir
        if dot < 0:
            return False  # 子弹远离 bot
        
        # 计算最近距离（投影）
        closest_dist = abs(to_bot[0] * bullet.dy - to_bot[1] * bullet.dx) / bullet_dir
        
        return closest_dist < TANK_SIZE * 1.5
    
    def _combat_mode(self, bot, pred_pos, dist):
        """增强战斗模式：瞄准射击 + 战术移动"""
        bot_pos = bot.rect.center
        angle_to_target = math.degrees(
            math.atan2(-(pred_pos[1] - bot_pos[1]), pred_pos[0] - bot_pos[0])
        )
        angle_diff = self._normalize_angle(angle_to_target - bot.angle)
        
        # 清空寻路
        self.current_path = []
        
        # 瞄准精度根据距离调整
        adjusted_tolerance = ANGLE_TOLERANCE * (1 + dist / VISION_DISTANCE)
        
        # 已瞄准
        if abs(angle_diff) < adjusted_tolerance:
            if bot.cooldown == 0:
                return 5  # 射击
            
            # 等待冷却时保持移动（更难被命中）
            if STRAFE_WHILE_SHOOTING and dist > 100:
                # 随机前进或后退
                return 1 if random.random() > 0.3 else 2
            return 0
        
        # 需要旋转瞄准
        # 如果角度差很大，可以边移动边旋转
        if abs(angle_diff) > 45 and dist > 150:
            # 50% 概率边移动边旋转
            if random.random() > 0.5:
                return 3 if angle_diff > 0 else 4
            return 1  # 前进靠近
        
        return 3 if angle_diff > 0 else 4  # 旋转
    
    def _tactical_pathfinding(self, bot, target_pos, walls):
        """战术寻路：包抄或直接追击"""
        bot_pos = bot.rect.center
        
        # 转换为网格坐标
        gx_bot, gy_bot = self.grid_map.pixel_to_grid(*bot_pos)
        gx_target, gy_target = self.grid_map.pixel_to_grid(*target_pos)
        
        # 决定是否包抄
        should_flank = (
            random.random() < FLANK_PROBABILITY and 
            self.steps % (PATHFINDING_UPDATE_FREQ * 3) == 0
        )
        
        if should_flank:
            flank_pos = self._calculate_flank_position(bot_pos, target_pos)
            if flank_pos:
                gx_flank, gy_flank = self.grid_map.pixel_to_grid(*flank_pos)
                self.flank_target = (gx_flank, gy_flank)
        
        # 定期更新路径
        if self.steps % PATHFINDING_UPDATE_FREQ == 0 or not self.current_path:
            goal = self.flank_target if self.flank_target else (gx_target, gy_target)
            self.current_path = self.pathfinder.find_path(
                (gx_bot, gy_bot), goal
            )
            
            # 包抄目标到达后清除
            if self.flank_target and len(self.current_path) < 3:
                self.flank_target = None
        
        # 沿路径移动
        if self.current_path:
            return self._follow_path(bot, bot_pos)
        
        return 0
    
    def _calculate_flank_position(self, bot_pos, target_pos):
        """计算包抄位置"""
        # 计算目标侧翼位置
        dx = target_pos[0] - bot_pos[0]
        dy = target_pos[1] - bot_pos[1]
        
        # 垂直于直线方向的偏移
        perp_x = -dy
        perp_y = dx
        
        # 归一化
        length = math.hypot(perp_x, perp_y)
        if length == 0:
            return None
        
        perp_x /= length
        perp_y /= length
        
        # 选择一侧包抄
        offset = 100 * (1 if random.random() > 0.5 else -1)
        flank_x = target_pos[0] + perp_x * offset
        flank_y = target_pos[1] + perp_y * offset
        
        # 边界检查
        flank_x = max(TANK_SIZE * 2, min(SCREEN_WIDTH - TANK_SIZE * 2, flank_x))
        flank_y = max(TANK_SIZE * 2, min(SCREEN_HEIGHT - TANK_SIZE * 2, flank_y))
        
        return (flank_x, flank_y)
    
    def _follow_path(self, bot, bot_pos):
        """沿寻路路径移动"""
        if not self.current_path:
            return 0
        
        # 获取下一个路点
        next_grid = self.current_path[0]
        next_pixel = self.grid_map.grid_to_pixel(*next_grid)
        
        dist_to_node = math.hypot(next_pixel[0] - bot_pos[0], next_pixel[1] - bot_pos[1])
        
        # 到达该节点
        if dist_to_node < NODE_ARRIVAL_DISTANCE:
            self.current_path.pop(0)
            if not self.current_path:
                return 0
            next_grid = self.current_path[0]
            next_pixel = self.grid_map.grid_to_pixel(*next_grid)
        
        # 导航向下一个路点
        move_angle = math.degrees(
            math.atan2(-(next_pixel[1] - bot_pos[1]), next_pixel[0] - bot_pos[0])
        )
        move_diff = self._normalize_angle(move_angle - bot.angle)
        
        # 移动决策（更灵活的转向）
        if abs(move_diff) > 30:
            return 3 if move_diff > 0 else 4  # 旋转
        elif abs(move_diff) > 10:
            # 小角度时可以边走边转
            if random.random() > 0.7:
                return 3 if move_diff > 0 else 4
            return 1
        else:
            return 1  # 前进
    
    def _check_line_of_sight(self, p1, p2, walls):
        """检查两点之间是否有直线视线"""
        return not any(w.rect.clipline(p1, p2) for w in walls)
    
    def _normalize_angle(self, angle):
        """归一化角度到 [-180, 180] 范围"""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle
