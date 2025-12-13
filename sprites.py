"""
游戏对象（精灵）模块
定义 Wall, Bullet, Tank 等游戏实体
"""

import pygame
import math
from constants import (
    TANK_SIZE, BULLET_SIZE, TANK_SPEED, ROTATION_SPEED, BULLET_SPEED,
    MAX_BOUNCES, BULLET_COOLDOWN, MAX_BULLETS_PER_TANK, SCREEN_WIDTH, SCREEN_HEIGHT,
    RED, BLUE, GRAY, BLACK
)


class Wall(pygame.sprite.Sprite):
    """墙壁对象"""
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(GRAY)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Bullet(pygame.sprite.Sprite):
    """子弹对象"""
    def __init__(self, x, y, angle, owner_id):
        super().__init__()
        self.image = pygame.Surface([BULLET_SIZE, BULLET_SIZE])
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
        """更新子弹位置，处理墙壁碰撞"""
        # X 方向移动
        self.rect.x += self.dx
        hit_list = pygame.sprite.spritecollide(self, walls, False)
        if hit_list:
            self.dx *= -1
            self.bounces += 1
            self.rect.x += self.dx
        
        # Y 方向移动
        self.rect.y += self.dy
        hit_list = pygame.sprite.spritecollide(self, walls, False)
        if hit_list:
            self.dy *= -1
            self.bounces += 1
            self.rect.y += self.dy
        
        # 检查是否超出边界或反弹次数过多
        if (self.rect.x < 0 or self.rect.x > SCREEN_WIDTH or 
            self.rect.y < 0 or self.rect.y > SCREEN_HEIGHT or 
            self.bounces > self.max_bounces):
            self.kill()


class Tank(pygame.sprite.Sprite):
    """坦克对象"""
    def __init__(self, x, y, color, tank_id):
        super().__init__()
        self.id = tank_id
        self.color = color
        
        # 创建坦克图像（正方形 + 炮塔指示）
        self.original_image = pygame.Surface([TANK_SIZE, TANK_SIZE], pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, color, (0, 0, TANK_SIZE, TANK_SIZE))
        pygame.draw.rect(self.original_image, BLACK, (TANK_SIZE - 8, TANK_SIZE//2 - 4, 8, 8))
        
        self.image = self.original_image
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        
        # 运动状态
        self.angle = 0
        self.vx = 0  # 速度用于预判
        self.vy = 0
        self.last_pos = (x, y)
        self.old_rect = self.rect.copy()
        
        # 射击冷却
        self.cooldown = 0

    def update_velocity(self):
        """更新速度（用于目标预判）"""
        self.vx = self.rect.centerx - self.last_pos[0]
        self.vy = self.rect.centery - self.last_pos[1]
        self.last_pos = (self.rect.centerx, self.rect.centery)

    def act(self, action, walls, bullets_group, all_sprites):
        """
        执行动作
        action: 0=待命, 1=前进, 2=后退, 3=顺时针旋转, 4=逆时针旋转, 5=射击
        返回值: True 如果本轮射击，False 否则
        """
        self.old_rect = self.rect.copy()
        rotated = False
        moved = False
        
        # 旋转
        if action == 3:
            self.angle += ROTATION_SPEED
            rotated = True
        elif action == 4:
            self.angle -= ROTATION_SPEED
            rotated = True
        
        # 移动
        rad = math.radians(self.angle)
        dx = math.cos(rad) * TANK_SPEED
        dy = -math.sin(rad) * TANK_SPEED
        
        if action == 1:
            self.rect.x += dx
            self.rect.y += dy
            moved = True
        elif action == 2:
            self.rect.x -= dx
            self.rect.y -= dy
            moved = True
        
        # 碰撞检测
        if rotated:
            self.rotate()
        if moved and pygame.sprite.spritecollide(self, walls, False):
            self.rect = self.old_rect
        
        # 射击冷却
        if self.cooldown > 0:
            self.cooldown -= 1
        
        # 射击
        if action == 5 and self.cooldown == 0:
            self.shoot(bullets_group, all_sprites)
            return True
        
        return False

    def rotate(self):
        """旋转坦克图像"""
        old_center = self.rect.center
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = old_center

    def shoot(self, bullets_group, all_sprites):
        """发射子弹"""
        # 检查当前子弹数是否达到上限
        current_bullets = sum(1 for b in bullets_group if b.owner_id == self.id)
        if current_bullets >= MAX_BULLETS_PER_TANK:
            return
        
        rad = math.radians(self.angle)
        bx = self.rect.centerx + math.cos(rad) * (TANK_SIZE / 1.5)
        by = self.rect.centery - math.sin(rad) * (TANK_SIZE / 1.5)
        
        bullet = Bullet(bx, by, self.angle, self.id)
        bullets_group.add(bullet)
        all_sprites.add(bullet)
        self.cooldown = BULLET_COOLDOWN
