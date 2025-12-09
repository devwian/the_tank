import pygame
import math

# --- 初始化 ---
pygame.init()

# --- 常量定义 ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("坦克动荡 (Tank Trouble) - Python版")

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GRAY = (100, 100, 100)

# 游戏参数
TANK_SIZE = 30
TANK_SPEED = 3
ROTATION_SPEED = 3
BULLET_SPEED = 5
MAX_BOUNCES = 5
BULLET_COOLDOWN = 30  # 帧数

# --- 类定义 ---

class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(GRAY)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, color):
        super().__init__()
        self.image = pygame.Surface([6, 6])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.speed = BULLET_SPEED
        # 将角度转换为弧度计算速度分量
        rad = math.radians(angle)
        self.dx = math.cos(rad) * self.speed
        self.dy = -math.sin(rad) * self.speed # Pygame Y轴向下，所以取负
        self.bounces = 0
        self.max_bounces = MAX_BOUNCES

    def update(self, walls):
        # X轴移动
        self.rect.x += self.dx
        hit_list = pygame.sprite.spritecollide(self, walls, False)
        if hit_list:
            self.dx *= -1 # 反弹
            self.bounces += 1
            # 防止卡墙，稍微回退
            self.rect.x += self.dx 

        # Y轴移动
        self.rect.y += self.dy
        hit_list = pygame.sprite.spritecollide(self, walls, False)
        if hit_list:
            self.dy *= -1 # 反弹
            self.bounces += 1
            # 防止卡墙
            self.rect.y += self.dy

        # 检查是否超出屏幕或达到最大反弹次数
        if (self.rect.x < 0 or self.rect.x > SCREEN_WIDTH or 
            self.rect.y < 0 or self.rect.y > SCREEN_HEIGHT or 
            self.bounces > self.max_bounces):
            self.kill()

class Tank(pygame.sprite.Sprite):
    def __init__(self, x, y, color, controls):
        super().__init__()
        self.original_image = pygame.Surface([TANK_SIZE, TANK_SIZE], pygame.SRCALPHA)
        # 绘制坦克身体
        pygame.draw.rect(self.original_image, color, (0, 0, TANK_SIZE, TANK_SIZE))
        # 绘制炮塔指示方向 (黑色小方块)
        pygame.draw.rect(self.original_image, BLACK, (TANK_SIZE - 8, TANK_SIZE//2 - 4, 8, 8))
        
        self.image = self.original_image
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        
        self.color = color
        self.angle = 0 # 0度向右
        self.speed = TANK_SPEED
        self.controls = controls # 字典包含按键配置
        self.cooldown = 0
        
        # 用于碰撞检测的回退机制
        self.old_rect = self.rect.copy()

    def update(self, walls, bullets_group, all_sprites):
        keys = pygame.key.get_pressed()
        self.old_rect = self.rect.copy()
        
        # 旋转
        if keys[self.controls['left']]:
            self.angle += ROTATION_SPEED
        if keys[self.controls['right']]:
            self.angle -= ROTATION_SPEED
            
        # 计算移动向量
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.speed
        dy = -math.sin(rad) * self.speed
        
        # 移动
        if keys[self.controls['up']]:
            self.rect.x += dx
            self.rect.y += dy
        if keys[self.controls['down']]:
            self.rect.x -= dx
            self.rect.y -= dy
            
        # 旋转图像
        self.rotate()
        
        # 墙壁碰撞检测
        if pygame.sprite.spritecollide(self, walls, False):
            self.rect = self.old_rect # 撞墙则回退
            
        # 射击
        if self.cooldown > 0:
            self.cooldown -= 1
        if keys[self.controls['shoot']] and self.cooldown == 0:
            self.shoot(bullets_group, all_sprites)

    def rotate(self):
        # 保持中心点旋转
        old_center = self.rect.center
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = old_center

    def shoot(self, bullets_group, all_sprites):
        # 在坦克前方生成子弹
        rad = math.radians(self.angle)
        spawn_dist = TANK_SIZE // 1.5
        bx = self.rect.centerx + math.cos(rad) * spawn_dist
        by = self.rect.centery - math.sin(rad) * spawn_dist
        
        bullet = Bullet(bx, by, self.angle, self.color)
        bullets_group.add(bullet)
        all_sprites.add(bullet)
        self.cooldown = BULLET_COOLDOWN

# --- 迷宫生成 ---
def create_maze():
    walls = pygame.sprite.Group()
    # 四周墙壁
    walls.add(Wall(0, 0, SCREEN_WIDTH, 10)) # 上
    walls.add(Wall(0, SCREEN_HEIGHT-10, SCREEN_WIDTH, 10)) # 下
    walls.add(Wall(0, 0, 10, SCREEN_HEIGHT)) # 左
    walls.add(Wall(SCREEN_WIDTH-10, 0, 10, SCREEN_HEIGHT)) # 右
    
    # 内部障碍物 (简单的示例布局)
    walls.add(Wall(150, 150, 10, 300))
    walls.add(Wall(640, 150, 10, 300))
    walls.add(Wall(300, 100, 200, 10))
    walls.add(Wall(300, 500, 200, 10))
    walls.add(Wall(395, 250, 10, 100))
    
    return walls

# --- 主程序 ---
def main():
    clock = pygame.time.Clock()
    running = True
    game_over = False
    winner_text = ""

    # 精灵组
    all_sprites = pygame.sprite.Group()
    walls = create_maze()
    bullets = pygame.sprite.Group()
    tanks = pygame.sprite.Group()

    # 玩家1 (红色): WASD移动, 空格射击
    p1_controls = {
        'up': pygame.K_w, 'down': pygame.K_s,
        'left': pygame.K_a, 'right': pygame.K_d,
        'shoot': pygame.K_SPACE
    }
    player1 = Tank(100, 100, RED, p1_controls)
    
    # 玩家2 (蓝色): 方向键移动, 回车射击
    p2_controls = {
        'up': pygame.K_UP, 'down': pygame.K_DOWN,
        'left': pygame.K_LEFT, 'right': pygame.K_RIGHT,
        'shoot': pygame.K_RETURN
    }
    player2 = Tank(700, 500, BLUE, p2_controls)
    player2.angle = 180 # 初始朝左

    all_sprites.add(walls)
    all_sprites.add(player1)
    all_sprites.add(player2)
    tanks.add(player1)
    tanks.add(player2)

    font = pygame.font.SysFont('Arial', 48)
    small_font = pygame.font.SysFont('Arial', 24)

    while running:
        # 1. 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # 游戏结束后按 R 重置
            if game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main() # 简单粗暴的重启
                    return

        if not game_over:
            # 2. 更新逻辑
            player1.update(walls, bullets, all_sprites)
            player2.update(walls, bullets, all_sprites)
            bullets.update(walls)

            # 检测子弹击中坦克
            for bullet in bullets:
                hit_tanks = pygame.sprite.spritecollide(bullet, tanks, False)
                for tank in hit_tanks:
                    game_over = True
                    bullet.kill()
                    if tank == player1:
                        winner_text = "BLUE WINS!"
                    else:
                        winner_text = "RED WINS!"

        # 3. 绘制画面
        screen.fill(WHITE)
        all_sprites.draw(screen)

        if game_over:
            text_surf = font.render(winner_text, True, BLACK)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            screen.blit(text_surf, text_rect)
            
            restart_surf = small_font.render("Press 'R' to Restart", True, BLACK)
            restart_rect = restart_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
            screen.blit(restart_surf, restart_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()