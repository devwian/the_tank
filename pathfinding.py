"""
寻路算法模块
实现 BFS 网格寻路和网格管理
"""

import numpy as np
from collections import deque
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, GRID_SIZE, GRID_BUFFER_RADIUS


class GridMap:
    """网格地图和寻路管理"""
    
    def __init__(self):
        self.grid_cols = SCREEN_WIDTH // GRID_SIZE
        self.grid_rows = SCREEN_HEIGHT // GRID_SIZE
        self.grid_map = np.zeros((self.grid_cols, self.grid_rows), dtype=int)
    
    def init_from_walls(self, walls):
        """
        从墙壁列表初始化网格地图
        包含缓冲区膨胀处理，避免坦克卡角
        """
        # 第一步：标记真实的墙壁
        temp_map = np.zeros_like(self.grid_map)
        for wall in walls:
            x_start = max(0, wall.rect.x // GRID_SIZE)
            x_end = min(self.grid_cols, (wall.rect.right) // GRID_SIZE + 1)
            y_start = max(0, wall.rect.y // GRID_SIZE)
            y_end = min(self.grid_rows, (wall.rect.bottom) // GRID_SIZE + 1)
            temp_map[x_start:x_end, y_start:y_end] = 1
        
        # 第二步：应用膨胀（缓冲区处理）
        # 将墙壁周围的格子也标记为不可走
        self.grid_map = temp_map.copy()
        for x in range(self.grid_cols):
            for y in range(self.grid_rows):
                if temp_map[x][y] == 1:
                    for dx in range(-GRID_BUFFER_RADIUS, GRID_BUFFER_RADIUS + 1):
                        for dy in range(-GRID_BUFFER_RADIUS, GRID_BUFFER_RADIUS + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                                self.grid_map[nx][ny] = 1
    
    def is_walkable(self, grid_x, grid_y):
        """检查格子是否可行走"""
        if not (0 <= grid_x < self.grid_cols and 0 <= grid_y < self.grid_rows):
            return False
        return self.grid_map[grid_x][grid_y] == 0
    
    def pixel_to_grid(self, px, py):
        """像素坐标转网格坐标"""
        return (int(px // GRID_SIZE), int(py // GRID_SIZE))
    
    def grid_to_pixel(self, gx, gy):
        """网格坐标转像素坐标（返回格子中心）"""
        return (gx * GRID_SIZE + GRID_SIZE // 2, gy * GRID_SIZE + GRID_SIZE // 2)


class BFSPathfinder:
    """BFS 寻路器"""
    
    def __init__(self, grid_map):
        """
        初始化寻路器
        grid_map: GridMap 实例
        """
        self.grid_map = grid_map
    
    def find_path(self, start_grid, end_grid):
        """
        使用 BFS 算法从起点到终点寻找路径
        start_grid, end_grid: (grid_x, grid_y) 网格坐标
        返回: 路径列表，如果无路返回空列表
        """
        # 如果终点在墙里，寻找最近的可走点
        if not self.grid_map.is_walkable(end_grid[0], end_grid[1]):
            end_grid = self._find_nearest_walkable(end_grid)
            if end_grid is None:
                return []
        
        # BFS 搜索
        queue = deque([start_grid])
        came_from = {start_grid: None}
        
        while queue:
            current = queue.popleft()
            if current == end_grid:
                break
            
            cx, cy = current
            # 4 方向搜索（不用 8 方向避免穿墙角）
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if self.grid_map.is_walkable(nx, ny) and (nx, ny) not in came_from:
                    queue.append((nx, ny))
                    came_from[(nx, ny)] = current
        
        # 重建路径
        if end_grid not in came_from:
            return []
        
        path = []
        curr = end_grid
        while curr is not None:
            path.append(curr)
            curr = came_from[curr]
        
        path.reverse()
        return path[1:] if len(path) > 1 else []  # 排除起点
    
    def _find_nearest_walkable(self, grid_pos, search_radius=5):
        """
        螺旋搜索找到最近的可行走格子
        """
        gx, gy = grid_pos
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if self.grid_map.is_walkable(nx, ny):
                        return (nx, ny)
        return None
