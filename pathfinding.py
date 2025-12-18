"""
寻路算法模块
实现 A* 网格寻路和网格管理
"""

import math
import heapq
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
    
    def get_neighbors(self, grid_x, grid_y, allow_diagonal=True):
        """获取可行走的相邻格子"""
        neighbors = []
        # 4方向
        directions_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # 8方向（包含对角线）
        directions_8 = directions_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        directions = directions_8 if allow_diagonal else directions_4
        
        for dx, dy in directions:
            nx, ny = grid_x + dx, grid_y + dy
            if self.is_walkable(nx, ny):
                # 对角线移动需要检查相邻两个格子是否可走（防止穿墙角）
                if dx != 0 and dy != 0:
                    if not self.is_walkable(grid_x + dx, grid_y) or not self.is_walkable(grid_x, grid_y + dy):
                        continue
                neighbors.append((nx, ny))
        
        return neighbors


class AStarPathfinder:
    """A* 寻路器"""
    
    def __init__(self, grid_map):
        """
        初始化寻路器
        grid_map: GridMap 实例
        """
        self.grid_map = grid_map
    
    def heuristic(self, a, b):
        """
        启发式函数：使用对角线距离（Chebyshev距离）
        比欧几里得距离更适合8方向移动
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        # 对角线移动代价为 √2 ≈ 1.414
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
    
    def find_path(self, start_grid, end_grid):
        """
        使用 A* 算法从起点到终点寻找路径
        start_grid, end_grid: (grid_x, grid_y) 网格坐标
        返回: 路径列表，如果无路返回空列表
        """
        # 如果起点不可走，找最近的可走点
        if not self.grid_map.is_walkable(start_grid[0], start_grid[1]):
            start_grid = self._find_nearest_walkable(start_grid)
            if start_grid is None:
                return []
        
        # 如果终点在墙里，寻找最近的可走点
        if not self.grid_map.is_walkable(end_grid[0], end_grid[1]):
            end_grid = self._find_nearest_walkable(end_grid)
            if end_grid is None:
                return []
        
        # A* 搜索
        # 优先队列: (f_score, counter, node) - counter用于打破平局
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, end_grid)}
        
        open_set_hash = {start_grid}  # 快速查找节点是否在开放列表中
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            
            if current == end_grid:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for neighbor in self.grid_map.get_neighbors(current[0], current[1]):
                # 计算移动代价（对角线 √2，直线 1）
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = math.sqrt(2) if (dx + dy == 2) else 1
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, end_grid)
                    f_score[neighbor] = f
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor))
                        open_set_hash.add(neighbor)
        
        # 无路可走
        return []
    
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


# 保留 BFSPathfinder 作为备用
class BFSPathfinder:
    """BFS 寻路器（备用）"""
    
    def __init__(self, grid_map):
        self.grid_map = grid_map
    
    def find_path(self, start_grid, end_grid):
        if not self.grid_map.is_walkable(end_grid[0], end_grid[1]):
            end_grid = self._find_nearest_walkable(end_grid)
            if end_grid is None:
                return []
        
        queue = deque([start_grid])
        came_from = {start_grid: None}
        
        while queue:
            current = queue.popleft()
            if current == end_grid:
                break
            
            for neighbor in self.grid_map.get_neighbors(current[0], current[1]):
                if neighbor not in came_from:
                    queue.append(neighbor)
                    came_from[neighbor] = current
        
        if end_grid not in came_from:
            return []
        
        path = []
        curr = end_grid
        while curr is not None:
            path.append(curr)
            curr = came_from[curr]
        
        path.reverse()
        return path[1:] if len(path) > 1 else []
    
    def _find_nearest_walkable(self, grid_pos, search_radius=5):
        gx, gy = grid_pos
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if self.grid_map.is_walkable(nx, ny):
                        return (nx, ny)
        return None
