import numpy as np
from scipy import ndimage
from .dwa import dwa_control, Config
from .a_star import AStarPlanner

class FrontierExplorationDWA:
    """
    结合Frontier Exploration和优化DWA的路径规划系统
    适用于未知地图的探索和导航
    """
    
    def __init__(self, map_size_meters=25.0, map_size_pixels=800, robot_radius=0.1):
        self.map_size_meters = map_size_meters
        self.map_size_pixels = map_size_pixels
        self.robot_radius = robot_radius
        
        # DWA配置优化
        self.config = Config(robot_radius=robot_radius)
        self.config.soft_inflate_dist = 0.5  # 软膨胀距离
        self.config.soft_inflate_penalty = 15.0  # 软膨胀惩罚
        
        # 探索状态
        self.exploration_grid = None  # 探索状态网格 (0:未探索, 1:已探索, 2:障碍物)
        self.frontiers = []  # 前沿点列表
        self.current_goal = None
        self.exploration_complete = False
        
        # 优化机制参数
        self.stuck_threshold = 0.1  # 卡住阈值(米)
        self.stuck_steps = 0
        self.max_stuck_steps = 50
        self.last_positions = []  # 记录最近位置用于检测卡住
        
        # 预警机制参数
        self.safety_margin = 0.2  # 安全边距(米)
        self.collision_warning = False
        self.boundary_warning = False
        
        # A*路径规划器(用于返回路径)
        self.astar_planner = None
        self.return_path = None
        self.return_path_index = 0
        
    def initialize_exploration(self, slam_simulator):
        """
        初始化探索系统
        """
        # 获取SLAM地图尺寸
        if hasattr(slam_simulator, 'occupancy_grid') and slam_simulator.occupancy_grid is not None:
            grid_shape = slam_simulator.occupancy_grid.shape
            self.exploration_grid = np.zeros(grid_shape, dtype=int)
            self.astar_planner = AStarPlanner(slam_simulator.occupancy_grid)
        else:
            # 如果没有occupancy_grid，使用默认尺寸
            grid_size = int(self.map_size_meters / 0.5)  # 0.5米分辨率
            self.exploration_grid = np.zeros((grid_size, grid_size), dtype=int)
            self.astar_planner = AStarPlanner(np.zeros((grid_size, grid_size), dtype=int))
        
        print(f"✅ 探索系统初始化完成，网格尺寸: {self.exploration_grid.shape}")
    
    def update_exploration_status(self, slam_simulator, robot_pose):
        """
        更新探索状态网格
        """
        if slam_simulator.occupancy_grid is None:
            return
        
        # 获取SLAM地图
        slam_map = slam_simulator.occupancy_grid
        grid_h, grid_w = slam_map.shape
        
        # 计算机器人网格位置
        robot_grid_x = int(round(robot_pose[0] * grid_w / self.map_size_meters))
        robot_grid_y = int(round(robot_pose[1] * grid_h / self.map_size_meters))
        
        # 更新探索状态
        # 将机器人周围区域标记为已探索
        explore_radius = 3  # 探索半径(网格单位)
        for dy in range(-explore_radius, explore_radius + 1):
            for dx in range(-explore_radius, explore_radius + 1):
                y = robot_grid_y + dy
                x = robot_grid_x + dx
                if 0 <= y < grid_h and 0 <= x < grid_w:
                    if slam_map[y, x] == 1:
                        self.exploration_grid[y, x] = 2  # 障碍物
                    elif self.exploration_grid[y, x] == 0:
                        self.exploration_grid[y, x] = 1  # 已探索
    
    def find_frontiers(self, robot_pose):
        """
        找到前沿点(已探索区域与未探索区域的边界)
        """
        if self.exploration_grid is None:
            return []
        
        grid_h, grid_w = self.exploration_grid.shape
        
        # 使用形态学操作找到前沿
        # 已探索区域
        explored = (self.exploration_grid == 1).astype(np.uint8)
        # 未探索区域
        unexplored = (self.exploration_grid == 0).astype(np.uint8)
        
        # 膨胀已探索区域 (使用scipy替代cv2)
        kernel = np.ones((3, 3), dtype=int)
        explored_dilated = ndimage.binary_dilation(explored, structure=kernel).astype(np.uint8)
        
        # 前沿 = 膨胀后的已探索区域 ∩ 未探索区域
        frontiers = explored_dilated * unexplored
        
        # 找到前沿点坐标
        frontier_points = []
        for y in range(grid_h):
            for x in range(grid_w):
                if frontiers[y, x] == 1:
                    # 转换为物理坐标
                    mx = x * self.map_size_meters / grid_w
                    my = y * self.map_size_meters / grid_h
                    frontier_points.append([mx, my])
        
        return frontier_points
    
    def select_best_frontier(self, frontiers, robot_pose):
        """
        选择最佳前沿点作为目标
        考虑距离、前沿大小、方向等因素
        """
        if not frontiers or len(frontiers) == 0:
            return None
        
        frontiers = np.array(frontiers)
        robot_pos = np.array([robot_pose[0], robot_pose[1]])
        
        # 计算到每个前沿点的距离
        distances = np.linalg.norm(frontiers - robot_pos, axis=1)
        
        # 计算前沿大小(周围未探索区域的数量)
        frontier_sizes = []
        for frontier in frontiers:
            size = self.calculate_frontier_size(frontier)
            frontier_sizes.append(size)
        
        frontier_sizes = np.array(frontier_sizes)
        
        # 综合评分: 距离越近、前沿越大越好
        # 归一化
        if len(distances) > 1:
            distances_norm = (distances - distances.min()) / (distances.max() - distances.min())
            sizes_norm = (frontier_sizes - frontier_sizes.min()) / (frontier_sizes.max() - frontier_sizes.min())
        else:
            distances_norm = np.array([0.0])
            sizes_norm = np.array([1.0])
        
        # 综合评分 (距离权重0.6，前沿大小权重0.4)
        scores = 0.6 * (1 - distances_norm) + 0.4 * sizes_norm
        
        # 选择得分最高的前沿点
        best_idx = np.argmax(scores)
        return frontiers[best_idx].tolist()
    
    def calculate_frontier_size(self, frontier_point):
        """
        计算前沿点周围未探索区域的大小
        """
        if self.exploration_grid is None:
            return 1
        
        grid_h, grid_w = self.exploration_grid.shape
        
        # 转换为网格坐标
        grid_x = int(round(frontier_point[0] * grid_w / self.map_size_meters))
        grid_y = int(round(frontier_point[1] * grid_h / self.map_size_meters))
        
        # 计算周围未探索区域的数量
        size = 0
        radius = 5
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y = grid_y + dy
                x = grid_x + dx
                if 0 <= y < grid_h and 0 <= x < grid_w:
                    if self.exploration_grid[y, x] == 0:  # 未探索
                        size += 1
        
        return size
    
    def check_stuck_condition(self, robot_pose):
        """
        检查机器人是否卡住
        """
        self.last_positions.append([robot_pose[0], robot_pose[1]])
        
        # 只保留最近的位置
        if len(self.last_positions) > 20:
            self.last_positions.pop(0)
        
        # 检查是否卡住
        if len(self.last_positions) >= 10:
            recent_positions = self.last_positions[-10:]
            start_pos = recent_positions[0]
            end_pos = recent_positions[-1]
            distance_moved = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            
            if distance_moved < self.stuck_threshold:
                self.stuck_steps += 1
                if self.stuck_steps > self.max_stuck_steps:
                    return True
            else:
                self.stuck_steps = 0
        
        return False
    
    def check_safety_warnings(self, robot_pose, obstacles, start_point=None, step_count=0):
        """
        检查安全警告 - 增加缓启动机制和起点豁免
        """
        self.collision_warning = False
        self.boundary_warning = False
        
        # 缓启动：前10步不触发警告，让机器人有机会从起点挣脱
        if step_count is not None and step_count < 10:
            return
        
        # 检查与障碍物的距离
        if len(obstacles) > 0:
            robot_pos = np.array([robot_pose[0], robot_pose[1]])
            distances = np.linalg.norm(obstacles - robot_pos, axis=1)
            min_distance = np.min(distances)
            
            if min_distance < self.robot_radius + self.safety_margin:
                self.collision_warning = True
                print(f"⚠️ 碰撞警告: 距离障碍物 {min_distance:.2f}m")
        
        # 检查边界 - 起点豁免，或使用更小的阈值
        if start_point is not None and np.allclose([robot_pose[0], robot_pose[1]], start_point, atol=0.05):
            return  # 起点附近豁免边界警告
        
        # 使用更小的边界警告阈值，避免过早触发
        boundary_margin = 0.02  # 只有距离边界2cm才警告
        if (robot_pose[0] < boundary_margin or 
            robot_pose[0] > self.map_size_meters - boundary_margin or
            robot_pose[1] < boundary_margin or 
            robot_pose[1] > self.map_size_meters - boundary_margin):
            self.boundary_warning = True
            print(f"⚠️ 边界警告: 接近地图边界 ({robot_pose[0]:.2f}, {robot_pose[1]:.2f})")
    
    def get_exploration_goal(self, slam_simulator, robot_pose):
        """
        获取探索目标点
        """
        # 更新探索状态
        self.update_exploration_status(slam_simulator, robot_pose)
        
        # 找到前沿点
        frontiers = self.find_frontiers(robot_pose)
        
        if frontiers:
            # 选择最佳前沿点
            goal = self.select_best_frontier(frontiers, robot_pose)
            return goal
        else:
            # 没有前沿点，探索完成
            self.exploration_complete = True
            return None
    
    def plan_return_path(self, robot_pose, start_point):
        """
        规划返回起点的路径
        """
        if self.astar_planner is None or self.exploration_grid is None:
            return None
        
        # 转换为网格坐标
        grid_h, grid_w = self.exploration_grid.shape
        robot_grid = (int(round(robot_pose[1] * grid_h / self.map_size_meters)), 
                     int(round(robot_pose[0] * grid_w / self.map_size_meters)))
        start_grid = (int(round(start_point[1] * grid_h / self.map_size_meters)), 
                     int(round(start_point[0] * grid_w / self.map_size_meters)))
        
        # 使用A*规划路径
        path = self.astar_planner.planning(robot_grid, start_grid)
        
        if path:
            # 转换为物理坐标
            return_path = []
            for grid_pos in path:
                x = grid_pos[1] * self.map_size_meters / grid_w
                y = grid_pos[0] * self.map_size_meters / grid_h
                return_path.append([x, y])
            
            self.return_path = return_path
            self.return_path_index = 0
            print(f"✅ 生成返回路径，共 {len(return_path)} 个点")
            return return_path
        else:
            print("❌ 无法生成返回路径")
            return None
    
    def get_return_goal(self):
        """
        获取返回路径中的下一个目标点
        """
        if self.return_path is None or self.return_path_index >= len(self.return_path):
            return None
        
        goal = self.return_path[self.return_path_index]
        return goal
    
    def step(self, robot_pose, slam_simulator, obstacles, start_point=None, return_mode=False, step_count=0):
        """
        执行一步探索或返回
        """
        # 检查安全警告 - 传入step_count和start_point
        self.check_safety_warnings(robot_pose, obstacles, start_point, step_count)
        
        # 检查是否卡住
        is_stuck = self.check_stuck_condition(robot_pose)
        
        # 获取目标点
        if return_mode:
            if self.return_path is None:
                self.plan_return_path(robot_pose, start_point)
            goal = self.get_return_goal()
        else:
            goal = self.get_exploration_goal(slam_simulator, robot_pose)
        
        if goal is None:
            return [0.0, 0.0], None, None, "no_goal"
        
        # 提取局部障碍物
        local_obstacles = self.extract_local_obstacles(slam_simulator, robot_pose)
        
        # 如果有安全警告，调整目标点
        if self.collision_warning or self.boundary_warning:
            goal = self.adjust_goal_for_safety(robot_pose, goal, obstacles)
        
        # 如果卡住，尝试新的目标点
        if is_stuck:
            print("🔄 检测到卡住，重新选择目标点")
            if return_mode:
                self.return_path_index += 1
                goal = self.get_return_goal()
            else:
                # 在探索模式下，选择不同的前沿点
                frontiers = self.find_frontiers(robot_pose)
                if frontiers:
                    goal = self.select_best_frontier(frontiers, robot_pose)
        
        # DWA控制
        u, trajectory = dwa_control(robot_pose, self.config, goal, local_obstacles)
        
        # 检查控制输出是否安全
        if self.collision_warning or self.boundary_warning:
            u = self.apply_safety_control(u, robot_pose, obstacles)
        
        status = "exploring" if not return_mode else "returning"
        if is_stuck:
            status += "_stuck"
        if self.collision_warning:
            status += "_collision_warning"
        if self.boundary_warning:
            status += "_boundary_warning"
        
        return u, trajectory, goal, status
    
    def extract_local_obstacles(self, slam_simulator, robot_pose, radius=3.0):
        """
        从SLAM地图中提取局部障碍物
        """
        if slam_simulator.occupancy_grid is None:
            return np.empty((0, 2))
        
        occ_grid = slam_simulator.occupancy_grid
        grid_h, grid_w = occ_grid.shape
        cell_size_x = self.map_size_meters / grid_w
        cell_size_y = self.map_size_meters / grid_h
        
        x0, y0 = robot_pose[0], robot_pose[1]
        local_obs = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                if occ_grid[i, j] == 1:  # 障碍物
                    x = j * cell_size_x
                    y = i * cell_size_y
                    if (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2:
                        local_obs.append([x, y])
        
        return np.array(local_obs)
    
    def adjust_goal_for_safety(self, robot_pose, goal, obstacles):
        """
        为安全考虑调整目标点
        """
        if len(obstacles) == 0:
            return goal
        
        robot_pos = np.array([robot_pose[0], robot_pose[1]])
        goal_pos = np.array(goal)
        
        # 检查目标点是否安全
        distances = np.linalg.norm(obstacles - goal_pos, axis=1)
        min_distance = np.min(distances)
        
        if min_distance < self.robot_radius + self.safety_margin:
            # 目标点不安全，寻找更安全的目标
            safe_goals = []
            for angle in np.linspace(0, 2*np.pi, 8):
                for dist in [1.0, 2.0, 3.0]:
                    new_goal = robot_pos + dist * np.array([np.cos(angle), np.sin(angle)])
                    
                    # 检查新目标是否在地图范围内
                    if (0 <= new_goal[0] <= self.map_size_meters and 
                        0 <= new_goal[1] <= self.map_size_meters):
                        
                        # 检查新目标是否安全
                        distances = np.linalg.norm(obstacles - new_goal, axis=1)
                        if len(distances) > 0 and np.min(distances) > self.robot_radius + self.safety_margin:
                            safe_goals.append(new_goal)
            
            if safe_goals:
                # 选择最接近原目标的安全目标
                safe_goals = np.array(safe_goals)
                distances_to_original = np.linalg.norm(safe_goals - goal_pos, axis=1)
                best_idx = np.argmin(distances_to_original)
                return safe_goals[best_idx].tolist()
        
        return goal
    
    def apply_safety_control(self, u, robot_pose, obstacles):
        """
        应用安全控制，防止碰撞 - 更温和的控制策略
        """
        # 只减速，不直接停止，保证有最小速度
        u[0] = max(u[0] * 0.7, 0.05)  # 保证至少有0.05m/s的速度，减速程度更温和
        
        # 如果接近障碍物，进一步减速但不停
        if len(obstacles) > 0:
            robot_pos = np.array([robot_pose[0], robot_pose[1]])
            distances = np.linalg.norm(obstacles - robot_pos, axis=1)
            min_distance = np.min(distances)
            
            if min_distance < self.robot_radius + 0.03:  # 只有极近（3cm）才停止
                u[0] = 0.0  # 停止前进
                u[1] *= 0.5  # 降低转向速度
            elif min_distance < self.robot_radius + 0.1:  # 10cm内进一步减速
                u[0] = max(u[0] * 0.5, 0.02)  # 进一步减速但保持最小速度
            elif min_distance < self.robot_radius + 0.2:  # 20cm内轻微减速
                u[0] = max(u[0] * 0.8, 0.03)  # 轻微减速
        
        return u
    
    def check_goal_reached(self, robot_pose, goal, threshold=0.5):
        """
        检查是否到达目标点
        """
        if goal is None:
            return False
        
        distance = np.sqrt((robot_pose[0] - goal[0])**2 + (robot_pose[1] - goal[1])**2)
        return distance < threshold
    
    def get_exploration_progress(self):
        """
        获取探索进度
        """
        if self.exploration_grid is None:
            return 0.0
        
        total_cells = self.exploration_grid.size
        explored_cells = np.sum(self.exploration_grid == 1)
        return explored_cells / total_cells
    
    def is_exploration_complete(self):
        """
        检查探索是否完成
        """
        return self.exploration_complete 