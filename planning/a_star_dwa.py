import numpy as np
from .a_star import AStarPlanner
from .dwa import dwa_control, Config

class AStarDWAPlanner:
    def __init__(self, maze_grid, config=None, replan_interval=10):
        # maze_grid应为膨胀后的grid，仅用于DWA/A*，其余逻辑外部保证
        self.maze_grid = maze_grid
        self.config = config if config is not None else Config()
        self.replan_interval = replan_interval
        self.path = []
        self.path_idx = 0
        self.last_goal = None
        self.steps_since_replan = 0

    def plan_path(self, start_grid, goal_grid):
        planner = AStarPlanner(self.maze_grid)
        path = planner.planning(start_grid, goal_grid)
        self.path = path if path is not None else []
        self.path_idx = 0
        return self.path

    def get_next_waypoint(self, robot_grid):
        # 距离机器人最近的路径点作为下一个waypoint
        if not self.path:
            return None
        dists = [np.linalg.norm(np.array(robot_grid) - np.array(p)) for p in self.path]
        min_idx = np.argmin(dists)
        # 向前lookahead，避免卡在当前点
        lookahead = min(min_idx + 3, len(self.path) - 1)
        return self.path[lookahead]

    def grid_to_meter(self, grid, map_size_m, grid_shape):
        # grid: (x, y)
        return [grid[0] * map_size_m / grid_shape[0], grid[1] * map_size_m / grid_shape[1]]

    def step(self, pose, goal_grid, obstacles, map_size_m, grid_shape, local_obstacles=None):
        # pose: [x, y, theta, v, omega] (meters)
        # goal_grid: (x, y) in膨胀后grid坐标
        # obstacles: np.array([[x, y], ...]) in meters (全局障碍)
        # local_obstacles: np.array([[x, y], ...]) in meters (局部障碍)
        # map_size_m: float
        # grid_shape: (cols, rows) of膨胀后grid
        robot_grid = (int(round(pose[0] * grid_shape[0] / map_size_m)), int(round(pose[1] * grid_shape[1] / map_size_m)))
        # 路径重规划条件：到达终点、路径为空、路径被阻断、定期重规划
        need_replan = False
        if self.last_goal != goal_grid:
            need_replan = True
        elif not self.path or self.path_idx >= len(self.path):
            need_replan = True
        elif self.steps_since_replan >= self.replan_interval:
            need_replan = True
        elif self.path and self.maze_grid[self.path[-1][0], self.path[-1][1]] == 1:
            need_replan = True
        if need_replan:
            self.plan_path(robot_grid, goal_grid)
            self.last_goal = goal_grid
            self.steps_since_replan = 0
        else:
            self.steps_since_replan += 1
        # 获取下一个waypoint
        waypoint_grid = self.get_next_waypoint(robot_grid)
        if waypoint_grid is None:
            return [0.0, 0.0], None, None
        waypoint_m = self.grid_to_meter(waypoint_grid, map_size_m, grid_shape)
        # DWA控制，优先用local_obstacles
        use_obstacles = local_obstacles if local_obstacles is not None else obstacles
        u, trajectory = dwa_control(pose, self.config, waypoint_m, use_obstacles)
        return u, trajectory, waypoint_m 