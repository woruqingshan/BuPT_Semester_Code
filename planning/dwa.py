import math
import numpy as np

def world_to_grid(x, y, grid_h, grid_w, map_size_m):
    i = int(y * grid_h / map_size_m)  # 行 = y方向
    j = int(x * grid_w / map_size_m)  # 列 = x方向
    return i, j

def grid_to_world(i, j, grid_h, grid_w, map_size_m):
    x = j * map_size_m / grid_w
    y = i * map_size_m / grid_h
    return x, y

class Config:
    def __init__(self, robot_radius=0.1):
        self.max_speed = 2.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.07  # [s] Time tick for motion prediction
        self.predict_time = 1.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_radius = robot_radius  # [m] for collision check
        self.soft_inflate_dist = 0.2  # [m] soft inflation distance
        self.soft_inflate_penalty = 10.0  # penalty gain for soft inflation

"""def dwa_control(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)
    return u, trajectory"""

def dwa_control(x, config, goal, occupancy_grid, resolution):
    dw = calc_dynamic_window(x, config)
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, occupancy_grid, resolution)
    return u, trajectory

def dwa_control_optimized(x, config, goal, ob, safety_margin=0.2, stuck_threshold=0.1):
    """
    优化的DWA控制，包含安全检查和卡住检测
    """
    # 安全检查：如果目标点太接近障碍物，调整目标
    if len(ob) > 0:
        goal_array = np.array(goal)
        distances = np.linalg.norm(ob - goal_array, axis=1)
        min_distance = np.min(distances)
        
        if min_distance < config.robot_radius + safety_margin:
            # 目标点不安全，寻找更安全的目标
            safe_goal = find_safe_goal(x, goal, ob, config, safety_margin)
            if safe_goal is not None:
                goal = safe_goal
    
    # 标准DWA控制
    dw = calc_dynamic_window(x, config)
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)
    
    # 安全检查：确保控制输出不会导致碰撞
    u = apply_safety_control(u, x, ob, config, safety_margin)
    
    return u, trajectory

def find_safe_goal(robot_pose, original_goal, obstacles, config, safety_margin):
    """
    寻找安全的目标点
    """
    if len(obstacles) == 0:
        return original_goal
    
    robot_pos = np.array([robot_pose[0], robot_pose[1]])
    original_goal_array = np.array(original_goal)
    
    # 在机器人周围寻找安全点
    for angle in np.linspace(0, 2*np.pi, 16):
        for distance in [1.0, 2.0, 3.0]:
            candidate = robot_pos + distance * np.array([np.cos(angle), np.sin(angle)])
            
            # 检查候选点是否安全
            distances = np.linalg.norm(obstacles - candidate, axis=1)
            if len(distances) > 0 and np.min(distances) > config.robot_radius + safety_margin:
                # 选择最接近原目标的安全点
                dist_to_original = np.linalg.norm(candidate - original_goal_array)
                if dist_to_original < 5.0:  # 限制最大偏移
                    return candidate.tolist()
    
    return None

def apply_safety_control(u, robot_pose, obstacles, config, safety_margin):
    """
    应用安全控制，防止碰撞
    """
    if len(obstacles) == 0:
        return u
    
    robot_pos = np.array([robot_pose[0], robot_pose[1]])
    distances = np.linalg.norm(obstacles - robot_pos, axis=1)
    min_distance = np.min(distances)
    
    # 如果太接近障碍物，降低速度
    if min_distance < config.robot_radius + safety_margin:
        u[0] *= 0.3  # 大幅降低线速度
        u[1] *= 0.5  # 降低角速度
    
    # 如果非常接近障碍物，停止前进
    if min_distance < config.robot_radius + 0.1:
        u[0] = 0.0  # 停止前进
        u[1] *= 0.2  # 大幅降低转向速度
    
    return u

def motion(x, u, dt):
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_dynamic_window(x, config):
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw

def predict_trajectory(x_init, v, y, config):
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt
    return trajectory

"""def calc_control_and_trajectory(x, dw, config, goal, ob):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trajectory = predict_trajectory(x_init, v, y, config)
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)
            final_cost = to_goal_cost + speed_cost + ob_cost
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory
"""

def calc_control_and_trajectory(x, dw, config, goal, occupancy_grid, resolution):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trajectory = predict_trajectory(x_init, v, y, config)
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, occupancy_grid, resolution, config)

            final_cost = to_goal_cost + speed_cost + ob_cost
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate

    return best_u, best_trajectory



"""
def calc_obstacle_cost(trajectory, ob, config):
    if ob.shape[0] == 0:
        return 0.0
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)
    
    # 硬碰撞检测：如果轨迹点与障碍物距离小于机器人半径，直接返回无穷大
    if np.array(r <= config.robot_radius).any():
        return float("Inf")
    
    min_r = np.min(r)
    
    # 软膨胀机制：当距离小于soft_inflate_dist时，大幅增加惩罚
    soft_penalty = 0.0
    if min_r < config.soft_inflate_dist:
        # 距离越近，惩罚越大
        soft_penalty = config.soft_inflate_penalty * (config.soft_inflate_dist - min_r) / config.soft_inflate_dist
    
    # 基础障碍物成本 + 软膨胀惩罚
    base_cost = 1.0 / min_r
    return base_cost + soft_penalty
"""

def calc_obstacle_cost(trajectory, occupancy_grid, resolution, config):
    height, width = occupancy_grid.shape
    for x, y in trajectory[:, :2]:
        # grid_x = int(x / resolution)
        # grid_y = int(y / resolution)
        grid_y, grid_x = world_to_grid(x, y, height, width, resolution * height)

        if grid_x < 0 or grid_x >= width or grid_y < 0 or grid_y >= height:
            return float("inf")  # 越界，视为碰撞

        if occupancy_grid[grid_y, grid_x] >= 1:  # 1表示有障碍
            return float("inf")  # 撞墙

    return 0.0  # 无碰撞


def calc_to_goal_cost(trajectory, goal):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    return cost