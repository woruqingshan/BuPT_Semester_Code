import numpy as np
from planning.dwa import dwa_control, Config
from simulation.maze_generator import generate_random_maze, segments_to_obstacles
from simulation.slam_simulator import SLAMSimulator
from utils.gui import GUIVisualizer

def motion(x, u, dt):
    """机器人运动模型"""
    x[2] += u[1] * dt
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

# 生成迷宫
width = 100
height = 100
num_segments = 20
segments = generate_random_maze(width, height, num_segments)
# 将像素坐标转换为米坐标（假设100像素 = 32米）
scale_factor = 32.0 / 100.0
obstacles = segments_to_obstacles(segments, scale_factor)

print(f"Generated maze with {len(obstacles)} obstacles")
print(f"Obstacles shape: {obstacles.shape if len(obstacles) > 0 else 'Empty'}")
if len(obstacles) > 0:
    print(f"Obstacle range: x=[{obstacles[:, 0].min():.2f}, {obstacles[:, 0].max():.2f}], "
          f"y=[{obstacles[:, 1].min():.2f}, {obstacles[:, 1].max():.2f}]")

# 初始化SLAM模拟器和可视化
slam = SLAMSimulator()
viz = GUIVisualizer(slam.MAP_SIZE_PIXELS, slam.MAP_SIZE_METERS, "SLAM + DWA Navigation")

# 起点和目标点
start = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, theta, v, omega]，单位米和弧度
goal = np.array([80.0, 80.0])  # 单位米

config = Config()
current_state = start.copy()
max_iterations = 1000
iteration = 0

print(f"Starting navigation from {start[:2]} to {goal}")
print(f"Robot radius: {config.robot_radius}")

while obstacles.shape[0] == 0:
    segments = generate_random_maze(width, height, num_segments)
    obstacles = segments_to_obstacles(segments, scale_factor)

while iteration < max_iterations:
    iteration += 1

    # 获取SLAM估计的当前位置（米, 米, 角度）
    slam_pos = slam.get_position()  # [x, y, theta(deg)]
    # DWA需要弧度，保持当前速度和角速度
    current_state[:3] = [slam_pos[0], slam_pos[1], np.deg2rad(slam_pos[2])]

    # DWA路径规划
    u, trajectory = dwa_control(current_state, config, goal, obstacles)

    # 更新机器人状态
    current_state = motion(current_state, u, config.dt)

    # 生成激光数据
    sensor_data = slam.simulate_laser_scan()
    # 计算位姿变化（单位：毫米、度、秒）
    pose_change = (u[0] * config.dt * 1000, np.rad2deg(u[1] * config.dt), config.dt)
    slam.slam.update(sensor_data, pose_change)

    # 可视化
    x_m, y_m, theta_deg = slam.get_position()
    mapbytes = slam.get_map()
    if not viz.display(x_m, y_m, theta_deg, mapbytes):
        print("Visualization window closed")
        break

    # 判断是否到达目标
    dist_to_goal = np.hypot(current_state[0] - goal[0], current_state[1] - goal[1])
    if dist_to_goal <= config.robot_radius:
        print(f"Goal reached at iteration {iteration}!")
        print(f"Final position: ({current_state[0]:.2f}, {current_state[1]:.2f})")
        break

    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Position = ({current_state[0]:.2f}, {current_state[1]:.2f}), "
              f"Velocity = ({current_state[3]:.2f}, {current_state[4]:.2f}), "
              f"Distance to goal = {dist_to_goal:.2f}")

if iteration >= max_iterations:
    print("Maximum iterations reached without reaching goal")
    print(f"Final position: ({current_state[0]:.2f}, {current_state[1]:.2f})")