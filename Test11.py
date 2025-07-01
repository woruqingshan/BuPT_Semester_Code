import numpy as np
import json
from utils.gui import DualMapVisualizer
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser
from planning.dwa import dwa_control, Config
from planning.a_star import AStarPlanner
from simulation.slam_simulator import SLAMSimulator

def is_exit(robot_grid, start_grid, maze_grid):
    rows, cols = maze_grid.shape
    is_on_boundary = (robot_grid[0] == 0 or robot_grid[0] == rows-1 or robot_grid[1] == 0 or robot_grid[1] == cols-1)
    not_start = (robot_grid != start_grid)
    is_free = (maze_grid[robot_grid] == 0)
    return is_on_boundary and is_free and not_start

def motion(x, u, dt):
    x[2] += u[1] * dt
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

# 1. 读取json迷宫
json_path = 'data/line_segments.json'
visualizer = DualMapVisualizer(map_size_pixels=800, map_size_meters=15.0, title="双图显示", show_trajectory=True)
visualizer.load_line_segments_from_json(json_path)

# 2. 解析起点
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
start_point = data.get('start_point', [0, 0])
print(f"起点: {start_point}")

# 3. 初始化SLAM、DWA、机器人状态
MAP_SIZE_PIXELS = 800
MAP_SIZE_METERS = 15.0
LASER_SCAN_SIZE = 360
LASER_SCAN_RATE_HZ = 10
LASER_DETECTION_ANGLE_DEGREES = 360
LASER_DETECTION_MAX_MM = 10000
LASER_DETECTION_MARGIN = 0
LASER_OFFSET_MM = 0

laser = Laser(LASER_SCAN_SIZE, LASER_SCAN_RATE_HZ, LASER_DETECTION_ANGLE_DEGREES, LASER_DETECTION_MAX_MM, LASER_DETECTION_MARGIN, LASER_OFFSET_MM)
slam = RMHC_SLAM(laser, MAP_SIZE_PIXELS, MAP_SIZE_METERS)

# 生成A*用的栅格地图（0=可通行, 1=障碍）
segments = data.get('segments', data.get('line_segments', []))
max_x = max([max(seg['start'][0], seg['end'][0]) for seg in segments])
max_y = max([max(seg['start'][1], seg['end'][1]) for seg in segments])
grid_size = (max_x+2, max_y+2)
maze_grid = np.zeros(grid_size, dtype=int)
for seg in segments:
    x0, y0 = [int(round(v)) for v in seg['start']]
    x1, y1 = [int(round(v)) for v in seg['end']]
    # Bresenham画线
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = int(x0), int(y0)
    while True:
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
            maze_grid[x, y] = 1
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

# Initialize SLAMSimulator for SLAM and laser simulation
slam_sim = SLAMSimulator(MAP_SIZE_PIXELS, MAP_SIZE_METERS)
slam_sim.set_occupancy_grid(maze_grid)
mapbytes = slam_sim.get_map()

# 机器人初始状态 [x, y, theta, v, omega]
pose = [float(start_point[0]), float(start_point[1]), 0.0, 0.0, 0.0]
#pose = [3.0, 0.0, 0.0, 0.0, 0.0]
config = Config()
config.robot_radius = 0.8
#设置 SLAM 起点（第一次 update()前）
#slam_sim.force_initial_position(pose[0], pose[1], pose[2])

# 设置 SLAM 内部真实初始位姿（必须在第一次 update 前）
try:
    pos = slam_sim.slam.position.copy()
    pos.x_mm = pose[0] * 1000
    pos.y_mm = pose[1] * 1000
    pos.theta_degrees = np.rad2deg(pose[2])  # 这里是角度
    slam_sim.slam.position = pos
    print(f"✅ 强制设置 SLAM 起点为: ({pose[0]}, {pose[1]})")
except Exception as e:
    print(f"❌ 设置 SLAM 起点失败: {e}")

# Generate obstacle points (in meters) for DWA and visualization
obstacle_points = []
for i in range(maze_grid.shape[0]):
    for j in range(maze_grid.shape[1]):
        if maze_grid[i, j] == 1:
            # Map grid (i, j) to meters (x, y)
            x = i * MAP_SIZE_METERS / maze_grid.shape[0]
            y = j * MAP_SIZE_METERS / maze_grid.shape[1]
            obstacle_points.append([x, y])
obstacles = np.array(obstacle_points)

max_steps = 2000
reached_goal = False
start_grid = (int(round(start_point[0])), int(round(start_point[1])))

# Find a valid exit cell (on boundary, not start, and free)
def find_exit(maze_grid, start_grid):
    rows, cols = maze_grid.shape
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows-1 or j == 0 or j == cols-1):
                if (i, j) != start_grid and maze_grid[i, j] == 0:
                    return (i, j)
    return None
exit_grid = find_exit(maze_grid, start_grid)
if exit_grid is None:
    raise RuntimeError("No valid exit found on maze boundary!")
# Map exit grid to meters for DWA goal
exit_goal = [exit_grid[0] * MAP_SIZE_METERS / maze_grid.shape[0], exit_grid[1] * MAP_SIZE_METERS / maze_grid.shape[1]]

# Initial display (no obstacles argument, pass None for laser_scan for now)
visualizer.display(x_m=pose[0], y_m=pose[1], theta_deg=np.rad2deg(pose[2]), mapbytes=mapbytes, laser_range=LASER_DETECTION_MAX_MM/1000, laser_scan=None)
for step in range(max_steps):
    # Use DWA to move toward the exit, avoiding obstacles
    u, _ = dwa_control(pose, config, exit_goal, obstacles)
    pose = motion(pose, u, config.dt)
    # Simulate laser scan based on maze obstacles and robot pose
    
    
    laser_scan = slam_sim.simulate_laser_scan([pose[0], pose[1], pose[2]])
    pose_change = (u[0] * config.dt * 1000, np.rad2deg(u[1] * config.dt), config.dt)
    slam_sim.update(laser_scan, pose_change)
    mapbytes = slam_sim.get_map()


    # Visualize robot, SLAM map, and laser scan
    laser_scan_m = [d/1000.0 for d in laser_scan]
    visualizer.display(x_m=pose[0], y_m=pose[1], theta_deg=np.rad2deg(pose[2]), mapbytes=mapbytes, laser_range=LASER_DETECTION_MAX_MM/1000, laser_scan=laser_scan_m)
    # Convert robot pose to grid for exit check
    robot_grid = (int(round(pose[0] * maze_grid.shape[0] / MAP_SIZE_METERS)), int(round(pose[1] * maze_grid.shape[1] / MAP_SIZE_METERS)))
    if is_exit(robot_grid, start_grid, maze_grid):
        print(f"Goal (exit) reached at step {step}! Final position: ({pose[0]:.2f}, {pose[1]:.2f})")
        reached_goal = True
        break
    if step % 50 == 0:
        print(f"Step {step}: Position=({pose[0]:.2f}, {pose[1]:.2f})")

if reached_goal:
    print("A* path planning to return to start...")
    goal_grid = robot_grid
    planner = AStarPlanner(maze_grid)
    path = planner.planning(goal_grid, start_grid)
    if path is None:
        print("A* could not find a valid path!")
    else:
        print(f"A* path length: {len(path)}")
        for idx, node in enumerate(path):
            # Map grid node to meters for visualization
            pose[0] = node[0] * MAP_SIZE_METERS / maze_grid.shape[0]
            pose[1] = node[1] * MAP_SIZE_METERS / maze_grid.shape[1]
            visualizer.display(x_m=pose[0], y_m=pose[1], theta_deg=np.rad2deg(pose[2]), mapbytes=mapbytes, laser_range=LASER_DETECTION_MAX_MM/1000, laser_scan=None)
            if idx % 10 == 0:
                print(f"A* return: reached {node}")
        print("Returned to start!")