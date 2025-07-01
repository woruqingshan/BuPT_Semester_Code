import numpy as np
import json
from utils.gui import DualMapVisualizer
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser
from planning.dwa import dwa_control, Config
from planning.a_star import AStarPlanner
from simulation.slam_simulator import SLAMSimulator
from planning.a_star_dwa import AStarDWAPlanner
import scipy.ndimage


'''
1.exsting the coordinate system problem like x denote row or y denote row
2.the soft inflation distance and penalty need to be considered and modify the a_star_dwa.py
'''



# =================== 全局参数和变量初始化 ===================
# 仿真参数
MAP_SIZE_PIXELS = 800
MAP_SIZE_METERS = 25.0
LASER_SCAN_SIZE = 360
LASER_SCAN_RATE_HZ = 10
LASER_DETECTION_ANGLE_DEGREES = 360
LASER_DETECTION_MAX_MM = 10000
LASER_DETECTION_MARGIN = 0
LASER_OFFSET_MM = 0

# 控制参数
max_steps = 2000
reached_goal = False

# 机器人初始状态 [x, y, theta, v, omega]
pose = [0.0, 0.0, 0.0, 0.0, 0.0]

# =================== 读取迷宫和起点 ===================
json_path = 'data/line_segments.json'
visualizer = DualMapVisualizer(map_size_pixels=MAP_SIZE_PIXELS, map_size_meters=MAP_SIZE_METERS, title="Dual Image Display", show_trajectory=True)
visualizer.load_line_segments_from_json(json_path)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
start_point = data.get('start_point', [0, 0])
print(f"起点: {start_point}")
pose[0] = float(start_point[0])
pose[1] = float(start_point[1])

# =================== 生成未膨胀maze_grid ===================
segments = data.get('segments', data.get('line_segments', []))
max_x = max([max(seg['start'][0], seg['end'][0]) for seg in segments])
max_y = max([max(seg['start'][1], seg['end'][1]) for seg in segments])
grid_size = (max_y+2, max_x+2)  # 列(x), 行(y)
raw_maze_grid = np.zeros((grid_size[1], grid_size[0]), dtype=int)  # shape: (rows, cols) = (y, x)

for seg in segments:
    x0, y0 = [int(round(v)) for v in seg['start']]
    x1, y1 = [int(round(v)) for v in seg['end']]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = int(x0), int(y0)
    while True:
        if 0 <= y < grid_size[1] and 0 <= x < grid_size[0]:
            raw_maze_grid[y, x] = 1
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

# =================== 出口查找（用未膨胀maze_grid） ===================

#exist problem of finding the exit
def find_exit(maze_grid, start_grid):
    rows, cols = maze_grid.shape
    for y in range(rows):
        for x in range(cols):
            if (x == 0 or x == cols-1 or y == 0 or y == rows-1):
                if (x, y) != start_grid and maze_grid[y, x] == 0:
                    return (y , x)
    return None

start_grid = (int(round(start_point[1])), int(round(start_point[0])))
exit_grid = find_exit(raw_maze_grid, start_grid)
if exit_grid is None:
    raise RuntimeError("No valid exit found on maze boundary!")
exit_grid_uninflated = exit_grid  # 记录未膨胀出口

print(f"起点: {start_grid}")
print(raw_maze_grid[max(0, start_grid[0]-1):start_grid[0]+2, max(0, start_grid[1]-1):start_grid[1]+2])

# =================== 膨胀maze_grid用于DWA/A* ===================
cell_size = MAP_SIZE_METERS / grid_size[0]
robot_radius = cell_size / 8
config = Config()
config.robot_radius = robot_radius

# 尝试最小化膨胀距离，优先保证可行区域
dis = min(cell_size / 8, 0.1)  # 取更小的，通常为0.1米或更小
inflate_cells = int(np.ceil(dis / cell_size))
if inflate_cells < 1:
    inflate_cells = 0
print(f"膨胀格数: {inflate_cells}")
if inflate_cells > 0:
    structure = np.ones((inflate_cells*2+1, inflate_cells*2+1), dtype=int)
    maze_grid = scipy.ndimage.binary_dilation(raw_maze_grid, structure=structure).astype(int)
else:
    maze_grid = raw_maze_grid.copy()
maze_grid[exit_grid_uninflated] = 0
maze_grid[start_grid] = 0

# 检查可行区域比例，若过少则自动降为0格膨胀
free_ratio = np.sum(maze_grid == 0) / maze_grid.size
if free_ratio < 0.15:  # 可行区域小于15%则不膨胀
    print("膨胀后可行区域过少，自动降为0格膨胀")
    maze_grid = raw_maze_grid.copy()
    maze_grid[exit_grid_uninflated] = 0
    maze_grid[start_grid] = 0

print("==== 未膨胀 raw_maze_grid ====")
for row in raw_maze_grid:
    print(' '.join(str(x) for x in row))

print("\n==== 膨胀后 maze_grid ====")
for row in maze_grid:
    print(' '.join(str(x) for x in row))
    

# =================== 生成障碍点云（膨胀后） ===================
obstacle_points = []
for y in range(maze_grid.shape[0]):
    for x in range(maze_grid.shape[1]):
        if maze_grid[y, x] == 1:
            mx = x * MAP_SIZE_METERS / maze_grid.shape[1]
            my = y * MAP_SIZE_METERS / maze_grid.shape[0]
            obstacle_points.append([mx, my])
obstacles = np.array(obstacle_points)

# =================== SLAM相关初始化 ===================
laser = Laser(LASER_SCAN_SIZE, LASER_SCAN_RATE_HZ, LASER_DETECTION_ANGLE_DEGREES, LASER_DETECTION_MAX_MM, LASER_DETECTION_MARGIN, LASER_OFFSET_MM)
slam = RMHC_SLAM(laser, MAP_SIZE_PIXELS, MAP_SIZE_METERS)
slam_sim = SLAMSimulator(MAP_SIZE_PIXELS, MAP_SIZE_METERS)
slam_sim.set_occupancy_grid(raw_maze_grid)
mapbytes = slam_sim.get_map()

# =================== 强制设置SLAM起点 ===================
try:
    pos = slam_sim.slam.position.copy()
    pos.x_mm = pose[0] * 1000
    pos.y_mm = pose[1] * 1000
    pos.theta_degrees = np.rad2deg(pose[2])  # 这里是角度
    slam_sim.slam.position = pos
    print(f"✅ 强制设置 SLAM 起点为: ({pose[0]}, {pose[1]})")
except Exception as e:
    print(f"❌ 设置 SLAM 起点失败: {e}")

# =================== GUI初始显示 ===================
visualizer.display(x_m=pose[0], y_m=pose[1], theta_deg=np.rad2deg(pose[2]), mapbytes=mapbytes, laser_range=LASER_DETECTION_MAX_MM/1000, laser_scan=None)

# =================== 初始化A*-DWA集成规划器 ===================
astar_dwa_planner = AStarDWAPlanner(maze_grid, config=config, replan_interval=20)

# =================== 主仿真循环 ===================
# 新增：记录真实轨迹
pose_history = [(pose[0], pose[1], np.rad2deg(pose[2]))]

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

# Map exit grid to meters for DWA goal
exit_goal = [exit_grid_uninflated[1] * MAP_SIZE_METERS / maze_grid.shape[1], exit_grid_uninflated[0] * MAP_SIZE_METERS / maze_grid.shape[0]]

def extract_local_obstacles(slam_sim, pose, radius=2.0):
    # 从SLAM地图中提取机器人当前位置半径r内的障碍点
    occ_grid = slam_sim.occupancy_grid
    if occ_grid is None:
        return np.empty((0, 2))
    grid_h, grid_w = occ_grid.shape
    map_size = slam_sim.MAP_SIZE_METERS
    cell_size_x = map_size / grid_w
    cell_size_y = map_size / grid_h
    x0, y0 = pose[0], pose[1]
    local_obs = []
    for i in range(grid_h):
        for j in range(grid_w):
            if occ_grid[i, j] == 1:
                x = j * cell_size_x
                y = i * cell_size_y
                if (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2:
                    local_obs.append([x, y])
    return np.array(local_obs)

for step in range(max_steps):
    # 智能提取局部障碍点（以机器人为中心半径2米内的障碍）
    local_obstacles = extract_local_obstacles(slam_sim, pose, radius=2.0)
    # 使用A*-DWA集成算法规划
    u, trajectory, waypoint = astar_dwa_planner.step(
        pose, exit_grid, obstacles, MAP_SIZE_METERS, maze_grid.shape, local_obstacles=local_obstacles)
    pose = motion(pose, u, config.dt)

    # 新增：记录真实轨迹
    pose_history.append((pose[0], pose[1], np.rad2deg(pose[2])))

    # Simulate laser scan based on maze obstacles and robot pose
    laser_scan = slam_sim.simulate_laser_scan([pose[0], pose[1], pose[2]])
    pose_change = (u[0] * config.dt * 1000, np.rad2deg(u[1] * config.dt), config.dt)
    slam_sim.update(laser_scan, pose_change)
    mapbytes = slam_sim.get_map()

    # 获取轨迹和回环检测结果
    # visited_positions = slam_sim.get_trajectory()  # 不再用SLAM轨迹
    loop_detected = slam_sim.get_last_loop_detected()

    # Visualize robot, SLAM map, and laser scan, 轨迹和回环点
    laser_scan_m = [d/1000.0 for d in laser_scan]
    visualizer.display(x_m=pose[0], y_m=pose[1], theta_deg=np.rad2deg(pose[2]), mapbytes=mapbytes, laser_range=LASER_DETECTION_MAX_MM/1000, laser_scan=laser_scan_m, trajectory=pose_history, loop_detected=loop_detected)

    # robot_grid计算也用[y, x]顺序
    robot_grid = (int(round(pose[1] * raw_maze_grid.shape[0] / MAP_SIZE_METERS)), int(round(pose[0] * raw_maze_grid.shape[1] / MAP_SIZE_METERS)))
    if is_exit(robot_grid, start_grid, raw_maze_grid):  # 用未膨胀grid判断出口
        print(f"Goal (exit) reached at step {step}! Final position: ({pose[0]:.2f}, {pose[1]:.2f})")
        reached_goal = True
        break
    if step % 20 == 0:
        print(f"Step {step}: Position=({pose[0]:.2f}, {pose[1]:.2f}), Waypoint={waypoint}")

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



