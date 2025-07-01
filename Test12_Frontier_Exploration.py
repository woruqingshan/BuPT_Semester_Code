import numpy as np
import json
from utils.gui import DualMapVisualizer
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser
from planning.dwa import dwa_control_optimized, Config
from planning.frontier_exploration_dwa import FrontierExplorationDWA
from simulation.slam_simulator import SLAMSimulator
import scipy.ndimage

"""
Frontier Exploration + 优化DWA 路径规划系统
适用于未知地图的探索和导航

主要特性：
1. Frontier Exploration: 自动找到前沿点进行探索
2. 优化DWA: 包含障碍膨胀、软膨胀、安全检查和卡住检测
3. 实时SLAM: 基于激光扫描的实时地图构建
4. 智能返回: 探索完成后使用A*返回起点
"""

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
max_steps = 3000
reached_goal = False

# 机器人初始状态 [x, y, theta, v, omega]
pose = [0.0, 0.0, 0.0, 0.0, 0.0]

# =================== 读取迷宫和起点 ===================
json_path = 'data/line_segments.json'
visualizer = DualMapVisualizer(map_size_pixels=MAP_SIZE_PIXELS, map_size_meters=MAP_SIZE_METERS, 
                              title="Frontier Exploration + DWA", show_trajectory=True)
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
def find_exit(maze_grid, start_grid):
    rows, cols = maze_grid.shape
    for y in range(rows):
        for x in range(cols):
            if (x == 0 or x == cols-1 or y == 0 or y == rows-1):
                if (x, y) != start_grid and maze_grid[y, x] == 0:
                    return (y, x)
    return None

start_grid = (int(round(start_point[1])), int(round(start_point[0])))
exit_grid = find_exit(raw_maze_grid, start_grid)
if exit_grid is None:
    raise RuntimeError("No valid exit found on maze boundary!")

print(f"起点: {start_grid}")
print(f"出口: {exit_grid}")

# =================== 膨胀maze_grid用于DWA ===================
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

maze_grid[exit_grid] = 0
maze_grid[start_grid] = 0

# 检查可行区域比例，若过少则自动降为0格膨胀
free_ratio = np.sum(maze_grid == 0) / maze_grid.size
if free_ratio < 0.15:  # 可行区域小于15%则不膨胀
    print("膨胀后可行区域过少，自动降为0格膨胀")
    maze_grid = raw_maze_grid.copy()
    maze_grid[exit_grid] = 0
    maze_grid[start_grid] = 0

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
laser = Laser(LASER_SCAN_SIZE, LASER_SCAN_RATE_HZ, LASER_DETECTION_ANGLE_DEGREES, 
              LASER_DETECTION_MAX_MM, LASER_DETECTION_MARGIN, LASER_OFFSET_MM)
slam = RMHC_SLAM(laser, MAP_SIZE_PIXELS, MAP_SIZE_METERS)
slam_sim = SLAMSimulator(MAP_SIZE_PIXELS, MAP_SIZE_METERS)
slam_sim.set_occupancy_grid(raw_maze_grid)
mapbytes = slam_sim.get_map()

# =================== 强制设置SLAM起点 ===================
try:
    pos = slam_sim.slam.position.copy()
    pos.x_mm = pose[0] * 1000
    pos.y_mm = pose[1] * 1000
    pos.theta_degrees = np.rad2deg(pose[2])
    slam_sim.slam.position = pos
    print(f"✅ 强制设置 SLAM 起点为: ({pose[0]}, {pose[1]})")
except Exception as e:
    print(f"❌ 设置 SLAM 起点失败: {e}")

# =================== 初始化Frontier Exploration + DWA系统 ===================
frontier_explorer = FrontierExplorationDWA(
    map_size_meters=MAP_SIZE_METERS,
    map_size_pixels=MAP_SIZE_PIXELS,
    robot_radius=robot_radius
)
frontier_explorer.initialize_exploration(slam_sim)

# =================== GUI初始显示 ===================
visualizer.display(x_m=pose[0], y_m=pose[1], theta_deg=np.rad2deg(pose[2]), 
                  mapbytes=mapbytes, laser_range=LASER_DETECTION_MAX_MM/1000, laser_scan=None)

# =================== 主仿真循环 ===================
# 记录真实轨迹
pose_history = [(pose[0], pose[1], np.rad2deg(pose[2]))]

# 状态变量
exploration_phase = True  # True: 探索阶段, False: 返回阶段
current_goal = None
return_path_index = 0

def is_exit(robot_grid, start_grid, maze_grid):
    """检查是否到达出口"""
    rows, cols = maze_grid.shape
    is_on_boundary = (robot_grid[0] == 0 or robot_grid[0] == rows-1 or 
                     robot_grid[1] == 0 or robot_grid[1] == cols-1)
    not_start = (robot_grid != start_grid)
    is_free = (maze_grid[robot_grid] == 0)
    return is_on_boundary and is_free and not_start

def motion(x, u, dt):
    """机器人运动模型"""
    x[2] += u[1] * dt
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

print("🚀 开始Frontier Exploration + DWA仿真...")
print("=" * 60)

for step in range(max_steps):
    # 获取当前目标点
    if exploration_phase:
        # 探索阶段：使用Frontier Exploration
        current_goal = frontier_explorer.get_exploration_goal(slam_sim, pose)
        
        # 检查是否到达出口
        robot_grid = (int(round(pose[1] * raw_maze_grid.shape[0] / MAP_SIZE_METERS)), 
                     int(round(pose[0] * raw_maze_grid.shape[1] / MAP_SIZE_METERS)))
        if is_exit(robot_grid, start_grid, raw_maze_grid):
            print(f"🎯 找到出口！位置: ({pose[0]:.2f}, {pose[1]:.2f})")
            exploration_phase = False
            current_goal = None
    else:
        # 返回阶段：使用A*路径
        if frontier_explorer.return_path is None:
            frontier_explorer.plan_return_path(pose, start_point)
        current_goal = frontier_explorer.get_return_goal()
    
    # 如果没有目标点，结束仿真
    if current_goal is None:
        if exploration_phase and frontier_explorer.is_exploration_complete():
            print("✅ 探索完成，开始返回起点...")
            exploration_phase = False
            frontier_explorer.plan_return_path(pose, start_point)
            current_goal = frontier_explorer.get_return_goal()
            if current_goal is None:
                print("❌ 无法生成返回路径")
                break
        elif not exploration_phase:
            print("✅ 已返回起点！")
            break
        else:
            print("❌ 无法获取目标点")
            break
    
    # 使用优化的DWA控制
    u, trajectory, goal, status = frontier_explorer.step(
        robot_pose=pose,
        slam_simulator=slam_sim,
        obstacles=obstacles,
        start_point=start_point,
        return_mode=not exploration_phase
    )
    
    # 更新机器人状态
    pose = motion(pose, u, config.dt)
    
    # 记录轨迹
    pose_history.append((pose[0], pose[1], np.rad2deg(pose[2])))
    
    # 更新SLAM
    laser_scan = slam_sim.simulate_laser_scan([pose[0], pose[1], pose[2]])
    pose_change = (u[0] * config.dt * 1000, np.rad2deg(u[1] * config.dt), config.dt)
    slam_sim.update(laser_scan, pose_change)
    mapbytes = slam_sim.get_map()
    
    # 检查是否到达当前目标
    if current_goal is not None:
        goal_distance = np.sqrt((pose[0] - current_goal[0])**2 + (pose[1] - current_goal[1])**2)
        if goal_distance < 0.5:  # 0.5米阈值
            if exploration_phase:
                print(f"📍 到达探索目标点: ({current_goal[0]:.2f}, {current_goal[1]:.2f})")
            else:
                return_path_index += 1
                if frontier_explorer.return_path is not None:
                    print(f"📍 到达返回路径点 {return_path_index}/{len(frontier_explorer.return_path)}")
                else:
                    print(f"📍 到达返回路径点 {return_path_index}")
    
    # 获取回环检测结果
    loop_detected = slam_sim.get_last_loop_detected()
    
    # 更新GUI
    if step % 10 == 0:  # 每10步更新一次GUI
        laser_scan_m = [d/1000.0 for d in laser_scan]
        visualizer.display(
            x_m=pose[0], y_m=pose[1], theta_deg=np.rad2deg(pose[2]),
            mapbytes=mapbytes, laser_range=LASER_DETECTION_MAX_MM/1000,
            laser_scan=laser_scan_m, trajectory=pose_history,
            loop_detected=loop_detected
        )
    
    # 打印状态信息
    if step % 50 == 0:
        progress = frontier_explorer.get_exploration_progress()
        phase_str = "探索" if exploration_phase else "返回"
        print(f"步骤 {step}: {phase_str}阶段 | 位置=({pose[0]:.2f}, {pose[1]:.2f}) | "
              f"目标=({current_goal[0]:.2f}, {current_goal[1]:.2f}) | "
              f"探索进度={progress:.1%} | 状态={status}")

print("=" * 60)
print(f"🎉 仿真结束，总步数: {step}")
print(f"📍 最终位置: ({pose[0]:.2f}, {pose[1]:.2f})")
print(f"📊 轨迹点数: {len(pose_history)}")
print(f"🗺️ 探索进度: {frontier_explorer.get_exploration_progress():.1%}")

# 显示最终结果
laser_scan_m = [d/1000.0 for d in laser_scan]
visualizer.display(
    x_m=pose[0], y_m=pose[1], theta_deg=np.rad2deg(pose[2]),
    mapbytes=mapbytes, laser_range=LASER_DETECTION_MAX_MM/1000,
    laser_scan=laser_scan_m, trajectory=pose_history,
    loop_detected=loop_detected
) 