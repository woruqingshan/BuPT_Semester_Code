'''
Testing the obstacle and slam array

1. read the json file
2. generate the maze_grid
3. visualize the maze_grid and the json segments

The result is good, as the Figure2 shows.

The obstacle slam map of maze_grid and the json segments map are the same.

'''


import matplotlib.pyplot as plt
import numpy as np
import json

print("start")

# 1. 读取json文件
with open('data/line_segments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
segments = data.get('segments', data.get('line_segments', []))
start_point = data.get('start_point', [0, 0])

# 2. 生成maze_grid
max_x = max([max(seg['start'][0], seg['end'][0]) for seg in segments])
max_y = max([max(seg['start'][1], seg['end'][1]) for seg in segments])
grid_size = (max_x+2, max_y+2)
maze_grid = np.zeros(grid_size, dtype=int)
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

# 3. 可视化maze_grid和json线段
plt.figure(figsize=(8,8))
plt.imshow(maze_grid.T, cmap='gray_r', origin='lower', alpha=0.5)  # 注意.T转置，保证坐标一致
for seg in segments:
    x0, y0 = seg['start']
    x1, y1 = seg['end']
    plt.plot([x0, x1], [y0, y1], 'r-', linewidth=2, alpha=0.7, label='Segment' if seg==segments[0] else "")
plt.scatter([start_point[0]], [start_point[1]], c='g', s=100, label='Start')
plt.title('maze_grid (gray) & json segments (red)')
plt.legend()
plt.grid(True)
plt.show()

print("end")