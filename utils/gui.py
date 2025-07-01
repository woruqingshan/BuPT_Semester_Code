# utils/gui.py
import matplotlib.pyplot as plt
import numpy as np

class DualMapVisualizer:
    def __init__(self, map_size_pixels=800, map_size_meters=20.0, title="Dual Map", show_trajectory=False):
        # GUI只显示未膨胀的障碍和轨迹
        self.map_size_pixels = map_size_pixels
        self.map_size_meters = map_size_meters
        self.show_trajectory = show_trajectory
        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle(title, fontsize=16)
        plt.ion()
        plt.show()
        self._maze_segments = []
        self._start_point = None
        self._slam_img_artist = None
        self._robot_left = None
        self._robot_right = None

    def load_line_segments_from_json(self, json_file_path):
        try:
            import json
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            segments = data.get('segments', data.get('line_segments', []))
            start_point = data.get('start_point', [0, 0])
            max_x = max_y = 0
            for segment in segments:
                max_x = max(max_x, segment['start'][0], segment['end'][0])
                max_y = max(max_y, segment['start'][1], segment['end'][1])
            grid_size = [max_x + 2, max_y + 2]
            self.ax_left.clear()
            for segment in segments:
                x0, y0 = segment['start']
                x1, y1 = segment['end']
                self.ax_left.plot([x0, x1], [y0, y1], color='black', linewidth=6)
            self.ax_left.plot(start_point[0], start_point[1], 'o', color='green', markersize=16, label='Start')
            self.ax_left.set_xlim([0, grid_size[0]])
            self.ax_left.set_ylim([0, grid_size[1]])
            self.ax_left.set_title("Maze Segments", fontsize=12)
            self.ax_left.set_xlabel('X')
            self.ax_left.set_ylabel('Y')
            self.ax_left.legend(loc='upper right')
            self.ax_left.set_aspect('equal')
            self.ax_left.grid(False)
            self._maze_segments = segments
            self._start_point = start_point
            print(f"✅ Loaded {len(segments)} maze segments from JSON.")
            print(f"📍 Start point: {start_point}")
            print(f"📐 Grid size: {grid_size}")
            return True
        except Exception as e:
            print(f"❌ Failed to load JSON: {e}")
            return False 

    def display(self, x_m=None, y_m=None, theta_deg=None, mapbytes=None, laser_range=None, laser_scan=None, trajectory=None, loop_detected=False):
        
        # Left: draw maze lines only
        if hasattr(self, '_maze_segments') and self._maze_segments:
            self.ax_left.clear()
            for segment in self._maze_segments:
                x0, y0 = segment['start']
                x1, y1 = segment['end']
                self.ax_left.plot([x0, x1], [y0, y1], color='black', linewidth=6)
            if self._start_point is not None:
                self.ax_left.plot(self._start_point[0], self._start_point[1], 'o', color='green', markersize=16, label='Start')
            self.ax_left.set_title("Maze Segments", fontsize=12)
            self.ax_left.set_xlabel('X')
            self.ax_left.set_ylabel('Y')
            self.ax_left.legend(loc='upper right')
            self.ax_left.set_aspect('equal')
            self.ax_left.grid(False)
        # Draw robot as a triangle on left
        if x_m is not None and y_m is not None and theta_deg is not None:
            if self._robot_left is not None:
                self._robot_left.remove()
            size = 0.6
            theta = np.deg2rad(theta_deg)
            p1 = np.array([x_m + size * np.cos(theta), y_m + size * np.sin(theta)])
            p2 = np.array([x_m + size * np.cos(theta + 2.5), y_m + size * np.sin(theta + 2.5)])
            p3 = np.array([x_m + size * np.cos(theta - 2.5), y_m + size * np.sin(theta - 2.5)])
            self._robot_left = self.ax_left.fill([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color='red', zorder=10)[0]
        
        
        # Draw laser rays/arcs on left
        if x_m is not None and y_m is not None and laser_scan is not None:
            self._draw_laser_rays(self.ax_left, x_m, y_m, theta_deg, laser_scan)
        

        
        # Right: show SLAM map as grayscale image
        if mapbytes is not None:
            mapimg = np.reshape(np.frombuffer(mapbytes, dtype=np.uint8), (self.map_size_pixels, self.map_size_pixels))
            self.ax_right.clear()
            self._slam_img_artist = self.ax_right.imshow(mapimg, cmap='gray', origin='lower', vmin=0, vmax=255)
            self.ax_right.set_title("SLAM Map")
            self.ax_right.set_xlabel("X")
            self.ax_right.set_ylabel("Y")
            self.ax_right.set_aspect('equal')
            self.ax_right.grid(False)
        # Draw robot as a triangle on right (pixel coordinates)
        if x_m is not None and y_m is not None and theta_deg is not None:
            if self._robot_right is not None:
                self._robot_right.remove()
            scale = self.map_size_meters / self.map_size_pixels
            px = x_m / scale
            py = y_m / scale
            size = 0.6 / scale
            theta = np.deg2rad(theta_deg)
            p1r = np.array([px + size * np.cos(theta), py + size * np.sin(theta)])
            p2r = np.array([px + size * np.cos(theta + 2.5), py + size * np.sin(theta + 2.5)])
            p3r = np.array([px + size * np.cos(theta - 2.5), py + size * np.sin(theta - 2.5)])
            self._robot_right = self.ax_right.fill([p1r[0], p2r[0], p3r[0]], [p1r[1], p2r[1], p3r[1]], color='red', zorder=10)[0]
        # Draw trajectory
        if trajectory is not None and len(trajectory) > 1:
            xs, ys, _ = zip(*trajectory)
            self.ax_left.plot(xs, ys, color='orange', linewidth=2, alpha=0.7, label='Trajectory')
            scale = self.map_size_meters / self.map_size_pixels
            pxs = [x / scale for x in xs]
            pys = [y / scale for y in ys]
            self.ax_right.plot(pxs, pys, color='orange', linewidth=2, alpha=0.7, label='Trajectory')


            #注意此处A. 轨迹记录延迟或不同步 轨迹（trajectory）是通过SLAM仿真器的get_trajectory()获得的，通常是SLAM内部每次update后记录的。
            #你的光标位置是pose，它是主循环中每次motion后立即更新的。
            #如果pose的更新和SLAM轨迹的记录不是严格同步，就会出现“光标和轨迹终点不一致”的现象。

            #已修改：将光标位置和轨迹终点同步，在test11中添加pose_history来表示轨迹

        # 回环高亮点不显示
        # if loop_detected and trajectory:
        #     x, y, _ = trajectory[-1]
        #     self.ax_left.plot(x, y, 'o', color='magenta', markersize=12, label='Loop Closure')
        #     px, py = x / scale, y / scale
        #     self.ax_right.plot(px, py, 'o', color='magenta', markersize=12, label='Loop Closure')
            #print(f"光标(px, py): {px}, {py} | 轨迹终点(pxs[-1], pys[-1]): {pxs[-1]}, {pys[-1]}")
        
        plt.draw()
        plt.pause(0.01)


        #这里如何设置??

    def _draw_laser_rays(self, ax, x_m, y_m, theta_deg, laser_scan):
        # Draw laser rays from robot position, given scan data (in meters)
        n = len(laser_scan)
        angle_min = 0
        angle_max = 2 * np.pi
        angles = np.linspace(angle_min, angle_max, n, endpoint=False)
        for r, a in zip(laser_scan, angles):
            # r: range in meters
            end_x = x_m + r * np.cos(a + np.deg2rad(theta_deg))
            end_y = y_m + r * np.sin(a + np.deg2rad(theta_deg))
            ax.plot([x_m, end_x], [y_m, end_y], color='cyan', alpha=0.2, linewidth=1)