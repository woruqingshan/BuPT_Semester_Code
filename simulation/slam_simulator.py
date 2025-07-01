import numpy as np
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser

class SLAMSimulator:
    def __init__(self, map_size_pixels=800, map_size_meters=15.0):
        self.MAP_SIZE_PIXELS = map_size_pixels
        self.MAP_SIZE_METERS = map_size_meters
        self.laser = Laser(360, 10, 360, 10000, 0, 0)
        self.slam = RMHC_SLAM(self.laser, self.MAP_SIZE_PIXELS, self.MAP_SIZE_METERS)
        self.occupancy_grid = None  # Binary maze grid (1=obstacle, 0=free)

    def set_occupancy_grid(self, grid):
        # Set the binary occupancy grid (maze)
        self.occupancy_grid = grid

    def simulate_laser_scan(self, pose=None):
        # Simulate a laser scan based on the occupancy grid and robot pose
        if self.occupancy_grid is None or pose is None:
            # Fallback to random if no grid or pose
            return list(np.random.randint(0, self.laser.distance_no_detection_mm, self.laser.scan_size))
        scan = []
        n_angles = self.laser.scan_size
        max_range = self.laser.distance_no_detection_mm / 1000.0  # meters
        angle_min = 0
        angle_max = 2 * np.pi
        angles = np.linspace(angle_min, angle_max, n_angles, endpoint=False)
        # Robot pose: [x, y, theta] in meters/radians
        x0 = pose[0]
        y0 = pose[1]
        theta0 = pose[2]
        grid_h, grid_w = self.occupancy_grid.shape
        scale_x = self.MAP_SIZE_METERS / grid_h
        scale_y = self.MAP_SIZE_METERS / grid_w
        for a in angles:
            found = False
            for r in np.linspace(0, max_range, int(max_range/0.02)):
                x = x0 + r * np.cos(a + theta0)
                y = y0 + r * np.sin(a + theta0)
                i = int(round(x / scale_x))
                j = int(round(y / scale_y))
                if 0 <= i < grid_h and 0 <= j < grid_w:
                    if self.occupancy_grid[i, j] == 1:
                        scan.append(int(r * 1000))  # mm
                        found = True
                        break
                else:
                    scan.append(int(r * 1000))
                    found = True
                    break
            if not found:
                scan.append(int(max_range * 1000))
        return scan

    def update(self, sensor_data=None, pose_change=None):
        if sensor_data is None:
            sensor_data = self.simulate_laser_scan()
        if pose_change is None:
            pose_change = (0, 0, 0.1)
        self.slam.update(sensor_data, pose_change)
    def get_map(self):
        mapbytes = bytearray(self.MAP_SIZE_PIXELS * self.MAP_SIZE_PIXELS)
        self.slam.getmap(mapbytes)
        return mapbytes
    def get_position(self):
        x_mm, y_mm, theta_degrees = self.slam.getpos()
        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        return np.array([x_m, y_m, theta_degrees])
