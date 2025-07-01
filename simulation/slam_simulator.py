import numpy as np
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser
from utils.gui import GUIVisualizer
from time import time

class SLAMSimulator:
    def __init__(self, map_size_pixels=800, map_size_meters=32):
        # 地图参数
        self.MAP_SIZE_PIXELS = map_size_pixels
        self.MAP_SIZE_METERS = map_size_meters

        # 激光传感器参数
        LASER_OFFSET_MM = 0
        LASER_SCAN_SIZE = 360
        LASER_SCAN_RATE_HZ = 10
        LASER_DETECTION_ANGLE_DEGREES = 360
        LASER_DETECTION_MAX_MM = 4000
        LASER_DETECTION_MARGIN = 0

        # 创建激光传感器对象
        self.laser = Laser(LASER_SCAN_SIZE, LASER_SCAN_RATE_HZ, LASER_DETECTION_ANGLE_DEGREES, 
                          LASER_DETECTION_MAX_MM, LASER_DETECTION_MARGIN, LASER_OFFSET_MM)

        # 创建RMHC SLAM对象
        self.slam = RMHC_SLAM(self.laser, self.MAP_SIZE_PIXELS, self.MAP_SIZE_METERS)

        # 创建GUI可视化对象
        self.viz = GUIVisualizer(self.MAP_SIZE_PIXELS, self.MAP_SIZE_METERS)

    def simulate_laser_scan(self):
        # 返回list而不是numpy数组
        return list(np.random.randint(0, self.laser.distance_no_detection_mm, self.laser.scan_size))

    def update(self, sensor_data=None):
        if sensor_data is None:
            sensor_data = self.simulate_laser_scan()
        # 更新SLAM算法
        self.slam.update(sensor_data)

    def get_map(self):
        # 创建正确大小的bytearray来存储地图数据
        mapbytes = bytearray(self.MAP_SIZE_PIXELS * self.MAP_SIZE_PIXELS)
        # 使用getmap方法填充地图数据
        self.slam.getmap(mapbytes)
        return mapbytes

    def get_position(self):
        x_mm, y_mm, theta_degrees = self.slam.getpos()
        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        return np.array([x_m, y_m, theta_degrees])

    def visualize(self):
        x_m, y_m, theta_degrees = self.get_position()
        mapbytes = self.get_map()
        return self.viz.display(x_m, y_m, theta_degrees, mapbytes)

    def run_simulation(self):
        prevtime = time()
        while True:
            # 模拟激光扫描数据
            scan = self.simulate_laser_scan()

            # 更新SLAM算法
            self.update(scan)

            # 显示当前位置和地图
            if not self.visualize():
                break

            # 控制仿真速度
            currtime = time.time()
            elapsed = currtime - prevtime
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)
            prevtime = currtime
