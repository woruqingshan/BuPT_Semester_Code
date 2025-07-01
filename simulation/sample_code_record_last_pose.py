from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser

# 初始化激光雷达和SLAM对象
laser = Laser(scan_size=360, scan_rate_hz=5, detection_angle_degrees=360, distance_no_detection_mm=12000, offset_mm=0)
slam = RMHC_SLAM(laser, map_size_pixels=800, map_size_meters=35)

# 记录已到达的位置
visited_positions = []

# 模拟更新过程
for i in range(10):
    # 模拟读取激光雷达数据
    scan = [1000] * 360
    # 更新SLAM
    slam.update(scan)
    # 获取当前位置
    x, y, theta = slam.getpos()
    # 记录位置
    visited_positions.append((x, y, theta))

print("Visited positions:", visited_positions)