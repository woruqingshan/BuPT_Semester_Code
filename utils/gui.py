# utils/gui.py
from roboviz import MapVisualizer
import numpy as np

class GUIVisualizer:
    def __init__(self, map_size_pixels, map_size_meters, title='SLAM Navigation', show_trajectory=True):
        """
        初始化GUI可视化器
        
        Args:
            map_size_pixels: 地图像素大小
            map_size_meters: 地图实际大小（米）
            title: 窗口标题
            show_trajectory: 是否显示轨迹
        """
        try:
            self.viz = MapVisualizer(map_size_pixels, map_size_meters, title, show_trajectory=show_trajectory)
            print(f"GUI initialized: {map_size_pixels}x{map_size_pixels} pixels, {map_size_meters}m")
        except Exception as e:
            print(f"Error initializing GUI: {e}")
            self.viz = None

    def display(self, x_m, y_m, theta_deg, mapbytes):
        """
        显示地图和机器人位置
        
        Args:
            x_m: 机器人x坐标（米）
            y_m: 机器人y坐标（米）
            theta_deg: 机器人朝向（度）
            mapbytes: 地图数据（bytearray）
            
        Returns:
            bool: 如果窗口被关闭返回False，否则返回True
        """
        if self.viz is None:
            print("GUI not initialized, skipping display")
            return True
            
        try:
            # 确保坐标在合理范围内
            if not (np.isfinite(x_m) and np.isfinite(y_m) and np.isfinite(theta_deg)):
                print(f"Invalid coordinates: x={x_m}, y={y_m}, theta={theta_deg}")
                return True
                
            # 调用PyRoboViz的display方法
            result = self.viz.display(x_m, y_m, theta_deg, mapbytes)
            return result
            
        except Exception as e:
            print(f"Error in display: {e}")
            return True