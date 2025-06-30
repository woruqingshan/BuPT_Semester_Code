# -*- coding: utf-8 -*-

import numpy as np

class DWAPlanner:
    def __init__(self, max_v=1.0, max_w=np.pi/4, dt=0.1):
        
        self.max_v = max_v
        self.max_w = max_w
        self.dt = dt

    def plan(self, current_pose, goal, lidar_data):
        
        # ��ʾ����δʵ�������� DWA �㷨
        v = 0.1
        w = 0.0
        return v, w