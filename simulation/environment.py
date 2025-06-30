# -*- coding: utf-8 -*-

import numpy as np

class SimulationEnvironment:
    def __init__(self, map_size=(100, 100), obstacle_density=0.2):
        
        self.map_size = map_size
        self.map = np.zeros(map_size)
        self.generate_obstacles(obstacle_density)
        self.robot_pose = np.array([0, 0, 0])  # x, y, theta

    def generate_obstacles(self, obstacle_density):
       
        num_obstacles = int(self.map_size[0] * self.map_size[1] * obstacle_density)
        obstacle_indices = np.random.randint(0, self.map_size[0], size=(num_obstacles, 2))
        for idx in obstacle_indices:
            self.map[idx[0], idx[1]] = 1

    def simulate_lidar(self, num_beams=360, max_range=10):
        
        angles = np.linspace(0, 2 * np.pi, num_beams)
        distances = []
        for angle in angles:
            dx = max_range * np.cos(angle + self.robot_pose[2])
            dy = max_range * np.sin(angle + self.robot_pose[2])
            end_x = int(self.robot_pose[0] + dx)
            end_y = int(self.robot_pose[1] + dy)
            distance = max_range
            for r in range(1, max_range):
                current_x = int(self.robot_pose[0] + r * np.cos(angle + self.robot_pose[2]))
                current_y = int(self.robot_pose[1] + r * np.sin(angle + self.robot_pose[2]))
                if 0 <= current_x < self.map_size[0] and 0 <= current_y < self.map_size[1]:
                    if self.map[current_x, current_y] == 1:
                        distance = r
                        break
            distances.append(distance)
        return angles, np.array(distances)

    def simulate_odometry(self):
        
        return self.robot_pose

    def update_robot_pose(self, new_pose):
       
        self.robot_pose = new_pose