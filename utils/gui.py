# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def show_map(sim_env, path=None):
    
    plt.imshow(sim_env.map, cmap='gray')
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_y, path_x, 'r-')
    plt.plot(sim_env.robot_pose[1], sim_env.robot_pose[0], 'go')
    plt.show()