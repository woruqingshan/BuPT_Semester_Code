# -*- coding: utf-8 -*-

from simulation.environment import SimulationEnvironment
from planning.a_star import AStarPlanner
from planning.dwa import DWAPlanner
from utils.path_comparison import compare_paths
from utils.gui import show_map

if __name__ == "__main__":
    print('start simulation')
    # ��ʼ�����滷��
    sim_env = SimulationEnvironment()

    # ���������յ�
    start = (0, 0)
    goal = (sim_env.map_size[0] - 1, sim_env.map_size[1] - 1)

    # ȫ��·���滮
    a_star_planner = AStarPlanner(sim_env.map)
    global_path = a_star_planner.planning(start, goal)

    # ��ʾ��ͼ��ȫ��·��
    show_map(sim_env, global_path)

    # ģ��������˶�
    dwa_planner = DWAPlanner()
    current_pose = sim_env.simulate_odometry()
    for point in global_path[1:]:
        lidar_angles, lidar_distances = sim_env.simulate_lidar()
        v, w = dwa_planner.plan(current_pose, point, (lidar_angles, lidar_distances))
        # ���»�����λ��
        new_x = current_pose[0] + v * np.cos(current_pose[2]) * sim_env.dt
        new_y = current_pose[1] + v * np.sin(current_pose[2]) * sim_env.dt
        new_theta = current_pose[2] + w * sim_env.dt
        new_pose = np.array([new_x, new_y, new_theta])
        sim_env.update_robot_pose(new_pose)
        current_pose = new_pose
        show_map(sim_env, global_path)

    # �����˷������
    return_path = a_star_planner.planning(tuple(sim_env.robot_pose[:2]), start)
    show_map(sim_env, return_path)