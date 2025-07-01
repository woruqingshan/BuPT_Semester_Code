# -*- coding: utf-8 -*-

import math
from heapq import heappush, heappop

class AStarPlanner:
    def __init__(self, map):
       
        self.map = map
        self.motion = self.get_motion_model()

    def planning(self, start, goal):
        
        open_list, closed_list = [], {}
        heappush(open_list, (0, start))
        cost = {start: 0}
        parent = {start: None}

        while open_list:
            _, current = heappop(open_list)
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path

            closed_list[current] = True
            for move_x, move_y, move_cost in self.motion:
                next_node = (current[0] + move_x, current[1] + move_y)
                if next_node in closed_list:
                    continue
                if not self.verify_node(next_node):
                    continue
                new_cost = cost[current] + move_cost
                if next_node not in cost or new_cost < cost[next_node]:
                    cost[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal)
                    heappush(open_list, (priority, next_node))
                    parent[next_node] = current

        return None

    def verify_node(self, node):
        # node: (y, x)
        if node[0] < 0 or node[0] >= self.map.shape[0] or node[1] < 0 or node[1] >= self.map.shape[1]:
            return False
        if self.map[node[0], node[1]] == 1:
            return False
        return True

    def heuristic(self, node, goal):
        
        return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    def get_motion_model(self):
        
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion