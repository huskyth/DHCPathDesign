import numpy as np
import pandas as pd
import re
import math
import os
from matplotlib import pyplot as plt
import sys


class StaticObstacle():
    def __init__(self, filepath='construct_map/coordinate.txt', agent_size=0.5):
        self.agent_size = agent_size
        self.readfile(filepath)
        self.static_map = self.grid()

    def readfile(self, filepath):
        map = {}
        with open(filepath, "r") as file:
            lines = file.readlines()
            i = 0
            while i < (len(lines)):
                if lines[i].strip() == 'Plane:':
                    i = i + 1
                    plane_coors = []
                    plane_coor = (lines[i:i + 4])
                    plane_coor = [list(eval(x.strip())) for x in plane_coor]
                    plane_coors.append(plane_coor)
                    map['plane'] = plane_coors
                    i = i + 5

                if lines[i].strip() == 'Obstacles:':
                    i = i + 1
                    obstacle_coors = []
                    while i < (len(lines)):
                        obstacle_coor = (lines[i:i + 4])
                        obstacle_coor = [list(eval(x.strip())) for x in obstacle_coor]
                        obstacle_coors.append(obstacle_coor)
                        i = i + 5
                    map['obstacles'] = obstacle_coors

                # if lines[i].strip() == 'Exhibitions:':
                #     i = i+1
                #     exhibition_coors = []
                #     while i < (len(lines)):
                #         exhibition_coor = (lines[i:i + 4])
                #         exhibition_coor = [list(eval(x.strip())) for x in exhibition_coor]
                #         exhibition_coors.append(exhibition_coor)
                #         i = i + 5
                # map['exhibition'] = exhibition_coors
                # return map
                self.map_data = map

    # def visual(self,map_data):
    #     fig = plt.figure(figsize=(5.5, 2))

    def grid(self):
        # 根据地图大小和agent大小栅格化
        plane = self.map_data['plane'][0]
        plane_left_bottom = (plane[0][0], plane[0][-1])
        plane_right_top = (plane[2][0], plane[2][-1])

        plane_length = plane_right_top[0] - plane_left_bottom[0]
        plane_width = plane_right_top[1] - plane_left_bottom[1]

        self.rows = math.ceil(plane_width / self.agent_size)
        self.columns = math.ceil(plane_length / self.agent_size)

        map_init = np.zeros((self.rows, self.columns), dtype=np.int32)

        # 填充障碍物
        obstacles = self.map_data['obstacles']
        for obstacle in obstacles:
            # 计算障碍物位置坐标
            obs_left_bottom = (obstacle[0][0], obstacle[0][-1])
            obs_right_top = (obstacle[2][0], obstacle[2][-1])

            # 占的行数和占的列数;
            relative_rpos = math.ceil((plane_right_top[1] - obs_right_top[1]) / self.agent_size) - 1
            relative_cpos = math.ceil((obs_left_bottom[0] - plane_left_bottom[0]) / self.agent_size) - 1

            obstacle_length_grid = math.ceil((obs_right_top[0] - obs_left_bottom[0]) / self.agent_size)
            obstacle_width_grid = math.ceil((obs_right_top[1] - obs_left_bottom[1]) / self.agent_size)

            map_init[relative_rpos:relative_rpos + obstacle_width_grid + 1,
            relative_cpos:relative_cpos + obstacle_length_grid + 1] = 1
            # print("static map generated")
        return map_init


if __name__ == '__main__':
    pass
    # map = StaticObstacle('coordinate.txt')
    # print("map end")
