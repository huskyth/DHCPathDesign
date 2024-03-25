import numpy as np
import pandas as pd
import re
import math
import os
from matplotlib import pyplot as plt


class DynamicPedestrian:
    def __init__(self, rows, columns, filepath='construct_map/pedestrian.txt'):
        self.agent_size = 0.5  # 需要与static_map.py保持一致哦
        self.readfile(filepath)
        self.get_planecoor()
        self.rows = rows
        self.columns = columns

    # 构建行人数据的 dataframe
    def readfile(self, filepath):
        time2pos = []
        with open(filepath, "r") as file:
            lines = file.readlines()
            ped_index = lines[0].strip().split(' ')

            for line in lines[1:]:
                # 匹配方括号里的字符
                info = re.findall(r"[(](.*?)[)]", line)
                pos = [list(eval(x)) for x in info]
                time2pos.append(pos)
        # i行代表i时刻的行人数据
        self.pde_df = pd.DataFrame(time2pos, columns=ped_index)

    # 输入t时刻的一组行人坐标，输出他们在栅格中的位置
    def get_planecoor(self, planepath='construct_map/coordinate.txt'):
        # 首先要读取原始地板的坐标，再计算相对位置
        with open(planepath, "r") as file:
            lines = file.readlines()
            plane_coor = (lines[1: 5])
            plane_coor = [list(eval(x.strip())) for x in plane_coor]
        # 获取plane上下左右的坐标
        self.plane_left_bottom = (plane_coor[0][0], plane_coor[0][-1])
        self.plane_right_top = (plane_coor[2][0], plane_coor[2][-1])

    # 获取行人在栅格中的位置
    def get_pedcoor(self, coorlist):
        self.dynamic_map = np.zeros((self.rows, self.columns), dtype=np.int32)
        for coor in coorlist:
            ped_coor = (coor[0], coor[-1])
            relative_rpos = math.ceil((self.plane_right_top[1] - ped_coor[1]) / self.agent_size) - 1
            relative_cpos = math.ceil((ped_coor[0] - self.plane_left_bottom[0]) / self.agent_size) - 1
            self.dynamic_map[relative_rpos, relative_cpos] = 1
        # print("dynamic map generated")
        return self.dynamic_map


if __name__ == '__main__':
    map = DynamicPedestrian(56, 22)
    coorlist = map.pde_df.iloc[0].tolist()
    pos = map.get_pedcoor(coorlist)
