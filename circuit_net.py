# -*- coding: utf-8 -*-
"""
文件头信息
------------------------------------------------------------------------------------------------------------------------
[开发人员]: Huang Yu\n
[创建日期]: 2024.4.9\n
[目标工具]: PyCharm\n
[版权信息]: © Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab All Rights Reserved\n

版本修改
------------------------------------------------------------------------------------------------------------------------
[Date]         [By]         [Version]         [Change Log]\n
2024.4.9       Huang Yu     1.0               first implementation\n

功能描述
------------------------------------------------------------------------------------------------------------------------
带有源器件的矩形电阻网络的定义
"""
import logging
import torch
import numpy as np
from torch import nn
import param_man as pm

sys_log = logging.getLogger('logger')


class local_net(nn.Module):
    """不考虑global IR-Drop只考虑local IR-Drop的电阻网络"""

    def __init__(self, size, type_mat, conduct, weight):
        """网络定义"""
        super(local_net, self).__init__()
        self.size = size  # 面板尺寸
        self.type_mat = type_mat  # 类型矩阵
        self.conduct = conduct  # 电导矩阵
        self.weight = weight
        self.current_coeff = torch.tensor(pm.get_param("current_coeff"), device=pm.get_param("Device"))  # 电流系数
        self.point = pm.get_param("elvdd_point")  # ELVDD点坐标
        self.elvdd = torch.tensor(pm.get_param("elvdd"), device=pm.get_param("Device"))  # ELVDD大小
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # 平均池化
        sys_log.info('Local IR-Drop network constructed')

    def forward(self, v_data, v):
        """正向传播过程"""
        voltage_mat = self.voltage_transform(v)  # 变换增广后的电压矩阵
        # 边界条件修正(必须进行，保证ELVDD点梯度为0)
        voltage_mat_fix = self.vmat_bound(voltage_mat, v)
        current_active = self.active_current(v_data, v)  # 有源器件电流矩阵
        current_grid = torch.mul(voltage_mat_fix, self.conduct)
        current_grid = torch.unsqueeze(current_grid, 0)
        current_grid = torch.unsqueeze(current_grid, 0)
        current_grid = 4 * self.pool(current_grid)  # 电阻网格电流矩阵
        current_grid = torch.squeeze(current_grid)
        current_loss = current_grid - current_active  # 电流残差
        loss = torch.mul(current_loss, self.weight)  # 实际残差
        return loss, current_loss, current_grid, current_active

    def active_current(self, v_data, v):
        """有源器件电流计算"""
        # current_mat = self.current_coeff * torch.pow(v - v_data, 2)
        current_mat = 1e-3 * torch.ones(self.size, device=pm.get_param("Device"))
        # 边界条件修正(?)
        for i in range(pm.get_param("elvdd_num")):
            current_mat[self.point[i][0], self.point[i][1]] = 0
        return current_mat
    
    def voltage_transform(self, v):
        """变换电压矩阵"""
        size = self.size
        voltage_mat = torch.zeros([2 * size[0], 2 * size[1]], device=pm.get_param("Device"))
        # 中间元素构建
        if size[0] > 2 and size[1] > 2:
            voltage_mat[2:2 * (size[0] - 2) + 1:2, 2:2 * (size[1] - 2) + 1:2] = \
                v[0:-2, 1:-1] - v[1:-1, 1:-1]  # 上像素与该像素电压差
            voltage_mat[2:2 * (size[0] - 2) + 1:2, 3:2 * (size[1] - 2) + 2:2] = \
                v[1:-1, 0:-2] - v[1:-1, 1:-1]  # 左像素与该像素电压差
            voltage_mat[3:2 * (size[0] - 2) + 2:2, 2:2 * (size[1] - 2) + 1:2] = \
                v[1:-1, 2:] - v[1:-1, 1:-1]  # 右像素与该像素电压差
            voltage_mat[3:2 * (size[0] - 2) + 2:2, 3:2 * (size[1] - 2) + 2:2] = \
                v[2:, 1:-1] - v[1:-1, 1:-1]  # 下像素与该像素电压差
        # 左上角
        voltage_mat[1, 0] = v[0, 1] - v[0, 0]  # 右压差 (2,1)
        voltage_mat[1, 1] = v[1, 0] - v[0, 0]  # 下压差 (2,2)
        # 右上角
        voltage_mat[0, -1] = v[0, -2] - v[0, -1]
        voltage_mat[1, -1] = v[1, -1] - v[0, -1]
        # 左下角
        voltage_mat[-2, 0] = v[-2, 0] - v[-1, 0]
        voltage_mat[-1, 0] = v[-1, 1] - v[-1, 0]
        # 右下角
        voltage_mat[-2, -2] = v[-2, -1] - v[-1, -1]
        voltage_mat[-2, -1] = v[-1, -2] - v[-1, -1]
        # 左右边线
        for i in range(size[0] - 2):
            # 左边线
            voltage_mat[2 * (i + 1)][0] = v[i][0] - v[i + 1][0]  # 上压差 (1,1)
            voltage_mat[2 * (i + 1) + 1][0] = v[i + 1][1] - v[i + 1][0]  # 右压差 (2,1)
            voltage_mat[2 * (i + 1) + 1][1] = v[i + 2][0] - v[i + 1][0]  # 下压差 (2,2)
            # 右边线
            voltage_mat[2 * (i + 1)][-2] = v[i][-1] - v[i + 1][-1]  # 上压差 (1,1)
            voltage_mat[2 * (i + 1)][-1] = v[i + 1][-2] - v[i + 1][-1]  # 左压差 (1,2)
            voltage_mat[2 * (i + 1) + 1][-1] = v[i + 2][-1] - v[i + 1][-1]  # 下压差 (2,2)
        # 上下边线
        for i in range(size[1] - 2):
            # 上边线
            voltage_mat[0][2 * (i + 1) + 1] = v[0][i] - v[0][i + 1]  # 左压差 (1,2)
            voltage_mat[1][2 * (i + 1)] = v[0][i + 2] - v[0][i + 1]  # 右压差 (2,1)
            voltage_mat[1][2 * (i + 1) + 1] = v[1][i + 1] - v[0][i + 1]  # 下压差 (2,2)
            # 下边线
            voltage_mat[-2][2 * (i + 1)] = v[-2][i + 1] - v[-1][i + 1]  # 上压差 (1,1)
            voltage_mat[-2][2 * (i + 1) + 1] = v[-1][i] - v[-1][i + 1]  # 左压差 (1,2)
            voltage_mat[-1][2 * (i + 1)] = v[-1][i + 2] - v[-1][i + 1]  # 右压差 (2,1)
        return voltage_mat

    def vmat_bound(self, voltage_mat, v):
        voltage_mat_fix = voltage_mat

        """根据边界条件修正voltage_mat"""
        for i in range(pm.get_param("elvdd_num")):
            x = self.point[i][0]
            y = self.point[i][1]
            # 本元素对应的电压矩阵清零，因为电流方程不再成立
            voltage_mat_fix[2 * x, 2 * y] = 0 * v[x, y].detach()
            voltage_mat_fix[2 * x, 2 * y + 1] = 0 * v[x, y].detach()
            voltage_mat_fix[2 * x + 1, 2 * y] = 0 * v[x, y].detach()
            voltage_mat_fix[2 * x + 1, 2 * y + 1] = 0 * v[x, y].detach()
            if self.type_mat[x, y] == 0:  # 中间的像素
                # 上元素(x-1,y)的下电压修正 (2,2)
                voltage_mat_fix[2 * (x - 1) + 1][2 * y + 1] = v[x, y].detach() - v[x - 1, y]
                # 左元素(x,y-1)的右电压修正 (2,1)
                voltage_mat_fix[2 * x + 1][2 * (y - 1)] = v[x, y].detach() - v[x, y - 1]
                # 右元素(x,y+1)的左电压修正 (1,2)
                voltage_mat_fix[2 * x][2 * (y + 1) + 1] = v[x, y].detach() - v[x, y + 1]
                # 下元素(x+1,y)的上电压修正(1,1)
                voltage_mat_fix[2 * (x + 1)][2 * y] = v[x, y].detach() - v[x + 1, y]
            elif self.type_mat[x, y] == 1:  # 左上角的像素
                # 右元素(x,y+1)的左电压修正 (1,2)
                voltage_mat_fix[2 * x][2 * (y + 1) + 1] = v[x, y].detach() - v[x, y + 1]
                # 下元素(x+1,y)的上电压修正(1,1)
                voltage_mat_fix[2 * (x + 1)][2 * y] = v[x, y].detach() - v[x + 1, y]
            elif self.type_mat[x, y] == 2:  # 右上角的像素
                # 左元素(x,y-1)的右电压修正 (2,1)
                voltage_mat_fix[2 * x + 1][2 * (y - 1)] = v[x, y].detach() - v[x, y - 1]
                # 下元素(x+1,y)的上电压修正(1,1)
                voltage_mat_fix[2 * (x + 1)][2 * y] = v[x, y].detach() - v[x + 1, y]
            elif self.type_mat[x, y] == 3:  # 左下角的像素
                # 上元素(x-1,y)的下电压修正 (2,2)
                voltage_mat_fix[2 * (x - 1) + 1][2 * y + 1] = v[x, y].detach() - v[x - 1, y]
                # 右元素(x,y+1)的左电压修正 (1,2)
                voltage_mat_fix[2 * x][2 * (y + 1) + 1] = v[x, y].detach() - v[x, y + 1]
            elif self.type_mat[x, y] == 4:  # 右下角的像素
                # 上元素(x-1,y)的下电压修正 (2,2)
                voltage_mat_fix[2 * (x - 1) + 1][2 * y + 1] = v[x, y].detach() - v[x - 1, y]
                # 左元素(x,y-1)的右电压修正 (2,1)
                voltage_mat_fix[2 * x + 1][2 * (y - 1)] = v[x, y].detach() - v[x, y - 1]
            elif self.type_mat[x, y] == 5:  # 上边线的像素
                # 左元素(x,y-1)的右电压修正 (2,1)
                voltage_mat_fix[2 * x + 1][2 * (y - 1)] = v[x, y].detach() - v[x, y - 1]
                # 右元素(x,y+1)的左电压修正 (1,2)
                voltage_mat_fix[2 * x][2 * (y + 1) + 1] = v[x, y].detach() - v[x, y + 1]
                # 下元素(x+1,y)的上电压修正(1,1)
                voltage_mat_fix[2 * (x + 1)][2 * y] = v[x, y].detach() - v[x + 1, y]
            elif self.type_mat[x, y] == 6:  # 左边线的像素
                # 上元素(x-1,y)的下电压修正 (2,2)
                voltage_mat_fix[2 * (x - 1) + 1][2 * y + 1] = v[x, y].detach() - v[x - 1, y]
                # 右元素(x,y+1)的左电压修正 (1,2)
                voltage_mat_fix[2 * x][2 * (y + 1) + 1] = v[x, y].detach() - v[x, y + 1]
                # 下元素(x+1,y)的上电压修正(1,1)
                voltage_mat_fix[2 * (x + 1)][2 * y] = v[x, y].detach() - v[x + 1, y]
            elif self.type_mat[x, y] == 7:  # 右边线的像素
                # 上元素(x-1,y)的下电压修正 (2,2)
                voltage_mat_fix[2 * (x - 1) + 1][2 * y + 1] = v[x, y].detach() - v[x - 1, y]
                # 左元素(x,y-1)的右电压修正 (2,1)
                voltage_mat_fix[2 * x + 1][2 * (y - 1)] = v[x, y].detach() - v[x, y - 1]
                # 下元素(x+1,y)的上电压修正(1,1)
                voltage_mat_fix[2 * (x + 1)][2 * y] = v[x, y].detach() - v[x + 1, y]
            elif self.type_mat[x, y] == 8:  # 下边线的像素
                # 上元素(x-1,y)的下电压修正 (2,2)
                voltage_mat_fix[2 * (x - 1) + 1][2 * y + 1] = v[x, y].detach() - v[x - 1, y]
                # 左元素(x,y-1)的右电压修正 (2,1)
                voltage_mat_fix[2 * x + 1][2 * (y - 1)] = v[x, y].detach() - v[x, y - 1]
                # 右元素(x,y+1)的左电压修正 (1,2)
                voltage_mat_fix[2 * x][2 * (y + 1) + 1] = v[x, y].detach() - v[x, y + 1]
            else:
                sys_log.critical("Invalid type!")
        return voltage_mat_fix


def grid_type():
    """得到提示节点类型的矩阵"""
    size = [pm.get_param("pixel_row"), pm.get_param("pixel_col")]  # 类型矩阵大小
    type_mat = torch.zeros(size, dtype=torch.int8)  # 位置矩阵
    type_mat[0, :] = 5
    type_mat[:, 0] = 6
    type_mat[:, -1] = 7
    type_mat[-1, :] = 8
    type_mat[0, 0] = 1
    type_mat[0, -1] = 2
    type_mat[-1, 0] = 3
    type_mat[-1, -1] = 4
    return type_mat


def build_pos():
    """构建位置矩阵"""
    size = [pm.get_param("pixel_row"), pm.get_param("pixel_col"), 2]  # 位置矩阵大小 [x, y]
    pos = torch.zeros(size)  # 位置矩阵
    if not pm.get_param("if_pos_load"):
        for i in range(size[0]):
            for j in range(size[1]):
                pos[i][j][0] = j * pm.get_param("p_x")  # 绝对x坐标
                pos[i][j][1] = i * pm.get_param("p_y")  # 绝对y坐标
    return pos


def build_conduct():
    """构建电导网络"""
    type_mat = grid_type()
    pos = build_pos()
    size = [pm.get_param("pixel_row"), pm.get_param("pixel_col")]  # 电阻矩阵大小
    conduct = torch.zeros([2 * size[0], 2 * size[1]])  # 电阻矩阵
    if not pm.get_param("if_res_load"):
        rs_x = pm.get_param("r_x")
        rs_y = pm.get_param("r_y")
        for i in range(size[0]):
            for j in range(size[1]):
                if type_mat[i][j] == 0:  # 中间的像素
                    conduct[i * 2 + 0][j * 2 + 0] = 1 / (rs_y * (pos[i][j][1] - pos[i - 1][j][1]))  # 与上像素的电导值
                    conduct[i * 2 + 0][j * 2 + 1] = 1 / (rs_x * (pos[i][j][0] - pos[i][j - 1][0]))  # 与左像素的电导值
                    conduct[i * 2 + 1][j * 2 + 0] = 1 / (rs_x * (pos[i][j + 1][0] - pos[i][j][0]))  # 与右像素的电导值
                    conduct[i * 2 + 1][j * 2 + 1] = 1 / (rs_y * (pos[i + 1][j][1] - pos[i][j][1]))  # 与下像素的电导值
                elif type_mat[i][j] == 1:  # 左上角的像素
                    conduct[i * 2 + 1][j * 2 + 0] = 1 / (rs_x * (pos[i][j + 1][0] - pos[i][j][0]))  # 与右像素的电导值
                    conduct[i * 2 + 1][j * 2 + 1] = 1 / (rs_y * (pos[i + 1][j][1] - pos[i][j][1]))  # 与下像素的电导值
                elif type_mat[i][j] == 2:  # 右上角的像素
                    conduct[i * 2 + 0][j * 2 + 1] = 1 / (rs_x * (pos[i][j][0] - pos[i][j - 1][0]))  # 与左像素的电导值
                    conduct[i * 2 + 1][j * 2 + 1] = 1 / (rs_y * (pos[i + 1][j][1] - pos[i][j][1]))  # 与下像素的电导值
                elif type_mat[i][j] == 3:  # 左下角的像素
                    conduct[i * 2 + 0][j * 2 + 0] = 1 / (rs_y * (pos[i][j][1] - pos[i - 1][j][1]))  # 与上像素的电导值
                    conduct[i * 2 + 1][j * 2 + 0] = 1 / (rs_x * (pos[i][j + 1][0] - pos[i][j][0]))  # 与右像素的电导值
                elif type_mat[i][j] == 4:  # 右下角的像素
                    conduct[i * 2 + 0][j * 2 + 0] = 1 / (rs_y * (pos[i][j][1] - pos[i - 1][j][1]))  # 与上像素的电导值
                    conduct[i * 2 + 0][j * 2 + 1] = 1 / (rs_x * (pos[i][j][0] - pos[i][j - 1][0]))  # 与左像素的电导值
                elif type_mat[i][j] == 5:  # 上边线的像素
                    conduct[i * 2 + 0][j * 2 + 1] = 1 / (rs_x * (pos[i][j][0] - pos[i][j - 1][0]))  # 与左像素的电导值
                    conduct[i * 2 + 1][j * 2 + 0] = 1 / (rs_x * (pos[i][j + 1][0] - pos[i][j][0]))  # 与右像素的电导值
                    conduct[i * 2 + 1][j * 2 + 1] = 1 / (rs_y * (pos[i + 1][j][1] - pos[i][j][1]))  # 与下像素的电导值
                elif type_mat[i][j] == 6:  # 左边线的像素
                    conduct[i * 2 + 0][j * 2 + 0] = 1 / (rs_y * (pos[i][j][1] - pos[i - 1][j][1]))  # 与上像素的电导值
                    conduct[i * 2 + 1][j * 2 + 0] = 1 / (rs_x * (pos[i][j + 1][0] - pos[i][j][0]))  # 与右像素的电导值
                    conduct[i * 2 + 1][j * 2 + 1] = 1 / (rs_y * (pos[i + 1][j][1] - pos[i][j][1]))  # 与下像素的电导值
                elif type_mat[i][j] == 7:  # 右边线的像素
                    conduct[i * 2 + 0][j * 2 + 0] = 1 / (rs_y * (pos[i][j][1] - pos[i - 1][j][1]))  # 与上像素的电导值
                    conduct[i * 2 + 0][j * 2 + 1] = 1 / (rs_x * (pos[i][j][0] - pos[i][j - 1][0]))  # 与左像素的电导值
                    conduct[i * 2 + 1][j * 2 + 1] = 1 / (rs_y * (pos[i + 1][j][1] - pos[i][j][1]))  # 与下像素的电导值
                elif type_mat[i][j] == 8:  # 下边线的像素
                    conduct[i * 2 + 0][j * 2 + 0] = 1 / (rs_y * (pos[i][j][1] - pos[i - 1][j][1]))  # 与上像素的电导值
                    conduct[i * 2 + 0][j * 2 + 1] = 1 / (rs_x * (pos[i][j][0] - pos[i][j - 1][0]))  # 与左像素的电导值
                    conduct[i * 2 + 1][j * 2 + 0] = 1 / (rs_x * (pos[i][j + 1][0] - pos[i][j][0]))  # 与右像素的电导值
                else:
                    sys_log.critical("Invalid type!")
    return conduct


def get_vdata():
    """得到Vdata数据"""
    size = [pm.get_param("pixel_row"), pm.get_param("pixel_col")]  # vdata矩阵大小
    v_data = torch.zeros(size)
    if not pm.get_param("if_vdata_load"):
        v_data = 4 * torch.ones(size)
    return v_data


def get_weight():
    """得到权重矩阵"""
    size = [pm.get_param("pixel_row"), pm.get_param("pixel_col")]  # 权重矩阵大小
    weight = torch.ones(size)
    x = torch.zeros(size)
    y = torch.zeros(size)
    for row in range(size[0]):
        x[row, :] = row
    for col in range(size[1]):
        y[:, col] = col
    point = torch.tensor(pm.get_param("elvdd_point"))  # ELVDD点坐标
    for i in range(pm.get_param("elvdd_num")):  # 添加点权重
        x_ = x.subtract(point[0, 0])
        y_ = y.subtract(point[0, 1])
        weight_tmp = 1 / (1 + torch.pow(x_, 2) + torch.pow(y_, 2))  # 暂存权重
        weight = weight + weight_tmp
    return weight
