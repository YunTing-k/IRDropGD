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
仿真参数(软硬件参数)具体定义：硬件参数包括横向/纵向电阻率，像素数目，像素间距，ELVDD注入情况，是否考虑全局IR-Drop，驱动电流系数。软件参数包括
求解迭代次数，学习率，优化器设置，是否使用双精度计算\n
左上角为(0,0)，右为x方向，下为y方向，坐标(a,b)表示在第a行，第b列，则x=b,y=a
"""
import logging
import numpy as np
import torch

import param_man as pm

sys_log = logging.getLogger('logger')


def set_default_param():
    """设置参数"""
    pm.set_param("if_res_load", False)  # 电阻率是否从文件中提取，否：均匀分布，理想均一 是：从文件中读取
    pm.set_param("r_x", 1e5)  # 横向单位长度电阻 Ω/m
    pm.set_param("r_y", 1e5)  # 纵向单位长度电阻 Ω/m

    pm.set_param("if_pos_load", False)  # 像素分布是否文件中提取，否：理想矩形排布 是：从文件中读取
    pm.set_param("pixel_col", 100)  # 横向像素数目
    pm.set_param("pixel_row", 100)  # 纵向像素数目
    if (pm.get_param("pixel_col") < 2) or (pm.get_param("pixel_row") < 2):
        sys_log.critical("Invalid panel size!")
    pm.set_param("pixel_num", pm.get_param("pixel_col") * pm.get_param("pixel_row"))  # 总像素数目
    pm.set_param("p_x", 1e-6)  # 横向间隔间距 m
    pm.set_param("p_y", 1e-6)  # 纵向间隔间距 m

    pm.set_param("if_elvdd_load", False)  # ELVDD是否从文件中提取，否：VDD点电压自定义 是：从文件中读取
    pm.set_param("elvdd", [5.])  # ELVDD大小 V
    pm.set_param("elvdd_point", [[0, 0]])  # ELVDD点的像素行/列坐标 [[row1, col1], [row2, col2], ...]
    pm.set_param("elvdd_num", np.size(pm.get_param("elvdd"), 0))  # ELVDD点的数目

    pm.set_param("glb_ird", False)  # 是否考虑全局IR-Drop
    pm.set_param("if_glb_r_load", False)  # 全局IR-Drop电阻是否从文件中提取，否：自定义 是：从文件中读取
    pm.set_param("glb_r", np.array([1]))  # 全局IR-Drop电阻大小 Ω

    pm.set_param("current_coeff", 1e-2)  # 有源器件电流计算参数 I = current_coeff * f(vdata, v), f自定义在net中
    pm.set_param("if_vdata_load", False)  # Vdata是否从文件提取，否：自定义 是：从文件提取
    sys_log.info('Hardware parameters added')

    pm.set_param("elvdd_ini", 5.)  # 求解初始化的电压大小 V
    pm.set_param("Device", "cuda:0")  # 运算设备
    # pm.set_param("Device", "cpu")  # 运算设备
    if not torch.cuda.is_available():  # 无法调用CUDA
        pm.set_param('Device', 'cpu')
        sys_log.warning('Cuda device is unavailable')
    sys_log.info('Current device is:' + pm.get_param('Device'))
    pm.set_param("dg_itr", 1000)  # 梯度下降迭代次数
    pm.set_param("dg_loss_type", "MSELoss")  # loss计算方法
    pm.set_param("dg_lr", 2)  # 梯度下降速率
    pm.set_param("dg_decay", 0)  # 梯度下降L2正则乘法
    pm.set_param("use_double", True)  # 是否使用双精度计算
    if pm.get_param("use_double"):
        torch.set_default_tensor_type(torch.DoubleTensor)
    sys_log.info('Software parameters added')
