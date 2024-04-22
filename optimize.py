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
2024.4.11      Huang Yu

功能描述
------------------------------------------------------------------------------------------------------------------------
利用梯度下降法求解结果
"""
import logging
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

import circuit_net
import param_man as pm
import sys_logger
from sys_logger import TqdmToLogger

sys_log = logging.getLogger('logger')


def dg_optimize(net):
    """利用梯度下降法求解结果"""
    size = [pm.get_param("pixel_row"), pm.get_param("pixel_col")]  # panel尺寸
    device = pm.get_param("Device")  # 运算设备
    loss_array = np.zeros([pm.get_param('dg_itr'), 1])  # Loss数组
    v = torch.zeros(size, requires_grad=True, device=pm.get_param("Device"))  # 节点电压

    """Boundary condition"""
    point = pm.get_param("elvdd_point")  # ELVDD点坐标
    elvdd = pm.get_param("elvdd")  # ELVDD大小
    with torch.no_grad():
        # v = pm.get_param("elvdd_ini") * torch.ones(size, device=pm.get_param("Device"))  # 初始电压
        # v = 2 * torch.ones(size, device=pm.get_param("Device"))  # 初始电压
        v = torch.zeros(size, device=pm.get_param("Device"))  # 初始电压
        # v = torch.rand(size, device=pm.get_param("Device")) * 2. + 2.  # 初始电压
        # v[0, 1] = 5 - 1.5e-4
        # v[1, 0] = 5 - 1.5e-4
        # v[1, 1] = 5 - 2e-4
        for i in range(pm.get_param("elvdd_num")):  # 边界修正
            v[point[i][0], point[i][1]] = elvdd[i]
    v.requires_grad = True

    """loss function and optimizer"""
    criterion = loss_function_prepare()  # 损失函数定义
    # optimizer = optim.Adam([v], lr=pm.get_param('dg_lr'), weight_decay=pm.get_param('dg_decay'), amsgrad=True)
    optimizer = optim.AdamW([v], lr=pm.get_param('dg_lr'), weight_decay=pm.get_param('dg_decay'), amsgrad=True)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9862)

    """gradient descent"""
    sys_log.debug("Gradient descent start")
    v_data = circuit_net.get_vdata().to(device)  # 获得vdata
    tqdm_out = TqdmToLogger(sys_log, level=sys_logger.LOG_LEVEL_PROCESS)
    process_bar = tqdm(range(pm.get_param('dg_itr')), desc='Gradient Descent', leave=True, file=tqdm_out)
    for i in process_bar:
        if i < 100:
            optimizer.param_groups[0]['lr'] = 1
        elif i < 200:
            optimizer.param_groups[0]['lr'] = 0.9
        elif i < 300:
            optimizer.param_groups[0]['lr'] = 0.8
        elif i < 400:
            optimizer.param_groups[0]['lr'] = 0.5
        elif i < 600:
            optimizer.param_groups[0]['lr'] = 0.3
        elif i < 800:
            optimizer.param_groups[0]['lr'] = 0.3
        else:
            optimizer.param_groups[0]['lr'] = 0.05
        loss_, current_loss, current_grid, current_active = net(v_data, v)  # 正向传播
        loss = criterion(loss_, torch.zeros(size).to(device))  # 计算损失函数
        optimizer.zero_grad()  # 梯度置0
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权值
        with torch.no_grad():
            # 裁剪电压，对解添加约束
            v[v < 0] = 0
            v[v > 5] = 5
        loss_array[i, 0] = loss.item()
        process_bar.set_postfix(loss=loss_array[i, 0], iteration=i+1, lr=optimizer.param_groups[0]['lr'])
    process_bar.close()
    loss_, current_loss, current_grid, current_active = net(v_data, v)  # 获得最终的准确结果
    sys_log.debug("Gradient descent end")
    return v, loss_array, current_grid, current_active


def loss_function_prepare():
    """定义损失函数"""
    if pm.get_param('dg_loss_type') == 'L1Loss':
        criterion = nn.L1Loss(reduction='sum')  # reduction='sum'
    elif pm.get_param('dg_loss_type') == 'MSELoss':
        criterion = nn.MSELoss(reduction='sum')
    elif pm.get_param('dg_loss_type') == 'NLLLoss':
        criterion = nn.NLLLoss(reduction='sum')
    elif pm.get_param('dg_loss_type') == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss(reduction='sum')
    else:
        criterion = nn.MSELoss(reduction='sum')
        sys_log.critical('Unknown target loss function,loss function is allocated as MSELoss')
    sys_log.info('Loss function prepared')
    return criterion


def adam_optimizer_prepare(param):
    """定义梯度下降的optimizer - adam"""
    optimizer = optim.Adam(param, lr=pm.get_param('dg_lr'), weight_decay=pm.get_param('dg_decay'), amsgrad=True)
    sys_log.info('Optimizer prepared')
    return optimizer
