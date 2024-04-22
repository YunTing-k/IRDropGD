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
利用梯度下降法求解带有源器件的矩形电阻网络IR-Drop分布，电阻网络格点之间连通度最大固定为4，网络大小至少为[2x2]
"""
import winsound
import scipy.io as sio
import circuit_net
import optimize
import sys_logger
import param_man as pm
import parameters

sys_log = sys_logger.Logger().logger  # logger

if __name__ == '__main__':
    """Program start"""
    sys_log.debug('Program start')

    """Parameter initialization"""
    pm.init()  # 构建全局参数字典
    parameters.set_default_param()  # 构建仿真参数

    """Build network"""
    device = pm.get_param("Device")  # 运算设备
    size = [pm.get_param("pixel_row"), pm.get_param("pixel_col")]  # panel尺寸
    type_mat = circuit_net.grid_type().to(device)  # 格点类型矩阵
    conduct = circuit_net.build_conduct().to(device)  # 电阻矩阵
    weight = circuit_net.get_weight().to(device)  # 权重矩阵
    if not pm.get_param("glb_ird"):
        net = circuit_net.local_net(size=size, type_mat=type_mat, conduct=conduct, weight=weight).to(device)  # 构建网络
    else:
        net = circuit_net.local_net(size=size, type_mat=type_mat, conduct=conduct, weight=weight).to(device)  # 构建网络
    v, loss_array, current_grid, current_active = optimize.dg_optimize(net)  # 优化

    """Store results"""
    data = {'v': v.detach().cpu().numpy(),
            'loss': loss_array,
            'i_grid': current_grid.detach().cpu().numpy(),
            'i_pixel': current_active.detach().cpu().numpy()}  # 创建空字典
    sio.savemat('./results/data.mat', data)

    """Program end"""
    pm.write_param()
    sys_log.debug('Program end')
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
