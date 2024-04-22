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
全局参数管理器: 参数全局传递，修改，删除，保存和加载
"""
import logging
import os
import numpy as np

current_name = os.path.basename(__file__)  # 当前模块名字
sys_log = logging.getLogger('logger')


def init():
    """全局参数管理器初始化"""
    global _global_dict
    _global_dict = {}
    sys_log.info('Global parameters dictionary created')


def set_param(key, value):
    """定义一个全局参数"""
    _global_dict[key] = value


def get_param(key):
    """获得一个全局参数，不存在则提示读取对应参数失败"""
    try:
        return _global_dict[key]
    except KeyError:
        _str = 'Read parameter”' + key + '“failed!'
        sys_log.error(_str)
        return 0


def print_param(key):
    """输出指定参数"""
    try:
        param_type = str(type(_global_dict[key])).replace('<class ', '').replace('>', '')
        _str = '"' + key + '"=' + str(_global_dict[key]) + '  Type=' + param_type
        sys_log.info(_str)
    except KeyError:
        _str = 'Print parameter”' + key + '“failed!'
        sys_log.error(_str)


def print_all_param():
    """输出所有参数"""
    sys_log.debug('Start printing all parameters')
    for key in _global_dict:
        param_type = str(type(_global_dict[key])).replace('<class ', '').replace('>', '')
        _str = '"' + key + '"=' + str(_global_dict[key]) + '  Type=' + param_type
        sys_log.info(_str)
    sys_log.debug('All parameters printed')


def write_param():
    """输出参数到文件"""
    file_name = './parameters/' + 'AllParam.npy'
    np.save(file_name, _global_dict)
    sys_log.info('Params saved in' + file_name)


def load_param():
    """从文件中读取参数"""
    try:
        global _global_dict
        _global_dict = np.load('./parameters/AllParam.npy', allow_pickle=True)
        sys_log.info('Params loaded from /parameters/AllParam.npy')
        print_all_param()
    except FileNotFoundError:
        sys_log.error('Nonexistent input parameters file!')


def delete_param(key):
    """删除指定参数"""
    try:
        _global_dict.pop(key)
    except KeyError:
        _str = 'Delete parameter”' + key + '“failed!'
        sys_log.error(_str)


def clear_param():
    """删除所有参数"""
    _global_dict.clear()
    sys_log.warning('All parameters deleted')
