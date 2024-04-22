# -*- coding: utf-8 -*-
"""
文件头信息
------------------------------------------------------------------------------------------------------------------------
[开发人员]: Huang Yu\n
[创建日期]: 2023.11.20\n
[目标工具]: PyCharm\n
[版权信息]: © Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab All Rights Reserved\n

版本修改
------------------------------------------------------------------------------------------------------------------------
[Date]         [By]         [Version]         [Change Log]\n
2023.11.20     Huang Yu     1.0               first implementation\n
2024.4.11      Huang Yu     1.1               add process level and tqdm to logger

功能描述
------------------------------------------------------------------------------------------------------------------------
带颜色的Logger，输出信息至Console中方便调试。对于5个层次的信息，模块定义：\n
'DEBUG': 'white'\n
'PROCESS': 'blue'\n
'INFO': 'green'\n
'WARNING': 'yellow'\n
'ERROR': 'red'\n
'CRITICAL': 'bold_red'
"""
import io
import logging
import colorlog

LOG_LEVEL_PROCESS = 25
logging.addLevelName(LOG_LEVEL_PROCESS, "PROCESS")  # 设置自定义日志级别


def _log_process(msg, *args, **kwargs):
    if logging.getLogger(__name__).isEnabledFor(LOG_LEVEL_PROCESS):
        logging.log(LOG_LEVEL_PROCESS, msg, *args, **kwargs)


logging.process = _log_process


class Logger:
    def __init__(self):
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)
        self.config = {
            'DEBUG': 'white',
            'PROCESS': 'bold_blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        fh = logging.FileHandler(filename='./log/sys_log.txt', mode='w')
        fh.setLevel(logging.DEBUG)
        log_format = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> [%(levelname)s] : %(message)s',
            datefmt='%H:%M',
            log_colors=self.config
        )
        sh.setFormatter(log_format)
        fh.setFormatter(log_format)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)
