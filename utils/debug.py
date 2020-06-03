# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Face-Recognize-Pipeline
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""
import os
import datetime
import logging
import threading
from logging.handlers import TimedRotatingFileHandler
from memory_profiler import profile


def set_format(handler, format):
    # handler.suffix = "%Y%m%d"
    if format:
        logFormatter = logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(levelname)s: %(message)s",
                                         "%Y-%m-%d %H:%M:%S")
    else:
        logFormatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(logFormatter)


def set_logger(name, level="info", logfile=None, format=False):
    """
    logger = set_logging(name="LOG", level="debug", logfile="log.txt", format=False)
    url:https://cuiqingcai.com/6080.html
    level级别：debug>info>warning>error>critical
    :param level: 设置log输出级别
    :param logfile: log保存路径，如果为None，则在控制台打印log
    :return:
    """
    logger = logging.getLogger(name)
    if logfile and os.path.exists(logfile):
        os.remove(logfile)
    # define a FileHandler write messages to file
    if logfile:
        # filehandler = logging.handlers.RotatingFileHandler(filename="./log.txt")
        filehandler = logging.handlers.TimedRotatingFileHandler(logfile, when="midnight", interval=1)
        set_format(filehandler, format)
        logger.addHandler(filehandler)

    # define a StreamHandler print messages to console
    console = logging.StreamHandler()
    set_format(console, format)
    logger.addHandler(console)
    # set initial log level
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    if level == 'info':
        logger.setLevel(logging.INFO)
    if level == 'warning':
        logger.setLevel(logging.WARN)
    if level == 'critical':
        logger.setLevel(logging.CRITICAL)
    if level == 'fatal':
        logger.setLevel(logging.FATAL)
    logger.info("Init log in %s level", level)
    return logger


def RUN_TIME(deta_time):
    '''
    计算时间差，返回毫秒,deta_time.seconds获得秒数=1000ms，deta_time.microseconds获得微妙数=1/1000ms
    :param deta_time: ms
    :return:
    '''
    time_ = deta_time.seconds * 1000 + deta_time.microseconds / 1000.0
    return time_


def TIME():
    '''
    获得当前时间
    :return:
    '''
    return datetime.datetime.now()


def run_time_decorator(title=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            T0 = TIME()
            result = func(*args, **kwargs)
            T1 = TIME()
            print("{}-- function : {}-- rum time : {}ms ".format(title, func.__name__, RUN_TIME(T1 - T0)))
            # logger.debug("{}-- function : {}-- rum time : {}s ".format(title, func.__name__, RUN_TIME(T1 - T0)/1000.0))
            return result

        return wrapper

    return decorator


@profile(precision=4)
def memory_test():
    """
    1.先导入：
    > from memory_profiler import profile
    2.函数前加装饰器：
    > @profile(precision=4,stream=open('memory_profiler.log','w+'))
　　　参数含义：precision:精确到小数点后几位
　　　stream:此模块分析结果保存到 'memory_profiler.log' 日志文件。如果没有此参数，分析结果会在控制台输出
    :return:
    """
    c = 0
    for item in range(10):
        c += 1
        # logger.error("c:{}".format(c))
    # print(c)


if __name__ == '__main__':
    logger = set_logger(name="LOG", level="debug", logfile="log.txt", format=False)
    # T0 = TIME()
    # do something
    # T1 = TIME()
    # print("rum time:{}ms".format(RUN_TIME(T1 - T0)))
    # t_logger = set_logging(name=__name__, level="info", logfile=None)
    # t_logger.debug('debug')
    # t_logger.info('info')
    # t_logger.warning('Warning exists')
    # t_logger.error('Finish')
    # memory_test()
    logger1 = set_logger(name="LOG", level="debug", logfile="log.txt", format=False)
    logger1.info("---" * 20)
    logger1.info("work_space:{}".format("work_dir"))
