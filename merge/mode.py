# @Author        : Justin Lee
# @Time          : 2025-3-27

from enum import Enum

'''
    用户模式的枚举类
'''


class User_Mode(Enum):
    WEB = 0  # web界面模式
    LOCAL_ENROLL = 1  # 本地录入mos
    LOCAL_RECOGNIZE = 2  # 本地识别模式
