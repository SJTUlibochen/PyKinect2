"""
该程序在PyKinect2的基础上进行了进一步封装，功能有：
 - 获取rgb图像、深度图像和红外图像
 - 实现rgb图像与深度图像的配准
"""

from pykinect2 import PyKinectV2, PyKinectRuntime
from pykinect2.PyKinectV2 import *
from pykinect2.PyKinectRuntime import PyKinectRuntime
import numpy as np
import ctypes
import time


class Kinect(object):
    def __init__(self):
        self._kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                       PyKinectV2.FrameSourceTypes_Depth |
                                       PyKinectV2.FrameSourceTypes_Infrared)
        self._color_data = None
        self._depth_data = None
        self._infrared_data = None
        self._align_color_data = None
        self._color_time = None
        self._depth_time = None
        self._infrared_time = None
        self._first_time = True

    def get_last_color_data(self):
        if self._kinect.has_new_color_frame():
            # 这里获得的frame为一维ndarray
            frame = self._kinect.get_last_color_frame()
            # 获得得到frame的时间
            t_color = time.time()
            # 返回4通道数据，alpha通道未注册
            reshape_frame = frame.reshape([self._kinect.color_frame_desc.Height,
                                           self._kinect.color_frame_desc.Width, 4])
            # 取出彩色图像数据
            d_color = reshape_frame[:, :, 0:3]
        else:
            d_color = None
            t_color = None
        return d_color, t_color

    def get_last_depth_data(self):
        if self._kinect.has_new_depth_frame():
            # 这里获得的frame也是一维ndarray
            frame = self._kinect.get_last_depth_frame()
            t_depth = time.time()
            reshape_frame = frame.reshape([self._kinect.depth_frame_desc.Height,
                                           self._kinect.depth_frame_desc.Width, 1])
            d_depth = reshape_frame[:, :, :]
        else:
            d_depth = None
            t_depth = None
        return d_depth, t_depth

    def get_last_infrared_data(self):
        if self._kinect.has_new_infrared_frame():
            frame = self._kinect.get_last_infrared_frame()
            t_infrared = time.time()
            reshape_frame = frame.reshape([self._kinect.infrared_frame_desc.Height,
                                           self._kinect.infrared_frame_desc.Width, 1])
            d_infrared = reshape_frame[:, :, :]
        else:
            d_infrared = None
            t_infrared = None
        return d_infrared, t_infrared

    def get_last_color_depth_infrared_data(self):
        if self._first_time:
            start_time = time.time()
            while True:
                now_time = time.time()
                used_time = now_time - start_time
                color_infor = self.get_last_color_data()
                depth_infor = self.get_last_depth_data()
                infrared_infor = self.get_last_infrared_data()
                if color_infor[1] is not None \
                        and depth_infor[1] is not None \
                        and infrared_infor[1] is not None:
                    self._first_time = False
                    break
                elif used_time > 5:
                    raise RuntimeError('连接超时，请使用Kinect Configuration Verifier检查连接')
        else:
            self._color_data, self._color_time = self.get_last_color_data()
            self._depth_data, self._depth_time = self.get_last_depth_data()
            self._infrared_data, self._infrared_time = self.get_last_infrared_data()
        kinect_infor = [self._color_data, self._color_time, self._depth_data,
                        self._depth_time, self._infrared_data, self._infrared_time]
        return kinect_infor

    def match_depth_and_color(self):
        depth2color_points_type = _DepthSpacePoint * int(512 * 424)
        depth2color_points = ctypes.cast(depth2color_points_type(), ctypes.POINTER(_ColorSpacePoint))
        self._kinect._mapper.MapDepthFrameToColorSpace(
            ctypes.c_uint(512 * 424), self._kinect._depth_frame_data,
            ctypes.c_uint(512 * 424), depth2color_points)
        color_xy = np.copy(np.ctypeslib.as_array(depth2color_points, shape=(424 * 512,)))
        color_xy = color_xy.view(np.float32).reshape(color_xy.shape + (-1,))
        color_xy = color_xy.reshape(424, 512, 2).astype(int)
        color_x = np.clip(color_xy[:, :, 0], 0, 1920 - 1)
        color_y = np.clip(color_xy[:, :, 1], 0, 1080 - 1)
        return color_x, color_y
