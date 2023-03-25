"""
该程序在PyKinect2的基础上进行了拓展，功能有：
 - 获取rgb图像、深度图像和红外图像
 - 基于MapDepthFrameToColorSpace的rgb图像与深度图像配准
 - 基于IR相机与RGB相机旋转矩阵的rgb图像与深度图像配准
"""

from pykinect2 import PyKinectV2, PyKinectRuntime
from pykinect2.PyKinectV2 import *
from pykinect2.PyKinectRuntime import PyKinectRuntime
import numpy as np
import ctypes
import time
from numpy.lib import recfunctions as rfn


class Kinect(object):
    def __init__(self):
        self._kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                       PyKinectV2.FrameSourceTypes_Depth |
                                       PyKinectV2.FrameSourceTypes_Infrared)
        self._color_image = None
        self._color_time = None
        self._depth_image = None
        self._depth_time = None
        self._infrared_image = None
        self._infrared_time = None
        self._first_time = True
        # RGB相机内参，使用Camera Calibrator进行标定得到
        self._rgb_fx = 1.066685777221471e+03
        self._rgb_fy = 1.065830137390439e+03
        self._rgb_cx = 9.745830913731519e+02
        self._rgb_cy = 5.610585531660831e+02
        self._K_rgb = np.mat([[self._rgb_fx, 0, self._rgb_cx],
                              [0, self._rgb_fy, self._rgb_cy],
                              [0, 0, 1]])
        self._rgb_k1 = 0.048710073205385
        self._rgb_k2 = 0.277277363733280
        self._rgb_p1 = 0.003925356132610
        self._rgb_p2 = 0.005927693561727
        self._rhb_k3 = -1.525862455694073
        # IR相机内参，使用Camera Calibrator进行标定得到
        self._ir_fx = 3.650971835583717e+02
        self._ir_fy = 3.643048891305945e+02
        self._ir_cx = 2.659050639845913e+02
        self._ir_cy = 2.147089862934283e+02
        self._K_ir = np.mat([[self._ir_fx, 0, self._ir_cx],
                             [0, self._ir_fy, self._ir_cy],
                             [0, 0, 1]])
        # RGB相机外参，使用Camera Calibrator进行标定得到
        self._R_rgb = np.mat([[0.996922067259661, -0.037047121914319, -0.069093433614126],
                              [0.038735313143497, 0.998978859095504, 0.023255420785729],
                              [0.068161333073813, -0.025860197951905, 0.997339101226603]])
        self._t_rgb = np.mat([[-1.558693833585515e+02],
                              [-1.431357830959230e+02],
                              [5.219758749612480e+02]])
        # IR相机外参，使用Camera Calibrator进行标定得到
        self._R_ir = np.mat([[0.996423856419242, -0.041853391328798, -0.073401580315983],
                             [0.043807552771219, 0.998721671852402, 0.025217464038639],
                             [0.072252312619057, -0.028342826369678, 0.996983594405934]])
        self._t_ir = np.mat([[-2.109524028770616e+02],
                             [-1.424899033948746e+02],
                             [5.239325893629424e+02]])
        # 由IR相机到RGB相机的旋转矩阵和平移矩阵
        # self._r = self._K_rgb * self._R_rgb * self._R_ir.I * self._K_ir.I
        self._r = np.matmul(np.matmul(self._K_rgb, self._R_rgb), np.matmul(self._R_ir.I, self._K_ir.I))
        # self._t = self._K_rgb * self._t_rgb - self._K_rgb * self._R_rgb * self._R_ir.I * self._t_ir
        self._t = (np.matmul(self._K_rgb, self._t_rgb) -
                   np.matmul(np.matmul(self._K_rgb, self._R_rgb), np.matmul(self._R_ir.I, self._t_ir)))

    def get_last_color_frame_infor(self):
        if self._kinect.has_new_color_frame():
            # 这里获得的frame为一维ndarray
            frame = self._kinect.get_last_color_frame_data()
            # 获得得到frame的时间
            self._color_time = self._kinect.get_last_color_frame_time()
            # 返回4通道数据，alpha通道未注册
            reshape_frame = frame.reshape([self._kinect.color_frame_desc.Height,
                                           self._kinect.color_frame_desc.Width, 4])
            # 取出彩色图像数据
            self._color_image = reshape_frame[:, :, 0:3]
        return self._color_image, self._color_time

    def get_last_depth_frame_infor(self):
        if self._kinect.has_new_depth_frame():
            # 这里获得的frame也是一维ndarray
            frame = self._kinect.get_last_depth_frame_data()
            self._depth_time = self._kinect.get_last_depth_frame_time()
            reshape_frame = frame.reshape([self._kinect.depth_frame_desc.Height,
                                           self._kinect.depth_frame_desc.Width, 1])
            self._depth_image = reshape_frame[:, :, :]
        return self._depth_image, self._depth_time

    def get_last_infrared_frame_infor(self):
        if self._kinect.has_new_infrared_frame():
            frame = self._kinect.get_last_infrared_frame_data()
            self._infrared_time = self._kinect.get_last_infrared_frame_time()
            reshape_frame = frame.reshape([self._kinect.infrared_frame_desc.Height,
                                           self._kinect.infrared_frame_desc.Width, 1])
            self._infrared_image = reshape_frame[:, :, :]
        return self._infrared_image, self._infrared_time

    def get_last_multi_frame_infor(self):
        color_data = None
        color_time = None
        depth_data = None
        depth_time = None
        infrared_data = None
        infrared_time = None
        if self._first_time:
            start_time = time.time()
            while True:
                now_time = time.time()
                used_time = now_time - start_time
                color_infor = self.get_last_color_frame_infor()
                depth_infor = self.get_last_depth_frame_infor()
                infrared_infor = self.get_last_infrared_frame_infor()
                if color_infor[1] is not None \
                        and depth_infor[1] is not None \
                        and infrared_infor[1] is not None:
                    self._first_time = False
                    break
                elif used_time > 5:
                    raise RuntimeError('连接超时，请使用Kinect Configuration Verifier检查连接')
        else:
            color_data, color_time = self.get_last_color_frame_infor()
            depth_data, depth_time = self.get_last_depth_frame_infor()
            infrared_data, infrared_time = self.get_last_infrared_frame_infor()
        kinect_infor = [color_data, color_time, depth_data,
                        depth_time, infrared_data, infrared_time]
        return kinect_infor

    def map_depth_to_color_api(self):
        depth2color_points_type = _DepthSpacePoint * int(512 * 424)
        depth2color_points = ctypes.cast(depth2color_points_type(), ctypes.POINTER(_ColorSpacePoint))
        self._kinect._mapper.MapDepthFrameToColorSpace(
            ctypes.c_uint(512 * 424), self._kinect._depth_frame_data,
            ctypes.c_uint(512 * 424), depth2color_points)
        color_xy = np.copy(np.ctypeslib.as_array(depth2color_points, shape=(424 * 512, )))
        # 将color_xy由结构体numpy数组转化为普通numpy数组
        # color_xy = color_xy.view(np.float32).reshape(color_xy.shape + (-1,))
        color_xy = rfn.structured_to_unstructured(color_xy)
        color_xy = color_xy.reshape((424, 512, 2)).astype(int)
        # color_x = np.clip(color_xy[:, :, 0], 0, 1920 - 1)
        # color_y = np.clip(color_xy[:, :, 1], 0, 1080 - 1)
        color_x = color_xy[:, :, 0]
        color_y = color_xy[:, :, 1]
        return color_x, color_y

    def map_depth_to_color_rt(self):
        color_xy = []
        for row in range(424):
            for col in range(512):
                z_ir = self._depth_image[row, col, 0]
                p_ir = np.mat([[row],
                               [col],
                               [1]])
                z_rgb_p_rgb = np.multiply(z_ir, np.matmul(self._r, p_ir)) - self._t
                p_rgb = [z_rgb_p_rgb[0] / z_rgb_p_rgb[2], z_rgb_p_rgb[1] / z_rgb_p_rgb[2]]
                color_xy.extend(p_rgb)
        color_xy = np.array(color_xy)
        color_xy = color_xy.reshape((424, 512, 2)).astype(int)
        # color_x = np.clip(color_xy[:, :, 1], 0, 1920 - 1)
        # color_y = np.clip(color_xy[:, :, 0], 0, 1080 - 1)
        color_x = color_xy[:, :, 1]
        color_y = color_xy[:, :, 0]
        return color_x, color_y
