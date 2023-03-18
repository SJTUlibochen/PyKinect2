# -*- coding：utf-8 -*-
"""
time:2019/5/1 19:34
author:Lance
organization: HIT
contact: QQ:261983626 , wechat:yuan261983626
——————————————————————————————
description：
$ 自己基于Pykinect2 写的一个Kinect的类。
主要包括：
彩色图像、深度图像、红外图像的获取
彩色图像和深度图像的坐标空间互换
——————————————————————————————
note：
$
"""
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import ctypes
import math
import cv2 as cv
import time
import copy


class Kinect(object):
    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
        self.depth_ori = None
        self.infrared_frame = None
        self.color_frame = None
        self.w_color = 1920
        self.h_color = 1080
        self.w_depth = 512
        self.h_depth = 424
        """————————————————(2019/5/10)——————————————————"""
        self.csp_type = _ColorSpacePoint * int(1920 * 1080)
        self.csp = ctypes.cast(self.csp_type(), ctypes.POINTER(_DepthSpacePoint))
        """————————————————(2019/9/4)——————————————————"""
        self.color = None
        self.depth = None
        self.depth_draw = None
        self.color_draw = None
        self.infrared = None
        self.first_time = True

    """————————————————(2019/5/1)——————————————————"""
    """获取最新的图像数据"""

    def get_the_last_color(self):
        """
        Time :2019/5/1
        FunC:获取最新的图像数据
        Input:无
        Return:无
        """
        if self._kinect.has_new_color_frame():
            # 获得的图像数据是二维的，需要转换为需要的格式
            frame = self._kinect.get_last_color_frame()
            # 返回的是4通道，还有一通道是没有注册的
            gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
            # 取出彩色图像数据
            self.color_frame = gbra[:, :, 0:3]
            return self.color_frame

    """————————————————(2019/5/1)——————————————————"""
    """获取最新的深度数据"""

    def get_the_last_depth(self):
        """
        Time :2019/5/1
        FunC:获取最新的图像数据
        Input:无
        Return:无
        """
        if self._kinect.has_new_depth_frame():
            # 获得深度图数据
            frame = self._kinect.get_last_depth_frame()
            # 转换为图像排列
            image_depth_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                             self._kinect.depth_frame_desc.Width])
            self.depth_ori = image_depth_all

            return self.depth_ori

    """————————————————(2019/5/1)——————————————————"""
    """获取最新的红外数据"""

    def get_the_last_infrared(self):
        """
        Time :2019/5/1
        FunC:获取最新的图像数据
        Input:无
        Return:无
        """
        if self._kinect.has_new_infrared_frame():
            # 获得深度图数据
            frame = self._kinect.get_last_infrared_frame()
            # 转换为图像排列
            image_infrared_all = frame.reshape([self._kinect.infrared_frame_desc.Height,
                                                self._kinect.infrared_frame_desc.Width])
            self.infrared_frame = image_infrared_all
            return self.infrared_frame

    """————————————————(2019/5/1)——————————————————"""
    """将深度像素点匹配到彩色图像中"""

    def map_depth_point_to_color_point(self, depth_point):
        """
        Time :2019/5/1
        FunC: 将深度图像坐标映射到彩色坐标中，
        Input: depth_points:深度像素点，列表、数组格式,图像坐标，且为人眼视角
        Return: color_points:对应的彩色像素点，列表格式
        均采用图像坐标的形式
        """
        depth_point_to_color = copy.deepcopy(depth_point)
        n = 0
        while 1:
            self.get_the_last_depth()
            self.get_the_last_color()
            if self.depth_ori is None:
                continue
            color_point = self._kinect._mapper.MapDepthPointToColorSpace(
                _DepthSpacePoint(511 - depth_point_to_color[1], depth_point_to_color[0]),
                self.depth_ori[depth_point_to_color[0], 511 - depth_point_to_color[1]])
            """————————————————(2019/6/22)——————————————————"""
            # cobot 第二次培训更改
            # color_point = self._kinect._mapper.MapDepthPointToColorSpace(
            #     _DepthSpacePoint(depth_point[0], depth_point[1]), self.depth[depth_point[1], depth_point[0]])
            """————————————————(2019/6/22)——————————————————"""
            if math.isinf(float(color_point.y)):
                n += 1
                if n >= 50000:
                    print('深度映射彩色，无效点')
                    color_point = [0, 0]
                    break
            else:
                color_point = [np.int0(color_point.y), 1920 - np.int0(color_point.x)]  # 图像坐标，人眼视角
                break
        return color_point

    """————————————————(2019/5/10)——————————————————"""
    """将彩色像素数组映射到深度图像中"""

    def map_color_points_to_depth_points(self, color_points):
        """
        Time :2019/5/1
        FunC: 将深度图像坐标映射到彩色坐标中，输入的为图像坐标
        Input: depth_points:彩色像素点
        Return: color_points:对应的深度像素点，图像坐标
        """
        self.get_the_last_depth()
        self.get_the_last_color()
        self._kinect._mapper.MapColorFrameToDepthSpace(
            ctypes.c_uint(512 * 424), self._kinect._depth_frame_data, ctypes.c_uint(1920 * 1080), self.csp)
        depth_points = [self.map_color_point_to_depth_point(x, True) for x in color_points]
        return depth_points

    """————————————————(2019/5/1)——————————————————"""
    """将彩色像素点映射到深度图像中"""

    def map_color_point_to_depth_point(self, color_point, if_call_flg=False):
        """
        Time :2019/5/1
        FunC: 将深度图像坐标映射到彩色坐标中，输入的为图像坐标
        Input: depth_points:彩色像素点
        Return: color_points:对应的深度像素点，图像坐标
        """
        n = 0
        color_point_to_depth = copy.deepcopy(color_point)
        color_point_to_depth[1] = 1920 - color_point_to_depth[1]
        while 1:
            self.get_the_last_depth()
            self.get_the_last_color()
            # self.depth = cv.medianBlur(image_depth_all, 5)
            if not if_call_flg:
                self._kinect._mapper.MapColorFrameToDepthSpace(
                    ctypes.c_uint(512 * 424), self._kinect._depth_frame_data, ctypes.c_uint(1920 * 1080), self.csp)
            if math.isinf(float(self.csp[color_point_to_depth[0] * 1920 + color_point_to_depth[1] - 1].y)) or np.isnan(
                    self.csp[color_point_to_depth[0] * 1920 + color_point_to_depth[1] - 1].y):
                n += 1
                if n >= 50000:
                    print('彩色映射深度，无效的点')
                    depth_point = [0, 0]
                    break
            else:
                self.cor = self.csp[color_point_to_depth[0] * 1920 + color_point_to_depth[1] - 1].y
                try:
                    depth_point = [np.int0(self.csp[color_point_to_depth[0] * 1920 + color_point_to_depth[1] - 1].y),
                                   np.int0(self.csp[color_point_to_depth[0] * 1920 + color_point_to_depth[1] - 1].x)]
                except OverflowError as e:
                    print('彩色映射深度，无效的点')
                    depth_point = [0, 0]
                break
        depth_point[1] = 512 - depth_point[1]
        return depth_point

        # depth_points = [self._kinect._mapper.MapColorPointToDepthSpace(_ColorSpacePoint(color_point[0],color_point[1]),self.color_frame[depth_point]))
        #                 for depth_point in depth_points]
        # return color_points

    """————————————————(2019/4/26)——————————————————"""
    """获得最新的彩色和深度图像以及红外图像"""

    def get_the_data_of_color_depth_infrared_image(self, Infrared_threshold=16000):
        """
        # ————————查看是否有新的一帧————————
        :return:
        """
        # 访问新的RGB帧
        time_s = time.time()
        if self.first_time:
            while 1:
                n = 0
                if self._kinect.has_new_color_frame():
                    #                 # 获得的图像数据是二维的，需要转换为需要的格式
                    frame = self._kinect.get_last_color_frame()
                    # 返回的是4通道，还有一通道是没有注册的
                    gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
                    # 取出彩色图像数据
                    # self.color = gbra[:, :, 0:3]
                    self.color = gbra[:, :, 0:3][:, ::-1, :]
                    # # 这是因为在python中直接复制该图像的效率不如直接再从C++中获取一帧来的快
                    # frame = self._kinect.get_last_color_frame()
                    # # 返回的是4通道，还有一通道是没有注册的
                    # gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
                    # # 取出彩色图像数据
                    # # self.color_draw = gbra[:, :, 0:3][:,::-1,:]
                    # self.color_draw = gbra[:, :, 0:3][:, ::-1, :]
                    n += 1
                # 访问新的Depth帧
                if self._kinect.has_new_depth_frame():
                    # 获得深度图数据
                    frame = self._kinect.get_last_depth_frame()
                    # 转换为图像排列
                    image_depth_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                                     self._kinect.depth_frame_desc.Width])
                    # 转换为（n，m，1） 形式
                    image_depth_all = image_depth_all.reshape(
                        [self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 1])
                    self.depth_ori = np.squeeze(image_depth_all)
                    self.depth = np.squeeze(image_depth_all)[:, ::-1]

                    """————————————————(2019/5/11)——————————————————"""
                    # # 获得深度图数据
                    # frame = self._kinect.get_last_depth_frame()
                    # # 转换为图像排列
                    # depth_all_draw = frame.reshape([self._kinect.depth_frame_desc.Height,
                    #                                 self._kinect.depth_frame_desc.Width])
                    # # 转换为（n，m，1） 形式
                    # depth_all_draw = depth_all_draw.reshape(
                    #     [self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 1])
                    # depth_all_draw[depth_all_draw >= 1500] = 0
                    # depth_all_draw[depth_all_draw <= 500] = 0
                    # depth_all_draw = np.uint8(depth_all_draw / 1501 * 255)
                    # self.depth_draw = depth_all_draw[:, ::-1, :]
                    n += 1
                # 获取红外数据
                if self._kinect.has_new_infrared_frame():
                    # 获得深度图数据
                    frame = self._kinect.get_last_infrared_frame()
                    # 转换为图像排列
                    image_infrared_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                                        self._kinect.depth_frame_desc.Width])
                    # 转换为（n，m，1） 形式
                    # image_infrared_all[image_infrared_all > Infrared_threshold] = 0
                    # image_infrared_all = image_infrared_all / Infrared_threshold * 255
                    self.infrared = image_infrared_all[:, ::-1]
                    n += 1
                t = time.time() - time_s
                if n == 3:
                    self.first_time = False
                    break
                elif t > 5:
                    print('未获取图像数据，请检查Kinect2连接是否正常')
                    break
        else:
            if self._kinect.has_new_color_frame():
                #                 # 获得的图像数据是二维的，需要转换为需要的格式
                frame = self._kinect.get_last_color_frame()
                # 返回的是4通道，还有一通道是没有注册的
                gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
                # 取出彩色图像数据
                # self.color = gbra[:, :, 0:3]
                self.color = gbra[:, :, 0:3][:, ::-1, :]
                # # 这是因为在python中直接复制该图像的效率不如直接再从C++中获取一帧来的快
                # frame = self._kinect.get_last_color_frame()
                # # 返回的是4通道，还有一通道是没有注册的
                # gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
                # # 取出彩色图像数据
                # # self.color_draw = gbra[:, :, 0:3][:,::-1,:]
                # self.color_draw = gbra[:, :, 0:3][:, ::-1, :]

            # 访问新的Depth帧
            if self._kinect.has_new_depth_frame():
                # 获得深度图数据
                frame = self._kinect.get_last_depth_frame()
                # 转换为图像排列
                image_depth_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                                 self._kinect.depth_frame_desc.Width])
                # 转换为（n，m，1） 形式
                image_depth_all = image_depth_all.reshape(
                    [self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 1])
                self.depth_ori = np.squeeze(image_depth_all)
                self.depth = np.squeeze(image_depth_all)[:, ::-1]

                # """————————————————(2019/5/11)——————————————————"""
                # # 获得深度图数据
                # frame = self._kinect.get_last_depth_frame()
                # # 转换为图像排列
                # depth_all_draw = frame.reshape([self._kinect.depth_frame_desc.Height,
                #                                 self._kinect.depth_frame_desc.Width])
                # # 转换为（n，m，1） 形式
                # depth_all_draw = depth_all_draw.reshape(
                #     [self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 1])
                # depth_all_draw[depth_all_draw >= 1500] = 0
                # depth_all_draw[depth_all_draw <= 500] = 0
                # depth_all_draw = np.uint8(depth_all_draw / 1501 * 255)
                # self.depth_draw = depth_all_draw[:, ::-1, :]

            # 获取红外数据
            if self._kinect.has_new_infrared_frame():
                # 获得深度图数据
                frame = self._kinect.get_last_infrared_frame()
                # 转换为图像排列
                image_infrared_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                                    self._kinect.depth_frame_desc.Width])
                # 转换为（n，m，1） 形式
                # image_infrared_all[image_infrared_all > Infrared_threshold] = 0
                # image_infrared_all = image_infrared_all / Infrared_threshold * 255
                self.infrared = image_infrared_all[:, ::-1]

        return self.color, self.color_draw, self.depth, self.depth_draw, self.infrared
        # return self.color, self.depth


if __name__ == '__main__':
    a = Kinect()
    while 1:
        t = time.time()
        color_data = a.get_the_data_of_color_depth_infrared_image()
        cv.imshow('a', color_data[4])
        # cv.imwrite("test.png", color_data[4])
        cv.waitKey(1)
