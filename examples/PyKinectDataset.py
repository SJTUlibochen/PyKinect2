"""
该程序是基于PyKinect2生成ORB_SLAM3所需要的RGB-D数据集
所生成的数据在以时间命名的文件夹“年-月-日 时-分-秒”中，文件结构：
 - folder 'depth': depth images, 512 * 424 16-bit PNG
 - folder 'rgb': rgb images, 512 * 424 24-bit PNG
 - file 'depth.txt'
 - file 'rgb.txt'
 - file 'association.txt'
深度图像与RGB图像已经配准，即两者的同一像素点代表同一空间位置

使用'Esc'键结束数据集生成
"""

from pykinect2 import PyKinectV2, PyKinectRuntime
from pykinect2.PyKinectV2 import *
from pykinect2.PyKinectRuntime import PyKinectRuntime
import cv2 as cv
import numpy as np
import ctypes
import time
import os


class KinectDataset(object):
    def __init__(self):
        self._kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                       PyKinectV2.FrameSourceTypes_Depth)
        self._color_data = None
        self._depth_data = None
        self._align_color_data = None
        self._color_time = None
        self._depth_time = None
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

    def get_last_color_depth_data(self):
        if self._first_time:
            start_time = time.time()
            while True:
                now_time = time.time()
                used_time = now_time - start_time
                if self.get_last_color_data() is not None and self.get_last_depth_data() is not None:
                    self._first_time = False
                    break
                elif used_time > 5:
                    raise RuntimeError('连接超时，请使用Kinect Configuration Verifier检查连接')
        else:
            self._color_data, self._color_time = self.get_last_color_data()
            self._depth_data, self._depth_time = self.get_last_depth_data()
        return self._color_data, self._color_time, self._depth_data, self._depth_time

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


def read_txt_file(path_to_txt: str):
    with open(path_to_txt, 'r') as t:
        txt = t.read()
        lines = txt.replace(',', '').replace('\t', '').split('\n')
        txt_list = [[seg.strip() for seg in line.split(' ') if seg.strip() != '']
                    for line in lines if len(line) > 0 and line[0] != '#']
        txt_list = [(float(line[0]), line[1:]) for line in txt_list if len(line) > 0]
        txt_dict = dict(txt_list)
        return txt_dict


def associate_color_depth(col_txt_dict: dict, dep_txt_dict: dict,
                          offset: float = 0, max_difference: float = 0.2):
    color_time_list = list(col_txt_dict)
    depth_time_list = list(dep_txt_dict)
    potential_match = [(abs(col - (dep + offset)), col, dep)
                       for col in depth_time_list
                       for dep in color_time_list
                       if abs(col - (dep + offset)) < max_difference]
    match = []
    for diff, col, dep in potential_match:
        if col in color_time_list and dep in depth_time_list:
            color_time_list.remove(col)
            depth_time_list.remove(dep)
            match.append((col, dep))
    match.sort()
    return match


if __name__ == '__main__':
    kinect = KinectDataset()
    dataset_folder_name = str(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()))
    dataset_folder_path = '../datasets/%s' % dataset_folder_name
    color_path = '%s/rgb' % dataset_folder_path
    color_txt = '%s.txt' % color_path
    depth_path = '%s/depth' % dataset_folder_path
    depth_txt = '%s.txt' % depth_path
    if not os.path.exists(dataset_folder_path):
        print('--- new dataset folder ---')
        os.makedirs(dataset_folder_path)
        os.makedirs(color_path)
        os.makedirs(depth_path)
        print('new dataset folder' + dataset_folder_path + 'created!')
    print('--- show rgb & depth image ---')
    while True:
        data = kinect.get_last_color_depth_data()
        align_color_coord = kinect.match_depth_and_color()
        ori_color_data = data[0]
        ori_depth_data = data[2]
        color_time = data[1]
        depth_time = data[3]
        if ori_color_data is not None and ori_depth_data is not None:
            align_color_data = ori_color_data[align_color_coord[1], align_color_coord[0], 0:3]
            cv.namedWindow('ori_color', cv.WINDOW_AUTOSIZE)
            cv.imshow('ori_color', ori_color_data)
            cv.namedWindow('align_color', cv.WINDOW_AUTOSIZE)
            cv.imshow('align_color', align_color_data)
            color_file = '%s/%f.png' % (color_path, color_time)
            cv.imwrite(color_file, align_color_data)
            color_txt_content = '%f rgb/%f.png' % (color_time, color_time)
            with open(color_txt, 'a+') as c:
                c.write(color_txt_content + '\n')
            cv.namedWindow('depth', cv.WINDOW_AUTOSIZE)
            cv.imshow('depth', ori_depth_data)
            depth_file = '%s/%f.png' % (depth_path, depth_time)
            cv.imwrite(depth_file, ori_depth_data)
            depth_txt_content = '%f depth/%f.png' % (depth_time, depth_time)
            with open(depth_txt, 'a+') as d:
                d.write(depth_txt_content + '\n')
        if cv.waitKey(1) == 27:
            break
    print('--- create association txt ---')
    color_txt_dict = read_txt_file(color_txt)
    depth_txt_dict = read_txt_file(depth_txt)
    matches = associate_color_depth(color_txt_dict, depth_txt_dict)
    association_txt = '%s/association.txt' % dataset_folder_path
    with open(association_txt, 'a+') as a:
        for c, d in matches:
            content = '%f %s %f %s' % (c, ' '.join(color_txt_dict[c]), d, ' '.join(depth_txt_dict[d]))
            a.write(content + '\n')
