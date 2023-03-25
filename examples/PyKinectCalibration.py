"""
该程序基于PyKinect2生成相机标定所用图片

使用'Enter'键拍摄照片
使用'Esc'键结束
"""

from pykinect2 import PyKinectRuntime, PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime
from pykinect2.PyKinectV2 import *
from PyKinectDataset import get_last_multi_frame_infor
import cv2 as cv
import os

if __name__ == '__main__':
    kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                             PyKinectV2.FrameSourceTypes_Depth |
                             PyKinectV2.FrameSourceTypes_Infrared)
    folder_path = '../calibration'
    color_path = '../calibration/rgb'
    infrared_path = '../calibration/infrared'
    count = 0
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(color_path)
        os.makedirs(infrared_path)
    while True:
        infor = get_last_multi_frame_infor(kinect)
        color_data = infor[0]
        infrared_data = infor[4]
        if color_data is not None and infrared_data is not None:
            cv.namedWindow('color', cv.WINDOW_AUTOSIZE)
            cv.imshow('color', color_data)
            cv.namedWindow('infrared', cv.WINDOW_AUTOSIZE)
            cv.imshow('infrared', infrared_data)
            if cv.waitKey(1) == 13:
                count += 1
                color_file = '%s/%d.png' % (color_path, count)
                cv.imwrite(color_file, color_data)
                infrared_file = '%s/%d.png' % (infrared_path, count)
                cv.imwrite(infrared_file, infrared_data)
        if cv.waitKey(1) == 27:
            break
