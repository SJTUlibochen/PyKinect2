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

from pykinect2 import PyKinectRuntime, PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime
from pykinect2.PyKinectV2 import *
import numpy as np
import cv2 as cv
import time
import os


def get_last_multi_frame_infor(camera):
    color_frame_data = None
    color_frame_time = None
    depth_frame_data = None
    depth_frame_time = None
    infrared_frame_data = None
    infrared_frame_time = None
    if camera.first_time:
        start_time = time.time()
        while True:
            now_time = time.time()
            used_time = now_time - start_time
            color_infor = camera.get_last_color_frame_infor()
            depth_infor = camera.get_last_depth_frame_infor()
            infrared_infor = camera.get_last_infrared_frame_infor()
            if color_infor[1] is not None \
                    and depth_infor[1] is not None \
                    and infrared_infor[1] is not None:
                camera.first_time = False
                break
            elif used_time > 5:
                raise RuntimeError('连接超时，请使用Kinect Configuration Verifier检查连接')
    else:
        color_frame_data, color_frame_time = camera.get_last_color_frame_infor()
        depth_frame_data, depth_frame_time = camera.get_last_depth_frame_infor()
        infrared_frame_data, infrared_frame_time = camera.get_last_infrared_frame_infor()
    camera_infor = [color_frame_data, color_frame_time, depth_frame_data,
                    depth_frame_time, infrared_frame_data, infrared_frame_time]
    return camera_infor


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
                          offset: float = 0, max_difference: float = 0.1):
    color_time_list = list(col_txt_dict)
    depth_time_list = list(dep_txt_dict)
    potential_match = [(abs(col - (dep + offset)), col, dep)
                       for col in color_time_list
                       for dep in depth_time_list
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
    kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                             PyKinectV2.FrameSourceTypes_Depth |
                             PyKinectV2.FrameSourceTypes_Infrared)
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
        print('new dataset folder ' + dataset_folder_path + ' created!')
    print('--- show rgb & depth image ---')
    print('press Esc to exit')
    while True:
        infor = get_last_multi_frame_infor(kinect)
        ori_color_image = infor[0]
        ori_depth_image = infor[2]
        color_time = infor[1]
        depth_time = infor[3]
        align_color_x, align_color_y = kinect.map_depth_to_color()
        if ori_color_image is not None and ori_depth_image is not None:
            align_color_image = ori_color_image[align_color_y, align_color_x, 0:3]
            enhanced_depth_image = ori_depth_image * (2 ** 3)

            ori_color_image_flip = cv.flip(ori_color_image, 1)
            cv.namedWindow('ori_color', cv.WINDOW_AUTOSIZE)
            cv.imshow('ori_color', ori_color_image_flip)

            align_color_image_flip = cv.flip(align_color_image, 1)
            cv.namedWindow('align_color', cv.WINDOW_AUTOSIZE)
            cv.imshow('align_color', align_color_image_flip)
            color_file = '%s/%f.png' % (color_path, color_time)
            cv.imwrite(color_file, align_color_image_flip)
            color_txt_content = '%f rgb/%f.png' % (color_time, color_time)
            with open(color_txt, 'a+') as c:
                c.write(color_txt_content + '\n')

            enhanced_depth_image_flip = cv.flip(enhanced_depth_image, 1)
            cv.namedWindow('ehd_depth', cv.WINDOW_AUTOSIZE)
            cv.imshow('ehd_depth', enhanced_depth_image_flip)

            ori_depth_image_flip = cv.flip(ori_depth_image, 1)
            cv.namedWindow('depth', cv.WINDOW_AUTOSIZE)
            cv.imshow('depth', ori_depth_image_flip)
            depth_file = '%s/%f.png' % (depth_path, depth_time)
            cv.imwrite(depth_file, ori_depth_image_flip)
            depth_txt_content = '%f depth/%f.png' % (depth_time, depth_time)
            with open(depth_txt, 'a+') as d:
                d.write(depth_txt_content + '\n')
        if cv.waitKey(1) == 27:
            cv.destroyAllWindows()
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
    print('association.txt created!')
    print('--- --- ---')
    print('fig pack created!')
