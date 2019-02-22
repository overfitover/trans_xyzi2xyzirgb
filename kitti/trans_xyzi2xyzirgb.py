'''
transform xyzi to xyzirgb file for training
Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
# import pcl
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti_util as utils
from kitti_object import *

import mayavi.mlab as mlab
from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d

data_path = '/home/ovo/data/data/Kitti/object/'
save_path = '/home/ovo/data/data/Kitti/object/training/velo_new_' 

def removePoints(PointCloud, BoundaryCond):
    
    # Boundary condition
    minX = BoundaryCond['minX'] ; maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY'] ; maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ'] ; maxZ = BoundaryCond['maxZ']
    
    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0]<=maxX) & (PointCloud[:, 1] >= minY) & (PointCloud[:, 1]<=maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2]<=maxZ))
    PointCloud = PointCloud[mask]
    return PointCloud

def trans_xyzi2xyzirgb(data_path, save_path, save_type='txt'):
    """
    transform xyzi to xyzirgb , you could change save_type to change save file's type
    """
    dataset = kitti_object(data_path, split='training')
    files = os.listdir(data_path+'/training/velodyne')
    files.sort(key=lambda x:int(x[:-4]))
    files_val = files.copy()
    for file in files:
        if (len(file) != 10):
            files_val.remove(file)
    
    test_num = 0
    for v in (files_val):
        file_name = v.split('.')[0]+'.' + save_type
        data_idx = int(v.split('.')[0])
        # Load data from dataset
        # objects = dataset.get_label_objects(data_idx)
        # objects[0].print_object()
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        # print(('Image shape: ', img.shape))
        #
        # Image.fromarray(img).show()
        pointCloud = dataset.get_lidar(data_idx)
        # remove pointcloud outside of bounding box
        # print("原始点云: ", pointCloud.shape)
        bc={}
        bc['minX'] = 0; bc['maxX'] = 70.4; bc['minY'] = -40; bc['maxY'] = 40
        bc['minZ'] =-2; bc['maxZ'] = 1.25
        pc_reduce = removePoints(pointCloud, bc)
        # print("移除检测范围以外的点: ", pc_reduce.shape)

        pc_velo = pc_reduce[:,0:3]
        pc_velo_intensity = 255 * np.array(pc_reduce[:,3])
        calib = dataset.get_calibration(data_idx)

        # give you some color to see see
        pc_velo_fov,fov_inds = show_lidar_with_boxes(pc_velo, calib, True, img_width, img_height)
        pc_velo_i =pc_velo_intensity[fov_inds]
        pc_velo_i = pc_velo_i.reshape(len(pc_velo_i),-1)

        # Visualize LiDAR points on images
        # print(' -------- LiDAR points projected to image plane --------')
        _,rgb_list = show_lidar_on_image(pc_velo, img, calib, img_width, img_height)

        rgb_list = np.array(rgb_list)
        # intensity = np.ones((rgb_list.shape[0],1))

        pc_velo_with_rgb = np.hstack((pc_velo_fov, rgb_list, pc_velo_i))
        np.round(pc_velo_with_rgb, decimals= 6)
        # print(pc_velo_with_rgb.shape[0])
        # print(pc_velo_with_rgb.dtype)

        if not os.path.exists(save_path+save_type):
            os.makedirs(save_path+save_type)
            
        if save_type == 'txt':
            np.savetxt(save_path+save_type+'/'+file_name, pc_velo_with_rgb)
        elif save_type == 'bin':
            pc_velo_with_rgb.tofile(save_path+save_type+'/'+file_name)
        elif save_type == 'npy':
            np.save(save_path+save_type+'/'+file_name, pc_velo_with_rgb)
        else:
            print("There are a invalid type! You should choose txt or bin or npy.")
        test_num += 1

        # if test_num == 2:
        #     break
        

if __name__=='__main__':
    
    # trans_xyzi2xyzirgb(data_path, save_path)
    trans_xyzi2xyzirgb(data_path, save_path, save_type='bin')
    
    


