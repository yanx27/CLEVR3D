""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/model_util_scannet.py
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), os.pardir, "lib")) # HACK add the lib folder
from lib.config import CONF
from utils.box_util import get_3d_box

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def rotate_aligned_boxes(input_boxes, rot_mat):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]
    
    
    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)

def rotate_aligned_boxes_along_axis(input_boxes, rot_mat, axis):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))

    if axis == "x":     
        d1, d2 = lengths[:,1]/2.0, lengths[:,2]/2.0
    elif axis == "y":
        d1, d2 = lengths[:,0]/2.0, lengths[:,2]/2.0
    else:
        d1, d2 = lengths[:,0]/2.0, lengths[:,1]/2.0

    new_1 = np.zeros((d1.shape[0], 4))
    new_2 = np.zeros((d1.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((d1.shape[0], 3))
        crnrs[:,0] = crnr[0]*d1
        crnrs[:,1] = crnr[1]*d2
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_1[:,i] = crnrs[:,0]
        new_2[:,i] = crnrs[:,1]
    
    new_d1 = 2.0*np.max(new_1, 1)
    new_d2 = 2.0*np.max(new_2, 1)    

    if axis == "x":     
        new_lengths = np.stack((lengths[:,0], new_d1, new_d2), axis=1)
    elif axis == "y":
        new_lengths = np.stack((new_d1, lengths[:,1], new_d2), axis=1)
    else:
        new_lengths = np.stack((new_d1, new_d2, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)

class Scan3rDatasetConfig(object):
    def __init__(self, num_classes=27):
        RIO27_CLASSES = ['-', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'counter',
                         'shelf', 'curtain', 'pillow', 'clothes', 'ceiling', 'fridge', 'tv', 'towel', 'plant', 'box',
                         'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'object', 'blanket']
        NYU40_CLASSES = ['-', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                         'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
                         'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel',
                         'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp',
                         'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

        if num_classes == 27:
            classes = RIO27_CLASSES
        elif num_classes == 40:
            classes = NYU40_CLASSES
        else:
            raise NotImplementedError

        self.type2class = {k: v for v, k in enumerate(classes)}
        self.class2type = {self.type2class[t]: t for t in self.type2class}

        self.mean_size_arr = np.load(os.path.join(CONF.PATH.SCAN3R, 'meta_data/scan3r_reference_means_%dcls.npy' % num_classes))
        self.mean_color = np.load(os.path.join(CONF.PATH.SCAN3R, 'meta_data/scan3r_color_means.npy'))

        self.num_class = len(self.type2class.keys())
        self.num_heading_bin = 1
        self.num_size_cluster = len(self.type2class.keys())

        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]


    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert(False)
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        
        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        
        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return np.zeros(pred_cls.shape[0])

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''      
        return self.mean_size_arr[pred_cls] + residual

    def class2size_batch(self, pred_cls, residual):
        ''' Inverse function to size2class '''      
        return self.mean_size_arr[pred_cls] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb

    def param2obb_batch(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle_batch(heading_class, heading_residual)
        box_size = self.class2size_batch(size_class, size_residual)
        obb = np.zeros((heading_class.shape[0], 7))
        obb[:, 0:3] = center
        obb[:, 3:6] = box_size
        obb[:, 6] = heading_angle*-1
        return obb
