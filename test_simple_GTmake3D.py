from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader

#from layers import disp_to_depth
#from utils import readlines
#from options import MonodepthOptions
#import datasets
#import networks


import sys
import glob
import argparse
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

#from torchvision import transforms, datasets
#from kitti_utils import *

#from utils import download_model_if_doesnt_exist
from scipy.interpolate import LinearNDInterpolator

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


#main_path = os.path.join(os.path.dirname(__file__), "make3d")
main_path = '/data3/zhuy/Make3D'

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti

    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
    test_filenames = f.read().splitlines()
test_filenames = list(map(lambda x: x[4:-4], test_filenames))


depths_gt = []
images = []


ratio = 2
h_ratio = 1 / (1.33333 * ratio)
color_new_height = 1704 / 2
depth_new_height = 21

i = 0

# for filename in test_filenames:
#     mat = scipy.io.loadmat(os.path.join(main_path,"test_GT_depthmap", "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)),verify_compressed_data_integrity=False)
#     depths_gt.append(mat["Position3DGrid"][:,:,3])
#     # image = cv2.imread(os.path.join(main_path,"test_image", "Test134", "img-{}.jpg".format(filename)))
#     # image = image[int((2272 - color_new_height) / 2):int((2272 + color_new_height) / 2), :]
#     # images.append(image[:, :, ::-1])
#     # cv2.imwrite(os.path.join(main_path, "Test134_cropped", "img-{}.jpg".format(filename)), image)
#
# #depths_gt_resized = list(map(lambda x: cv2.resize(x, (1704, 852), interpolation=cv2.INTER_NEAREST), depths_gt))
# depths_gt_resized = list(map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt))
# #depths_gt_cropped = map(lambda x: x[(55 - 21)/2:(55 + 21)/2], depths_gt)
# depths_gt_cropped = list(map(lambda x: x[17:38], depths_gt))
#
# for filename in test_filenames:
#     #gt_depth = depths_gt_new[i]
#     gt_depth = depths_gt_cropped[i]
#     depth = (gt_depth * 256).astype(np.uint16)
#     save_file = os.path.join(main_path, "Depth", "{}.png".format(filename))
#     cv2.imwrite(save_file, depth)
#     i = i+1
#     print("  Processed"," {} of 134  images done ".format(int(i)))




for filename in test_filenames:
    mat = scipy.io.loadmat(os.path.join(main_path,"test_GT_depthmap", "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)),verify_compressed_data_integrity=False)
    depths_gt.append(mat["Position3DGrid"][:,:,3])

depths_gt_resized = list(map(lambda x: cv2.resize(x, (1704, 852), interpolation=cv2.INTER_NEAREST), depths_gt))

#depths_gt_cropped = map(lambda x: x[(55 - 21)/2:(55 + 21)/2], depths_gt)
depths_gt_cropped = list(map(lambda x: x[17:38], depths_gt))
#depths_gt_new = depths_gt_cropped
#depths_gt_new = list(map(lambda x: cv2.resize(x, (512, 256), interpolation=cv2.INTER_NEAREST), depths_gt_cropped))
#depths_gt_new = list(map(lambda x: cv2.resize(x, (512, 256), interpolation=cv2.INTER_LINEAR), depths_gt_cropped))
#depths_gt_new = list(map(lambda x: cv2.resize(x, (512, 256), interpolation=cv2.INTER_LANCZOS4), depths_gt_cropped))
#depths_gt_new = list(map(lambda x: cv2.resize(x, (512, 256), interpolation=cv2.INTER_CUBIC), depths_gt_cropped))
depths_gt_new = list(map(lambda x: cv2.resize(x, (512, 256), interpolation=cv2.INTER_AREA), depths_gt_cropped))
for filename in test_filenames:


    gt_depth = depths_gt_new[i]
    gt_height, gt_width = gt_depth.shape[:3]
    # print("gt_height {:d}".format(gt_height))
    # print("gt_width {:d}".format(gt_width))
    #gt_depth = 1 /(gt_depth + 1e-3)

    #  Get location (xy) for valid pixeles
    x, y = np.where(gt_depth > 0)
    #x, y = np.where(gt_depth != 0)
    #  Get depth values for valid pixeles
    #m_d = gt_depth[gt_depth != 0]
    m_d = gt_depth[gt_depth > 0]
    d = 1/m_d


    #  Generate an array Nx3
    xyd = np.stack((y,x,d)).T

    gt = lin_interp(gt_depth.shape, xyd)


    disp_resized_np = gt
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)



    name_dest_im = os.path.join(main_path, "GTimage", "{}_dispGT.jpeg".format(filename))
    im.save(name_dest_im)
    i = i+1

    print("  Processed"," {} of 134  images done ".format(int(i)))


