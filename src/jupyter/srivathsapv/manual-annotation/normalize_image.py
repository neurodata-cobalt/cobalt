import tifffile as tiff
import csv
import math
import numpy as np

import sys
sys.path.append('../../../util')

from ImageDrawer import ImageDrawer
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import opening

tif_img = tiff.imread('subvolumes/cell_detection_0.tiff').astype(float)

tiff_img = tif_img.astype(np.uint16)

tif_img *= (255.0/ tif_img.max())

shape_z, shape_y, shape_x = tif_img.shape

img = np.ndarray((shape_z, shape_y, shape_x, 3))
img[:, :, :, 0] = tif_img
img[:, :, :, 1] = tif_img
img[:, :, :, 2] = tif_img

tiff.imsave('subvolumes/cell_detection_0.tiff', img.astype(np.uint8))

for z in range(0,shape_z):
    print('{}-slice'.format(z))
    for y in range(0,shape_y):
        for x in range(0,shape_x):
            if img[z,y,x, 0] > 30:
                img[z,y,x, :] = [255, 255, 255]
            else:
                img[z,y,x, :] = [0, 0, 0]

tiff.imsave('subvolumes-normalized/cell_detection_0.tiff', img.astype(np.uint8))
