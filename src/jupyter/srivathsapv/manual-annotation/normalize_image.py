import tifffile as tiff
import csv
import math
import numpy as np

import sys
sys.path.append('../../../util')

from ImageDrawer import ImageDrawer
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import opening

source_path = sys.argv[1]
target_path = sys.argv[2]

img = tiff.imread(source_path)

# tif_img = tif_img.astype(np.float64)
#
# tif_img *= (255.0/ tif_img.max())

shape_z, shape_y, shape_x, _ = img.shape

# img = np.ndarray((shape_z, shape_y, shape_x, 3))
# img[:, :, :, 0] = tif_img
# img[:, :, :, 1] = tif_img
# img[:, :, :, 2] = tif_img
#
# tiff.imsave(source_path, img.astype(np.uint8))

for z in range(0,shape_z):
    print('{}-slice'.format(z))
    for y in range(0,shape_y):
        for x in range(0,shape_x):
            if img[z,y,x, 0] > 125:
                img[z,y,x, :] = [255, 255, 255]
            else:
                img[z,y,x, :] = [0, 0, 0]

tiff.imsave(target_path, img.astype(np.uint8))
