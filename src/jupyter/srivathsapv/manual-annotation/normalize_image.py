import tifffile as tiff
import csv
import math
import numpy as np

import sys
sys.path.append('../../../util')

from ImageDrawer import ImageDrawer
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import opening

img = tiff.imread('s3617_cutout.tif')
tif_img *= (255.0/ tif_img.max())

img = np.ndarray((100,1000,1000,3))
img[:, :, :, 0] = tif_img
img[:, :, :, 1] = tif_img
img[:, :, :, 2] = tif_img

with open('centroids.csv') as csv_file:
    reader = csv.reader(csv_file)
    centroids = list(reader)

subvolume_img = img[:, 300:400, 650:750, :]

for z in range(0,100):
    print('{}-slice'.format(z))
    for y in range(0,1000):
        for x in range(0,1000):
            if img[z,y,x, 0] > 30:
                img[z,y,x, :] = [255, 255, 255]
            else:
                img[z,y,x, :] = [0, 0, 0]

tiff.imsave('s3617_cutout_normalized.tif', img.astype(np.uint8))

centroids = [[math.ceil(float(c[0].split('.')[1])), math.ceil(float(c[1].split('.')[1])), math.ceil(float(c[2])] for c in centroids]

formatted_centroids = []

for c in centroids:
    cz = int(c[0])-1
    cy = int(math.ceil(float(c[1]) * (1000/39.37)))
    cx = int(math.ceil(float(c[2]) * (1000/39.37)))
    formatted_centroids.append([cz, cy, cx])

csv_file = open('centroids_formatted.csv', 'w')

for c in formatted_centroids:
    csv_file.write(','.join(map(str,c)) + '\n')

csv_file.close()
