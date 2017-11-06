import tifffile as tiff
import csv
import math
import numpy as np

import sys
sys.path.append('../../../util')

from ImageDrawer import ImageDrawer

def draw_square(image, coord, size=2):
    shape_z, shape_y, shape_x = image.shape
    z_range = range(max(0, coord[0]-size), min(shape_z, coord[0]+size))
    y_range = range(max(0, coord[1]-size), min(shape_y, coord[1]+size))
    x_range = range(max(0, coord[2]-size), min(shape_x, coord[2]+size))

    for z in z_range:
        for y in y_range:
            for x in x_range:
                image[z, y, x] = 255

    return image

def draw_sphere(image, coord, radius=3):
    # shape_z, shape_y, shape_x = image.shape
    # z_range = range(max(0, coord[0]-size), min(shape_z, coord[0]+size))
    # y_range = range(max(0, coord[1]-size), min(shape_y, coord[1]+size))
    # x_range = range(max(0, coord[2]-size), min(shape_x, coord[2]+size))
    #
    # for z in z_range:
    #     for y in y_range:
    #         for x in x_range:
    #             dist = np.linalg.norm(np.array(coord)-np.array([z,y,x]))
    #             image[z, y, x] = int(color/(dist+1))
    #
    # return image
    r2 = np.arange(-radius, radius+1)**2
    dist2 = r2[:, None, None] + r2[:, None] + r2
    sphere = (dist2 <= radius**2).astype(np.uint8) * 255
    s_z, s_y, s_x = [s/2 for s in sphere.shape]
    z, y, x = coord
    chunk = image[z-s_z:z+s_z, y-s_y:y+s_y, x-s_x:x+s_x]

    chunk_z, chunk_y, chunk_x = chunk.shape
    chunk_sphere = sphere[:chunk_z, :chunk_y, :chunk_x]

    image[z-s_z:z+s_z, y-s_y:y+s_y, x-s_x:x+s_x] = chunk_sphere

    return image

ename = sys.argv[1]

with open('/Users/srivathsa/projects/bloby/img/subvolumes-predicted/{}.csv'.format(ename)) as csv_file:
#with open('../../../../../bloby/img/subvolumes-predicted/cell_detection_0.csv') as csv_file:
    reader = csv.reader(csv_file)
    centroids = [[int(row[0]), int(row[1]), int(row[2])] for row in list(reader)]

#annotated_image = tiff.imread('subvolumes-normalized/cell_detection_0.tiff')[:, :, :, 0]
ref_image = tiff.imread('subvolumes/{}.tiff'.format(ename))
shape_z, shape_y, shape_x, _ = ref_image.shape
annotated_image = np.zeros((shape_z, shape_y, shape_x))

for i, c in enumerate(centroids):
    annotated_image = draw_square(annotated_image, c)

tiff.imsave('predicted-images/{}.tiff'.format(ename), annotated_image.astype(np.uint8))
