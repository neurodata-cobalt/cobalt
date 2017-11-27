import tifffile as tiff
import csv
import numpy as np

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

with open('cell_detection_3_predicted.csv') as csv_file:
    reader = csv.reader(csv_file)
    centroids = [[int(row[0]), int(row[1]), int(row[2])] for row in list(reader)]

ref_image = tiff.imread('cell_detection_3.tiff')
shape_z, shape_y, shape_x, _ = ref_image.shape

annotated_image = np.ndarray((shape_z, shape_y, shape_x))

for i, c in enumerate(centroids):
    annotated_image = draw_square(annotated_image, c)

tiff.imsave('cell_detection_3_predicted_empty.tiff', annotated_image.astype(np.uint8))
