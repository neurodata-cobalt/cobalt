import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, io
from PIL import Image
import pylab
import tifffile as tiff
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import collections
import csv
import datetime

start_time = datetime.datetime.now()

def show_image(img):
  pylab.imshow(img)
  pylab.show()

with open(sys.argv[2]) as csvfile:
  rows = csv.reader(csvfile)
  x, y, z = zip(*rows)

  cell_centers = []
  for i in range(len(x)):
    cell_centers.append([int(x[i]), int(y[i]), int(z[i])])

img = tiff.imread(sys.argv[1])[:, :, :, 0]
img = img.astype('float32')
print('image read and convert')
distance = ndi.distance_transform_edt(img)
print('distance transform')
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3, 3)), labels=img)
print('local maxi')
markers = ndi.label(local_maxi)[0]

print(markers.shape)

true_positives = 0

for center in cell_centers:
  if markers[center[2], center[1], center[0]] != 0:
    true_positives += 1

false_positives = len(cell_centers) - true_positives

print(markers[14, 224, 224])
labels = watershed(-distance, markers, mask=img)

tiff.imsave(sys.argv[3], labels.astype(np.uint8))

occ = np.unique(markers, return_counts=True)
uniq_intensities = len(list(set(occ[0][1:])))
uniq_count = len(list(set(occ[1][1:])))

cell_count = (uniq_intensities/uniq_count)
accuracy = (min(cell_count, len(cell_centers)) / min(cell_count, len(cell_centers))) * 100

precision = (true_positives)/(true_positives + false_positives)
f_measure = (2*precision)/(precision + 1)

print("cell count:", (uniq_intensities/uniq_count))
print("accuracy:", accuracy)
print("f_measure:", f_measure)

end_time = datetime.datetime.now()
print(end_time - start_time)
