from skimage import data, feature, color, img_as_float
from scipy.ndimage import gaussian_filter
import scipy
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tifffile as tiff
import numdifftools as nd
import math
from sklearn.mixture import VBGMM, DPGMM
import warnings
import imp

import sys
sys.path.append('../../jyim6/util')

from ImageDrawer import ImageDrawer

warnings.filterwarnings('ignore')

def DoG(img, sigma1=1.0, sigma2=2.0):
  s1 = gaussian_filter(img, sigma1)
  s2 = gaussian_filter(img, sigma2)

  dog = (s1 - s2)

  return dog

def hessian(x):
  x_grad = np.gradient(x)
  hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
  for k, grad_k in enumerate(x_grad):
    tmp_grad = np.gradient(grad_k)
    for l, grad_kl in enumerate(tmp_grad):
      hessian[k, l, :, :] = grad_kl
  return hessian

def is_negative_definite(m):
  d1 = m[0, 0]
  d2 = np.linalg.det(m[:1, :1])
  d3 = np.linalg.det(m)

  return d1 > 0.0 and d2 > 0.0 and d3 < 0.0

considered_candidates = []

def get_neighbours(point):
  cx = point[0]
  cy = point[1]
  cz = point[2]

  neighbors = [[cx, cy, cz + 1],
               [cx, cy, cz - 1],
               [cx, cy + 1, cz],
               [cx, cy - 1, cz],
               [cx, cy + 1, cz + 1],
               [cx, cy + 1, cz - 1],
               [cx, cy - 1, cz + 1],
               [cx, cy - 1, cz - 1],
               [cx + 1, cy, cz],
               [cx - 1, cy, cz],
               [cx + 1, cy, cz + 1],
               [cx + 1, cy, cz - 1],
               [cx - 1, cy, cz + 1],
               [cx - 1, cy, cz - 1],
               [cx + 1, cy + 1, cz],
               [cx + 1, cy - 1, cz],
               [cx - 1, cy + 1, cz],
               [cx - 1, cy - 1, cz],
               [cx + 1, cy + 1, cz + 1],
               [cx + 1, cy + 1, cz - 1],
               [cx + 1, cy - 1, cz + 1],
               [cx + 1, cy - 1, cz - 1],
               [cx - 1, cy + 1, cz + 1],
               [cx - 1, cy + 1, cz - 1],
               [cx - 1, cy - 1, cz + 1],
               [cx - 1, cy - 1, cz - 1]]

  return [n for n in neighbors if n[0] >= 0 and n[1] >= 0 and n[2] >= 0]

def is_well_connected(candidate, blob_candidates, use_considered_candidates=False):
  global considered_candidates
  neighbors = get_neighbours(candidate)

  l = len([n for n in neighbors if n in blob_candidates and n not in considered_candidates])
  considered_candidates = considered_candidates + neighbors
  considered_candidates.append(candidate)

  return l >= 6

def get_hessian_eig(candidate_point, hes):
  neighbors = get_neighbours(candidate_point)

  hessian_sum = np.ndarray((3, 3))
  for n in neighbors:
    hessian_sum = np.add(hessian_sum, hes[:, :, n[2], n[1], n[0]])

  eigenvalues, _ = np.linalg.eig(hessian_sum)
  return [eigenvalues, hessian_sum, neighbors]

def get_intensity(gvalue):
  return (gvalue/255.0)

def compute_regional_features(candidate_point, hes, dog_img):
  [lp3, lp2, lp1], hessian_sum, neighbors = get_hessian_eig(candidate_point, hes)

  rb = abs(lp1 * lp2 * lp3)/(max(abs(lp1*lp2), abs(lp2*lp3), abs(lp1*lp3))**1.5)

  rb = (3 * (abs(np.linalg.det(hessian_sum))**(2.0/3.0)))/(lp1*lp2 + lp2*lp3 + lp3*lp1)
  st = math.sqrt(lp1**2 + lp2**2 + lp3**2)

  total_intensity = 0.0
  neighbors.append(candidate_point)

  for n in neighbors:
    total_intensity += get_intensity(dog_img[n[2], n[1], n[0]])

  at = (total_intensity/float(len(neighbors)))

  return [rb, st, at]

tiff_img = tiff.imread('input.tif')
original_image = tiff_img[:, 200:300, 200:300, :]
img = tiff_img[:, 200:300, 200:300, 0]
dog_img = DoG(img)
hes = hessian(dog_img)

tiff.imsave('slice.tif', dog_img.astype(np.uint8))

blob_candidates = []

for z in range(100):
  print('reading z-{}'.format(z))
  for y in range(100):
    for x in range(100):
      if is_negative_definite(hes[:, :, z, y, x]):
        blob_candidates.append([x, y, z])

is_well_connected(blob_candidates[0], blob_candidates)

well_connected_components = [c for c in blob_candidates if is_well_connected(c, blob_candidates, True)]

X = []

for c in well_connected_components:
  x = []
  x.extend(c)
  x.extend(compute_regional_features(c, hes, dog_img))
  X.append(x)

model = DPGMM(n_components=2, alpha=100, n_iter=100, covariance_type='spherical')
model.fit(X)
class_labels = model.predict(X)
print(class_labels)
scores = model.score(np.array(X))

avg_scores = np.zeros(len(np.unique(class_labels)))

for i, x in enumerate(X):
  avg_scores[class_labels[i]] += scores[i]

for i,avg_score in enumerate(avg_scores):
  n = len([x for x in class_labels if x == i])
  avg_scores[i] = avg_score/float(n)

blob_class_index = list(avg_scores).index(max(avg_scores))
blobs = [b for i,b in enumerate(well_connected_components) if class_labels[i] == blob_class_index]

labeled_img = dog_img

for b in blobs:
  labeled_img = ImageDrawer.draw_square(original_image, b[0], b[1], b[2], 2, [255, 0, 0])

tiff.imsave('labelled.tif', labeled_img.astype(np.uint8))
