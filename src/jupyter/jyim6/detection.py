
# coding: utf-8

# In[1]:


import sys
sys.path.append('../..')
sys.path.append('../../util')
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from collections import namedtuple

from tifffile import imsave, imread
from util.Grapher import Grapher
from util.ImageGenerator import ImageGenerator
from util.ImageDrawer import ImageDrawer
from util.helper import (
    bound_check,
    set_rgb
)
from sklearn.mixture import DPGMM, GaussianMixture
from scipy import ndimage, signal
IMG_DIR = './img/'

import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# In[2]:


def raster_3d_generator(img_shape):
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            for k in range(img_shape[2]):
                yield (i, j, k)

DOG_STACK_SIZE = 1
def DoG(img, gamma = 2, dark_blobs = 1, sigma = 2, print_level = 0):
    # if the image has white blobs then invert them to being dark blobs
    if not dark_blobs:
        img = 1 - img

    # Differential difference
    a = 0.5
    DoG_stack = []
    sigma_range = np.linspace(sigma, sigma+10, DOG_STACK_SIZE)
    for sigma in sigma_range:
        scale_constant = np.power(sigma, gamma - 1)
        # TODO: Do we need a inhouse gaussian filter to control filter size?
        G_1 = ndimage.filters.gaussian_filter(img, sigma+a)
        G_2 = ndimage.filters.gaussian_filter(img, sigma)
#         G_1 = ndimage.filters.gaussian_filter(img, 1.0)
#         G_2 = ndimage.filters.gaussian_filter(img, 2.0)
        DoG = scale_constant * (G_1 - G_2)/(a*sigma)
#         DoG = (G_1 - G_2)
        DoG_stack.append((sigma, DoG))
    return DoG_stack

def img_3d_hessian(x):
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

def find_concave_points(H):
    img_iter = raster_3d_generator(H.shape[2:])
    concave_points = set()
    for i,j,k in img_iter:
        if is_negative_definite(H[:,:,i,j,k]):
            concave_points.add((i,j,k))
    return concave_points

def is_negative_definite(m):
    d1 = m[0, 0]
    d2 = np.linalg.det(m[:1, :1])
    d3 = np.linalg.det(m)
    return d1 > 0.0 and d2 > 0.0 and d3 < 0.0

def neighboring_pixels(z, y, x):
    # returns an iterator for 26 voxels around the center
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if i != 0 or j != 0 or k != 0:
                    yield (i+z,j+y,k+x)

def get_neighbours(point):
    cz, cy, cx = point[0], point[1], point[2]
    neighbor_iter = neighboring_pixels(cz, cy, cx)
    return [(i, j, k) for i,j,k in neighbor_iter]

def find_connected_component(center, U, img):
    cz, cy, cx = center[0], center[1], center[2]
    shape_z, shape_y, shape_x = img.shape

    neighbor_iter = neighboring_pixels(cz, cy, cx)
    connected_component = []
    for i,j,k in neighbor_iter:
        if i >= 0 and j >= 0 and k >= 0 and i < shape_z and j < shape_y and k < shape_x and img[i,j,k] > 0:
            if (i,j,k) in U:
                connected_component.append((i,j,k))
    return connected_component

def draw_connected_components(img, connected_components, fname):
    draw_points = []
    for ccenter,cc in connected_components:
        for c in cc:
            draw_points.append(c)
    ImageDrawer.draw_centers(original_image, draw_points, (255, 0, 0), fname=fname)

def format_H(H):
    # If H was computed from a Z,Y,X image then the derivatives are inverted so
    # this method rectifies the mixup
    fxx = H[2, 2]
    fyy = H[1, 1]
    fzz = H[0, 0]
    fxy = H[1, 2]
    fxz = H[0, 2]
    fyz = H[0, 1]
    return np.array([
        [fxx, fxy, fxz],
        [fxy, fyy, fyz],
        [fxz, fyz, fzz]
    ])

def block_principal_minors(H):
    # This method assumes the hessian is set up canonically
    D_1 = H[0,0]
    D_2 = np.linalg.det(H[::2, ::2])
    D_3 = np.linalg.det(H)
    return D_1 + D_2 + D_3


def regional_blobness(H):
#     # This method assumes the hessian is set up canonically
#     det = np.linalg.det(H)
#     # Note these are the 2x2 principal minors
#     pm = block_principal_minors(H)
#     return 3*np.abs(det)**(2.0/3)/pm

    [lp3, lp2, lp1],_ = np.linalg.eig(H)
    return abs(lp1 * lp2 * lp3)/(max(abs(lp1*lp2), abs(lp2*lp3), abs(lp1*lp3))**1.5)


def regional_flatness(H):
    # This method assumes the hessian is set up canonically
#     tr = np.trace(H)
#     pm = block_principal_minors(H)
#     return np.sqrt(tr**2 - 2*pm)
    [lp3, lp2, lp1],_ = np.linalg.eig(H)
    return math.sqrt(lp1**2 + lp2**2 + lp3**2)

class BlobCandidate(namedtuple('BlobCandidate', ['center', 'blobness', 'flatness', 'avg_int'])):
    pass

def normalize_image(img):
    img_max = np.amax(img)
    img_min = np.amin(img)
    img = np.subtract(img, img_min)
    img = np.divide(img, img_max - img_min)
    return img

def post_prune(blob_candidates):
#     blob_candidates = [c for c in blob_candidates if not math.isnan(c[1]) and c[1] > 0.0 and c[2] > 0.0 and c[3] > 0.0]
    candidates_features = []
    candidate_coords = []
    PIXELS_PER_BLOB = 35
    for c in blob_candidates:
        data_point = []
        #data_point.extend(list(c[0]))
        data_point.extend([c[1], c[2], c[3]])
        candidates_features.append(data_point)
        candidate_coords.append(list(c[0]))

    if len(candidates_features) == 0:
        return []
    model = GaussianMixture(n_components=2, covariance_type='full')
    #model = DPGMM(n_components=2, alpha=100, n_iter=100, covariance_type='spherical')
    model.fit(candidates_features)
    class_labels = model.predict(candidates_features)
    scores = model.score_samples(np.array(candidates_features))
    #scores = model.score(np.array(candidates_features))

    avg_scores = np.zeros(len(np.unique(class_labels)))

    for i, x in enumerate(candidates_features):
        avg_scores[class_labels[i]] += scores[i]

    for i,avg_score in enumerate(avg_scores):
        n = len([x for x in class_labels if x == i])
        avg_scores[i] = avg_score/float(n)

    blob_class_index = list(avg_scores).index(max(avg_scores))
    blobs = [b for i,b in enumerate(candidate_coords) if class_labels[i] == blob_class_index]

    blobs = [(b[0], b[1], b[2]) for b in blobs]

    clusters = int(len(blobs)/PIXELS_PER_BLOB)

    max_score = 0
    max_kmeans = None

    for k in range(max(2,clusters-10), clusters+10):
      kmeans = KMeans(n_clusters=k, init='k-means++')
      cluster_labels = kmeans.fit_predict(blobs)
      s_score = silhouette_score(blobs, cluster_labels)

      if s_score > max_score:
        max_score = s_score
        max_kmeans = kmeans

    return [[math.ceil(b[0]), math.ceil(b[1]), math.ceil(b[2])] for b in max_kmeans.cluster_centers_]

# In[3]:


class BlobDetector():

    @classmethod
    def detect_3d_blobs(cls, img, original_img, scale):
        # IMPORTANT: This method assumes the image comes in Z,Y,X format
        dog_stack = DoG(img, sigma=scale)
        INTENSITY_THRESHOLD = 0.5
        print("Done computing DoG")
        for sigma, img_dog in dog_stack:
            img_hessian = img_3d_hessian(img_dog)
            concave_points = find_concave_points(img_hessian)
            print("{} concave points found".format(len(concave_points)))

            connected_components = []
            concave_points_cpy = set(concave_points)
            for c in concave_points:
                cc = find_connected_component(c, concave_points_cpy, img_dog)
                if len(cc) >= 6:
                    concave_points_cpy -= set(cc+[c])
                    connected_components.append((c, cc))
            print("{} connected components found".format(len(connected_components)))
#             draw_connected_components(original_img, connected_components, fname='output.tif')

            blob_candidates = []
            for center, cc in connected_components:
                regional_hessian = np.empty(img_hessian.shape[:2])
                average_intensity = 0
                for i,j,k in cc:
                    regional_hessian += img_hessian[:,:,i,j,k]
                    average_intensity += img[i,j,k]
                blobness = regional_blobness(regional_hessian)
                flatness = regional_flatness(regional_hessian)
                average_intensity /= len(cc)
                blob_candidates.append(
                    BlobCandidate(
                        center,
                        blobness,
                        flatness,
                        average_intensity
                    )
                )
            print("Done computing blob descriptors")

            z_max, y_max, x_max = img.shape

            blob_candidates = [b for b in blob_candidates if b[0][0] < z_max and b[0][1] < y_max and b[0][2] < x_max]
            detected_blobs = post_prune(blob_candidates)
            print("{} blobs are detected".format(len(detected_blobs)))

            for b in detected_blobs:
              original_image[b[0], b[1], b[2], :] = [255, 0, 0]

            imsave(IMG_DIR+'output.tif', original_image.astype(np.uint8))
            #ImageDrawer.draw_centers(original_image, detected_blobs, (255, 0, 0), fname='output.tif')

# # SANDBOX

# In[4]:

slice_start = 300
slice_end = 600

tiff_img = imread(IMG_DIR + 'blurred_320_randomized_gauss_cells.tif')
original_image = tiff_img[:, slice_start:slice_end, slice_start:slice_end, :]
img = tiff_img[:, slice_start:slice_end, slice_start:slice_end, 0]
imsave(IMG_DIR+'img_slice.tif', img.astype(np.uint8))
img = 1-normalize_image(img)
BlobDetector.detect_3d_blobs(img, original_image, 0.01)

# In[ ]:

tiff_img = imread(IMG_DIR + 'blurred_320_randomized_gauss_cells.tif')
original_image = tiff_img[:, slice_start:slice_end, slice_start:slice_end, :]
img = tiff_img[:, slice_start:slice_end, slice_start:slice_end, 0]
imsave(IMG_DIR+'img_slice.tif', img.astype(np.uint8))
