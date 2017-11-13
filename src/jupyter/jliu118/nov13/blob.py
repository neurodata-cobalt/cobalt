import sys
import SimpleITK as sitk
import numpy as np

from scipy.ndimage.filters import gaussian_laplace
from skimage.feature.peak import peak_local_max

from math import sqrt, hypot, log, pi

import itertools as itt

from skimage.filters import threshold_otsu, threshold_local
import csv

def detect_blobs_log(input_path, min_sigma = 1, max_sigma = 50, num_sigma = 10, overlap = 0.5, output_path='centroids.csv', print_level=False):
    # loading tiff stack
    stack_0 = sitk.ReadImage(input_path)
    stack_0 = sitk.GetArrayFromImage(stack_0)
    image_0 = stack_0 * np.float32(255.0 / stack_0.max())

    sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
                             
    image = np.copy(image_0)  

    # computing gaussian laplace
    # s**2 provides scale invariance
    log_stack = []
    for sigma in sigma_list:
        if print_level:
            print("Sigma: {}".format(sigma))
        # TODO: Do we need a inhouse gaussian filter to control filter size?
        log = -gaussian_laplace(image, sigma) * sigma ** 2
        log_stack.append(log)
    gl_scale_space = np.array(log_stack)
    
    # gl_images = [-gaussian_laplace(image, s) * s ** 2 for s in sigma_list]
    # gl_scale_space = np.array(gl_images)
                             
    gl_thresh = threshold_otsu(gl_scale_space)
                             
    local_maxima_2 = peak_local_max(gl_scale_space, threshold_abs=gl_thresh,
                                min_distance=10,
                              threshold_rel=0.0,
                              exclude_border=False)
    lm2 = local_maxima_2.astype(np.float64)
    # Convert the first index to its corresponding scale value
    lm2[:, 0] = sigma_list[local_maxima_2[:, 0]]
    local_maxima_2 = lm2
                          
    # move the scale column to be the right most column
    col_permutation = [1,2,3,0]
    local_maxima_2 = local_maxima_2[:,col_permutation]
                             
    output = _prune_blobs(local_maxima_2, overlap)
                             
    write_csv(output[:,0:3], output_path)
                             
    
def _prune_blobs(b_arr, overlap):
    # iterating again might eliminate more blobs, but one iteration suffices
    # for most cases
    blobs_array = np.copy(b_arr)
    
    for blob1, blob2 in itt.combinations(blobs_array, 2):
        if _blob_overlap(blob1, blob2) > overlap:
            if blob1[2] > blob2[2]:
                blob2[2] = -1
            else:
                blob1[2] = -1

    # return blobs_array[blobs_array[:, 2] > 0]
    return np.array([b for b in blobs_array if b[2] > 0])
                             
def _blob_overlap(blob1, blob2):
    root = sqrt(3)

    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[3] * root
    r2 = blob2[3] * root

    d = hypot_3d(blob1[0] - blob2[0], blob1[1] - blob2[1], blob1[2] - blob2[2])

    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = np.arccos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = np.arccos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * sqrt(abs(a * b * c * d))

    return area / (pi * (min(r1, r2) ** 2))
                             
def hypot_3d(x, y, z):
    return sqrt(x*x + y*y + z*z)

def write_csv(rows, fname):
    if ".csv" not in fname:
        fname = fname + ".csv"
    save_path = fname
    # save_path = "../centers/"+fname if os.path.isdir("../centers/") else fname
    # save_path = "./centers/"+fname if os.path.isdir("./centers/") else fname
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print("Saved csv as: ", fname, " at ", save_path)
                             
