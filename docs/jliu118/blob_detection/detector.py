from skimage import data
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import disk, binary_erosion, binary_opening, label
from skimage.measure import regionprops

import numpy as np

from tifffile import imsave, imread

import pickle

def detect_blobs(input_file, verbos=False):
    img_stack = imread(input_file)
    z_comps = get_component_props(img_stack)
    blob_centroids = reconstruction_3d(z_comps)
    return blob_centroids

def get_component_props(img_stack, output_file=None, verbose=False):
    (z_dim, y_dim, x_dim) = img_stack.shape
    z_props = []

    for k in range(0, z_dim):
        z_slice = img_stack[k,:,:]
        
        # Initial thresholding: making all voxels with intensity less than half of otsu's 0.
        thresh_value = threshold_otsu(z_slice) / 2
        initial_thresholding = np.zeros(z_slice.shape)
        for i in range(0, x_dim):
            for j in range(0, y_dim):
                initial_thresholding[j, i] = 0 if z_slice[j, i] < thresh_value else z_slice[j, i]

        # Adaptive thresholding. After this step, images should be binary.
        block_size = 35
        local_thresh = threshold_local(initial_thresholding, block_size)
        otsu_thresh_value = threshold_otsu(local_thresh)
        otsu_thresh = local_thresh > otsu_thresh_value

        # Morphological tranformations using a disk shaped kernal of radius 5 pixels
        erosion = binary_erosion(otsu_thresh, disk(5))
        opening = binary_opening(erosion, disk(5))
        
        # Labelling connected components
        components, num_components = label(opening, return_num=True, connectivity=2)
        if (verbose):
            print('%d detected component(s) for z=%d ' % (num_components, z))
        props = regionprops(components)
        z_props.append(props)
        
    if (output_file != None):
        with open(output_file + '.pkl', 'wb') as f:
            pickle.dump(z_props, f)
        
    return z_props
        
def reconstruction_3d(z_comps):
    # {comp_hash_key : blob_number}
    blobs = {}

    # {blob_number : blob_num_z_slices}
    blob_z_size = {}

    blob_num = 0 # counter/id for blobs

    for k in range(2, len(z_comps) - 2):
        # print('z: %d' % k)
        below_comps = z_comps[k - 1]
        above_comps = z_comps[k + 1]
        for curr_comp in z_comps[k]:
            min_dist_below, min_comp_below = get_min_dist(curr_comp, below_comps)
            min_dist_above, min_comp_above = get_min_dist(curr_comp, above_comps)
            if (min_dist_below < 5 and min_dist_above < 5):
                # if the the blob is in the z slice below and the blob is in the z slice above
                hash_key = comp_hash(k, curr_comp)
                hash_below = comp_hash(k - 1, min_comp_below)
                if hash_below in blobs:
                    # Blob below is already in the blobs dict.
                    blob_num_below = blobs[hash_below]
                    blobs[hash_key] = blob_num_below
                    blob_z_size[blob_num_below] += 1
                else:
                    # New blob detected
                    blobs[hash_key] = blob_num
                    blob_num += 1
                    blob_z_size[blobs[hash_key]] = 1
                  
    # Getting centroids of 3d blobs.
    blob_centroids = []
    for blob_id in range(0, blob_num):
        curr_blob_centroids = []
        for key, val in blobs.iteritems():
            if (blob_id == val):
                curr_blob_centroids.append([key[0], key[1], key[2]])
        blob_centroids.append(get_3d_centroid(curr_blob_centroids))
                    
    return blob_centroids

# Helper functions for reconstruction_3d.
def comp_hash(z, comp):
#     hash_key = 17;
#     hash_key = hash_key * 31 + int(round(comp.centroid[0]));
#     hash_key = hash_key * 31 + int(round(comp.centroid[1]));
#     hash_key = hash_key * 31 + z;
    hash_key = (z, int(round(comp.centroid[0])), int(round(comp.centroid[1])))
    return hash_key

def get_min_dist(cc_comp, comps):
    min_dist = float("inf")
    min_comp = None
    for comp in comps:
        euc_dist = np.linalg.norm(np.asarray(cc_comp.centroid) - np.asarray(comp.centroid))
        if (euc_dist < min_dist):
            min_dist = euc_dist
            min_comp = comp   
            
    return min_dist, min_comp

def get_3d_centroid(points):
    zs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xs = [p[2] for p in points]
    z = int(round(float(sum(zs)) / len(points)))
    y = int(round(float(sum(ys)) / len(points)))
    x = int(round(float(sum(xs)) / len(points)))
    centroid = [z, y, z]
    return centroid
