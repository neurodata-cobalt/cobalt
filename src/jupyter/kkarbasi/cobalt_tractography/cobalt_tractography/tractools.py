# tractography tools

from bossHandler import bossHandler
from tractography import vertices
from intern.resource.boss.resource import *
from intern.remote.boss import BossRemote
from skimage import filters
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.mlab as mlab
import glob
import skfmm
from scipy.ndimage.filters import laplace
from skimage.morphology import binary_opening, binary_closing, binary_dilation
from skimage.morphology import skeletonize_3d,label
from scipy.ndimage.morphology import *
from tifffile import imsave
from skimage import img_as_ubyte, img_as_uint, color




def run_tractography(data_cutout_raw , methodn):
    
    # Binarize
    if methodn == 3:
        print('slice-by-slice with subsampling and percentile')
        # with percentile
        gmm_nc = 4
        sub_sample_to = 1000
        data_cutout_binarized = np.copy(data_cutout_raw)
        vol_size = data_cutout_raw.shape
        for i in np.arange(0 , vol_size[2]):
            data_slice = data_cutout_binarized[:,:,i]
        #     uniq = np.unique(data_slice , return_counts=True)

            data_slice_shuffled = data_slice.flatten()
            prcntile = np.percentile(data_slice_shuffled,80)
            data_slice_shuffled = data_slice_shuffled[data_slice_shuffled >= prcntile]


            np.random.shuffle(data_slice_shuffled)
            gmm = GaussianMixture(gmm_nc, covariance_type = 'spherical').fit(data_slice_shuffled[0:sub_sample_to].reshape(-1,1))



        #     gmm = GaussianMixture(gmm_nc, covariance_type = 'diag').fit(data_slice.reshape(-1,1))
            cluster_labels = gmm.predict(data_slice.reshape(-1,1))
            cluster_labels = cluster_labels.reshape(data_slice.shape)
        #     x = np.arange(0,uniq[1].shape[0])
            c_id = np.argmax(gmm.means_) # index of the cluster with highest mean

            data_slice[cluster_labels == c_id] = 1
            data_slice[cluster_labels != c_id] = 0
            data_cutout_binarized[:,:,i] = data_slice
    if methodn == 1:
        print('slice-by-slice with subsampling')
        gmm_nc = 4 
        data_cutout_binarized = np.copy(data_cutout_raw)
        vol_size = data_cutout_raw.shape
        for i in np.arange(0 , vol_size[2]):
            data_slice = data_cutout_binarized[:,:,i]
            data_slice_shuffled = data_slice.flatten()
            np.random.shuffle(data_slice_shuffled)


            gmm = GaussianMixture(gmm_nc, covariance_type = 'spherical').fit(data_slice_shuffled[0:10000].reshape(-1,1))
            cluster_labels = gmm.predict(data_slice.reshape(-1,1))
            cluster_labels = cluster_labels.reshape(data_slice.shape)

            c_id = np.argmax(gmm.means_) # index of the cluster with highest mean

            data_slice[cluster_labels == c_id] = 1
            data_slice[cluster_labels != c_id] = 0
            data_cutout_binarized[:,:,i] = data_slice
    if methodn == 0:
        print('slice-by-slice without subsampling')
        # slice-by-slice without subsampling 
        gmm_nc = 4
        data_cutout_binarized = np.copy(data_cutout_raw)
        vol_size = data_cutout_raw.shape
        for i in np.arange(0 , vol_size[2]):
            data_slice = data_cutout_binarized[:,:,i]
            uniq = np.unique(data_slice , return_counts=True)

            gmm = GaussianMixture(gmm_nc, covariance_type = 'full').fit(data_slice.reshape(-1,1))
            cluster_labels = gmm.predict(data_slice.reshape(-1,1))
            cluster_labels = cluster_labels.reshape(data_slice.shape)
            x = np.arange(0,uniq[1].shape[0])
            c_id = np.argmax(gmm.means_) # index of the cluster with highest mean

            data_slice[cluster_labels == c_id] = 1
            data_slice[cluster_labels != c_id] = 0
            data_cutout_binarized[:,:,i] = data_slice
    if methodn == 2:
        print('sub-vol by sub-vol with subsampling')
        # sub-vol by sub-vol with subsampling 
        gmm_nc = 3
        slices_per_vol = 5
        data_cutout_binarized = np.copy(data_cutout_raw)
        vol_size = data_cutout_raw.shape
        for i in np.arange(0, vol_size[2], slices_per_vol):

            data_slice = data_cutout_binarized[:, :, i : i+slices_per_vol]

            data_slice_shuffled = data_slice.flatten()
            np.random.shuffle(data_slice_shuffled)
            gmm = GaussianMixture(gmm_nc, covariance_type = 'diag').fit(data_slice_shuffled[0:1000].reshape(-1,1))



        
            cluster_labels = gmm.predict(data_slice.reshape(-1,1))
            cluster_labels = cluster_labels.reshape(data_slice.shape)
        
            c_id = np.argmax(gmm.means_) # index of the cluster with highest mean

            data_slice[cluster_labels == c_id] = 1
            data_slice[cluster_labels != c_id] = 0
            data_cutout_binarized[:,:,i : i+slices_per_vol] = data_slice
	#binary openning
    data_cutout_binarized = binary_opening(data_cutout_binarized, np.ones((3,3,3), dtype='uint16'))
    ttt = vertices(data_cutout_binarized , data_cutout_raw)
    vw = ttt.compute_vertex_wight()
    skeleton = skeletonize_3d(vw)
    concomp = label(np.copy(skeleton) , connectivity=3)
    # skeleton = binary_closing(skeleton, np.ones((5,5,5), dtype='uint8'))
    # skeleton = binary_opening(skeleton, np.ones((3,3,3), dtype='uint8'))
    cmap = plt.cm.get_cmap('nipy_spectral' , np.unique(concomp).size)

    concomp_col = np.empty(concomp.shape + (3,), dtype = 'uint8')
    for col in np.arange(np.unique(concomp).size):
        tmp = cmap(col)[0:-1]
        tmp = tuple(i*255 for i in tmp)
        concomp_col[concomp == col] = tmp
    #     print(skeleton.dtype)
        
    
    return skeleton, concomp, concomp_col




def get_filename(xx, yy, zz):
    return 'x-'+str(xx[0])+'-'+str(xx[1])+'_y-'+str(yy[0])+'-'+str(yy[1])+'_z-'+str(zz[0])+'-'+str(zz[1])+'.tiff'
