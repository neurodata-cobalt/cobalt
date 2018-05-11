from scipy.ndimage.filters import laplace
from bossHandler import bossHandler
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
from sklearn.cluster import KMeans

class vertices:
    def __init__(self, mask , vol):
        '''
        Constructor
        '''
        self.mask = np.copy(mask) # binarized volume
        self.masked_vol = np.copy(vol)
        self.masked_vol[self.mask == 0] = 0
        
        

    def compute_w1(self):
        '''
        Compute W1 for vertex weight computation
        '''
        min_intensity = np.float64(np.min(self.masked_vol[self.masked_vol !=0 ]))
        max_intensity = np.max(self.masked_vol)
        w1 = np.copy(self.masked_vol)
        w1 = (w1 - min_intensity)/(max_intensity - min_intensity)
        w1[w1 < 0] = 0

        return w1
    
    def compute_w2(self):
        '''
        Compute W2 for vertex weight computation; Distance from boundaries.
        '''
        w2 = np.copy(self.masked_vol)
        w2 = skfmm.distance(w2)
        min_distance = np.float64(np.min(w2[w2 != 0]))
        max_distance = np.max(w2)
        w2 = (w2 - min_distance)/(max_distance - min_distance)
        w2[w2 < 0] = 0
        
        return w2
    
    def compute_w3(self):
        '''
        Compute W3 for vertex weight computation; Laplacian of edges
        '''
        w2 = self.compute_w2()
        lplc = laplace(w2)
        lplc = np.tanh(2.0*lplc)
        w3 = np.minimum(np.ones(lplc.shape) , 1.0 - lplc)
        w3[self.mask == 0] = 0
        return w3
        
    def compute_vertex_wight(self):
        '''
        Compute vetex weight (average of w1, w2, and w3)
        '''
        w1 = self.compute_w1()
        w2 = self.compute_w2()
        w3 = self.compute_w3()
        
        vw = (w1 + w2 + w3)/3.0
#         vw[self.mask == 0] = 0
        
        return vw


class tractoHandler:
    def __init__(self, data_cutout_raw):
        '''
        Constructor
        '''
        self.data_cutout_raw = data_cutout_raw
        
    def run_tractography(self, methodn):

        # Binarize
        if methodn == 3:
            print('slice-by-slice with subsampling and percentile')
            # with percentile
            gmm_nc = 4
            sub_sample_to = 1000
            data_cutout_binarized = np.copy(self.data_cutout_raw)
            vol_size = self.data_cutout_raw.shape
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
            data_cutout_binarized = np.copy(self.data_cutout_raw)
            vol_size = self.data_cutout_raw.shape
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
            data_cutout_binarized = np.copy(self.data_cutout_raw)
            vol_size = self.data_cutout_raw.shape
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
            data_cutout_binarized = np.copy(self.data_cutout_raw)
            vol_size = self.data_cutout_raw.shape
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
        ttt = vertices(data_cutout_binarized , self.data_cutout_raw)
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

        return skeleton, concomp, concomp_col, data_cutout_binarized
    
    def find_closest_voxel(self, voxel, vol_idx):
        '''
        vol_idx: indices of nonzero elements; shape: (nx3)
        Finds the closest non-zero voxel in vol_idx to point x,y,z
        '''
        subtrc = vol_idx - [voxel[0], voxel[1], voxel[2]]
        minIdx = np.argmin(np.linalg.norm(subtrc, axis = 1))

        return vol_idx[minIdx,:]

    def quantify(self, nz_data_1, nz_data_2):
        '''
        Method of quantification: for each voxel in nz_data_1, find the closest voxel in the nz_data_2
        '''
        # find indices of non-zero elements in test data:
        closest_voxels = np.empty(nz_data_1.shape)
        for idx, voxel in enumerate(nz_data_1):
            closest_voxel = self.find_closest_voxel(voxel , nz_data_2)
            closest_voxels[idx,:] = closest_voxel

        return closest_voxels    
        
     
    



           
