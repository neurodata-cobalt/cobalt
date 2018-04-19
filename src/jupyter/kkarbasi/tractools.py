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




def run_tractography(data_cutout_raw):
	# Binarize
	gmm_nc = 4
	data_cutout_binarized = np.copy(data_cutout_raw)
	vol_size = data_cutout_raw.shape
	for i in np.arange(0 , vol_size[2]):
	    data_slice = data_cutout_binarized[:,:,i]
	    uniq = np.unique(data_slice , return_counts=True)
	    
	    gmm = GaussianMixture(gmm_nc, covariance_type = 'diag').fit(data_slice.reshape(-1,1))
	    cluster_labels = gmm.predict(data_slice.reshape(-1,1))
	    cluster_labels = cluster_labels.reshape(data_slice.shape)
	    x = np.arange(0,uniq[1].shape[0])
	    c_id = np.argmax(gmm.means_) # index of the cluster with highest mean
	    
	    data_slice[cluster_labels == c_id] = 1
	    data_slice[cluster_labels != c_id] = 0
	    data_cutout_binarized[:,:,i] = data_slice

	#binary openning
	data_cutout_binarized = binary_opening(data_cutout_binarized, np.ones((3,3,3), dtype='uint16'))

	#Extract seyoun weights
	ttt = vertices(data_cutout_binarized , data_cutout_raw)
	vw = ttt.compute_vertex_wight()
	
	# skeletonize and connected components
	skeleton = skeletonize_3d(vw)
#	concomp = label(np.copy(skeleton) , connectivity=3)
	# skeleton = binary_closing(skeleton, np.ones((5,5,5), dtype='uint8'))
	# skeleton = binary_opening(skeleton, np.ones((3,3,3), dtype='uint8'))
#	cmap = plt.cm.get_cmap('nipy_spectral' , np.unique(concomp).size)
#
#	concomp_col = np.empty(concomp.shape + (3,), dtype = 'uint8')
#	for col in np.arange(np.unique(concomp).size):
#	    tmp = cmap(col)[0:-1]
#	    tmp = tuple(i*255 for i in tmp)
#	    concomp_col[concomp == col] = tmp
	return skeleton	
		

