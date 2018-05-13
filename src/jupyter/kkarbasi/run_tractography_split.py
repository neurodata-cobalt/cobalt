from tractools import run_tractography, get_filename
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
from cobalt_tractography.bossHandler import *
from cobalt_tractography.tractography import *

# Parameters:
coll_name = 'ailey-dev'
exp_name = 'Insula_Atenolol-1_171204_new'
chan_name = 'Ch0'

bHandler = bossHandler(coll_name)
bHandler.select_experiment(exp_name)
bHandler.select_channel(chan_name)
coor = bHandler.get_coordinate_frame()

x_range = coor.x_stop
y_range = coor.y_stop
z_range = coor.z_stop

x_slices = range(x_range)
x_slices = np.array(x_slices[0::500] + [x_range + 1])
x_slices = zip(x_slices , np.roll(x_slices-1,-1))[0:-1]


y_slices = range(y_range)
y_slices = np.array(y_slices[0::500] + [y_range + 1])
y_slices = zip(y_slices , np.roll(y_slices-1,-1))[0:-1]


z_slices = range(z_range)
z_slices = np.array(z_slices[0::100] + [z_range + 1])
z_slices = zip(z_slices , np.roll(z_slices-1,-1))[0:-1]


for xx in x_slices:
    for yy in y_slices:
        for zz in z_slices:
            data_cutout_raw = bHandler.get_cutout(list(xx) , list(yy), list(zz))
            data_cutout_raw = np.transpose(img_as_uint(data_cutout_raw),(1,2,0))
            th = tractoHandler(data_cutout_raw)
            skeleton, concomp, concomp_col, data_cutout_binarized =th.run_tractography(1)
#             skeleton = run_tractography(data_cutout_raw, methodn=3)
            filename = get_filename(xx,yy,zz)
            #save
            imsave('/run/mount/DRN-BLA_2378_2p_glycerol_Ch0/output_Insula_Atenolol-1_171204_new/' + filename , skeleton)
            #upload_to_boss
#             chan_resource = rmt.get_channel(chan_name= new_chan_name, coll_name=new_coll_name, exp_name=new_exp_name)
#             rmt.create_cutout(channel_resource, 0, [xx[0], xx[1]], [yy[0], yy[1]], [zz[0], zz[1]], skeleton)
            print('done: ' + filename)

