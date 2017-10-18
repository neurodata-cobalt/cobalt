import sys
import pickle
import argparse
import numpy as np
import nibabel as nib
from skimage import io
from helper import delta_epsilon
from tifffile import imsave
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource

class NeuroDataResource:
    def __init__(self, host, token, collection, experiment, chanList):
        self._collection = collection
        self._experiment = experiment
        self._bossRemote = BossRemote({'protocol':'https',
                                       'host':host,
                                       'token':token})
        self._chanList = {}
        for chanDict in chanList:
            try:
                self._chanList[chanDict['name']] = ChannelResource(chanDict['name'],
                                                                   collection,
                                                                   experiment,
                                                                   'image',
                                                                   datatype=chanDict['dtype'])
            except:
                #TODO error handle here
                raise

    def get_cutout(self, chan, zRange=None, yRange=None, xRange=None):
        if not chan in self._chanList.keys():
            print('Error: Channel Not Found in this Resource')
            return
        if zRange is None or yRange is None or xRange is None:
            print('Error: You must supply zRange, yRange, xRange kwargs in list format')
        data = self._bossRemote.get_cutout(self._chanList[chan],
                                           0,
                                           xRange,
                                           yRange,
                                           zRange)
        return data



    def save_img(self, fname, cutout, format = 'tif'):
        '''
        Saves a image as a tif or nii file
        Leave the file type suffix out of fname
        '''
        if (not fname) or (type(fname) != str):
            print('Error: Give valid filename')
        if (not cutout) or type(cutout) != np.array:
            print('Error: Give valid image cutout')

        if format == 'tif':
            max_grey_val = cutout.max(axis=1).max(axis=1).max()
            dim = cutout.shape()
            cutout_rgb = np.zeros((dim[0], dim[1], dim[2], 3))
            for i in range(dim[0]):
                for j in range(dim[1]):
                    for k in range(dim[2]):
                        normalized_rgb = int( 255 * (cutout[i][j][k]/max_grey_val) )
                        for p in range(3):
                            cutout_rgb[i][j][k][p] = normalized_rgb
            imsave(fname+'.tif', cutout_rgb.astype(np.uint8))
        else:
            nifti_img = nib.Nifti1Image(cutout, np.eye(4))
            nib.save(nifti_img, fname+'.nii')


def test_function():
    #load ground truth file
    gt = io.imread('./data/gt.tiff')

    host = 'api.boss.neurodata.io'
    token = pickle.load(open('./data/token.pkl', 'rb'))
    myResource = NeuroDataResource(host,
                                  token,
                                  'collman',
                                  'collman15v2',
                                  [{'name':'annotation', 'dtype':'uint64'},
                                   {'name':'DAPI1st', 'dtype':'uint8'},
                                   {'name':'GABA488', 'dtype':'uint8'}])

    cutout = myResource.get_cutout('annotation', [2,4], [2300,2500], [3400,3700])

    print(gt.shape == cutout.shape)
    print(delta_epsilon(np.mean(gt), np.mean(cutout), 1e-4))


if __name__ == '__main__':
    test_function()