from ndreg import *
import matplotlib
#import ndio.remote.neurodata as neurodata
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *

from NeuroDataResource import NeuroDataResource

import pickle
import numpy as np

from requests import HTTPError
import time
import configparser
startTime = time.time()


matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# Assume a valid configuration file exists at .keys/intern.cfg.
cfg_file = '.keys/intern.cfg'
if cfg_file.startswith('~'): 
    cfg_file = os.path.expanduser('~') + cfg_file[1:]
config = configparser.ConfigParser()
config.read_file(file(cfg_file))
TOKEN = config['Default']['token']
rmt = BossRemote(cfg_file_or_dict=cfg_file)

REFERENCE_COLLECTION = 'ara_2016'
REFERENCE_EXPERIMENT = 'sagittal_50um'
REFERENCE_COORDINATE_FRAME = 'ara_2016' 
REFERENCE_CHANNEL = 'average_50um'
# Set/Modify these parameters
REFERENCE_RESOLUTION = 0
REFERENCE_ISOTROPIC = True

# copied code from ndreg because for some reason it wasn't working

def setup_experiment_boss(remote, collection, experiment):
    exp_setup = ExperimentResource(experiment, collection)
    try:
        exp_actual = remote.get_project(exp_setup)
        coord_setup = CoordinateFrameResource(exp_actual.coord_frame)
        coord_actual= remote.get_project(coord_setup)
        return (exp_setup, coord_actual)
    except HTTPError as e:
        print(e.message)

def setup_channel_boss(remote, collection, experiment, channel, channel_type='image', datatype='uint16'):
    (exp_setup, coord_actual) = setup_experiment_boss(remote, collection, experiment)
 
    chan_setup = ChannelResource(channel, collection, experiment, channel_type, datatype=datatype)
    try:
        chan_actual = remote.get_project(chan_setup)
        return (exp_setup, coord_actual, chan_actual)
    except HTTPError as e:
        print(e.message)
        
        
def imgDownload_boss(remote, channel_resource, coordinate_frame_resource, resolution=0, size=[], start=[], isotropic=False):
    """
    Download image with given token from given server at given resolution.
    If channel isn't specified the first channel is downloaded.
    """
    # TODO: Fix size and start parameters

    voxel_unit = coordinate_frame_resource.voxel_unit
    voxel_units = ('nanometers', 'micrometers', 'millimeters', 'centimeters')
    factor_divide = (1e-6, 1e-3, 1, 10)
    fact_div = factor_divide[voxel_units.index(voxel_unit)]
    
    spacingBoss = [coordinate_frame_resource.x_voxel_size, coordinate_frame_resource.y_voxel_size, coordinate_frame_resource.z_voxel_size]
    spacing = [x * fact_div for x in spacingBoss] # Convert spacing to mm
    if isotropic:
	spacing = [x * 2**resolution for x in spacing]
    else:
	spacing[0] = spacing[0] * 2**resolution
	spacing[1] = spacing[1] * 2**resolution
	# z spacing unchanged since not isotropic

    if size == []: size = get_image_size_boss(coordinate_frame_resource, resolution, isotropic)
    if start == []: start = get_offset_boss(coordinate_frame_resource, resolution, isotropic)
    
    #size[2] = 200
    #dataType = metadata['channels'][channel]['datatype']
    dataType = channel_resource.datatype
    
    # Download all image data from specified channel
    array = remote.get_cutout(channel_resource, resolution, [start[0], size[0]], [start[1], size[1]], [start[2], size[2]])
    
    # Cast downloaded image to server's data type
#     img = sitk.Cast(sitk.GetImageFromArray(array),ndToSitkDataTypes[dataType]) # convert numpy array to sitk image
    img = sitk.Cast(sitk.GetImageFromArray(array),sitk.sitkUInt16) # convert numpy array to sitk image


    # Reverse axes order
    #img = sitk.PermuteAxesImageFilter().Execute(img,range(dimension-1,-1,-1))
    img.SetDirection(identityDirection)
    img.SetSpacing(spacing)

    # Convert to 2D if only one slice
    img = imgCollaspeDimension(img)

    return img

(ref_exp_resource, ref_coord_resource, ref_channel_resource) = setup_channel_boss(rmt, REFERENCE_COLLECTION, REFERENCE_EXPERIMENT, REFERENCE_CHANNEL)

refImg = imgDownload_boss(rmt, ref_channel_resource, ref_coord_resource, resolution=REFERENCE_RESOLUTION, isotropic=REFERENCE_ISOTROPIC)

refThreshold = imgPercentile(refImg, 0.99)

REFERENCE_ANNOTATION_COLLECTION = 'ara_2016'
REFERENCE_ANNOTATION_EXPERIMENT = 'sagittal_50um'
REFERENCE_ANNOTATION_COORDINATE_FRAME = 'ara_2016' 
REFERENCE_ANNOTATION_CHANNEL = 'annotation_50um'
REFERENCE_ANNOTATION_RESOLUTION = REFERENCE_RESOLUTION
REFERENCE_ANNOTATION_ISOTROPIC = True

(refAnnotation_exp_resource, refAnnotation_coord_resource, refAnnotation_channel_resource) = setup_channel_boss(rmt, REFERENCE_ANNOTATION_COLLECTION, REFERENCE_ANNOTATION_EXPERIMENT, REFERENCE_ANNOTATION_CHANNEL)
refAnnotationImg = imgDownload_boss(rmt, refAnnotation_channel_resource, refAnnotation_coord_resource, resolution=REFERENCE_ANNOTATION_RESOLUTION, isotropic=REFERENCE_ANNOTATION_ISOTROPIC)

randValues = np.random.rand(1000,3)
randValues = np.concatenate(([[0,0,0]],randValues))

randCmap = matplotlib.colors.ListedColormap(randValues)

# Remove missing parts of the brain
remove_regions = [507, 212, 220, 228, 236, 244, 151, 188, 196, 204]

refAnnoImg = sitk.GetArrayFromImage(refAnnotationImg)
remove_indices = np.isin(refAnnoImg, remove_regions)

refAnnoImg[remove_indices] = 0

# adjust annotations
refAnnoImg_adj = sitk.GetImageFromArray(refAnnoImg)
refAnnoImg_adj.SetSpacing(refAnnotationImg.GetSpacing())
refAnnotationImg = refAnnoImg_adj
# adjust atlas with corresponding indices
# refImg_adj = sitk.GetArrayFromImage(refImg)
# refImg_adj[remove_indices] = 0
# refImg_adj = sitk.GetImageFromArray(refImg_adj)
# refImg_adj.SetSpacing(refImg.GetSpacing())
# refImg = refImg_adj

# Downloading input image
# Modify these parameters for your specific experiment
SAMPLE_COLLECTION = 'ailey-dev'
SAMPLE_EXPERIMENT = 's3617'
SAMPLE_COORDINATE_FRAME = 'aileydev_s3617'
SAMPLE_CHANNEL = 'channel1'
SAMPLE_RESOLUTION = 4
SAMPLE_ISOTROPIC = False

sample_exp_resource, sample_coord_resource, sample_channel_resource = setup_channel_boss(rmt, SAMPLE_COLLECTION, SAMPLE_EXPERIMENT, SAMPLE_CHANNEL)

sampleImg = imgDownload_boss(rmt, sample_channel_resource, sample_coord_resource, resolution=SAMPLE_RESOLUTION, isotropic=SAMPLE_ISOTROPIC)

sampleThreshold = imgPercentile(sampleImg, .999)


#Reorienting input image
# modify sampleOrient based on your image orientation
sampleOrient = "RPI"
refOrient = "ASR"

sampleImgReoriented = imgReorient(sampleImg, sampleOrient, refOrient)

# Downsample images
DOWNSAMPLE_SPACING = 0.010 # millimeters
spacing = [DOWNSAMPLE_SPACING,DOWNSAMPLE_SPACING,DOWNSAMPLE_SPACING]

refImg_ds = sitk.Clamp(imgResample(refImg, spacing), upperBound=refThreshold)

sampleImg_ds = sitk.Clamp(imgResample(sampleImgReoriented, spacing), upperBound=sampleThreshold)

sampleImgSize_reorient = sampleImgReoriented.GetSize()
sampleImgSpacing_reorient= sampleImgReoriented.GetSpacing()


# Saving outputs
# sitk.WriteImage(sitk.Cast(image, sitk.sitkUInt16), 's3617_cutout.tif')
sitk.WriteImage(sitk.Cast(sampleImg_ds, sitk.sitkUInt16), 'output/sampleImg_ds.tif')
sitk.WriteImage(sitk.Cast(refImg_ds, sitk.sitkUInt16), 'output/refImg_ds.tif')
sitk.WriteImage(sitk.Cast(sampleImgReoriented, sitk.sitkUInt16), 'output/sampleImgReoriented.tif')
sitk.WriteImage(sitk.Cast(refImg, sitk.sitkUInt16), 'output/refImg.tif')



# Affine Registration
affine = imgAffineComposite(sampleImg_ds, refImg_ds, iterations=200, useMI=True, verbose=True)

sampleImg_affine = imgApplyAffine(sampleImgReoriented, affine, size=refImg.GetSize(), spacing=refImg.GetSpacing())

sampleImg_affine_bounded = sitk.Clamp(sampleImg_affine,upperBound=sampleThreshold)
refImg_bounded = sitk.Clamp(refImg, upperBound=refThreshold)


# LDDMM Registration
(field, invField) = imgMetamorphosisComposite(sampleImg_ds, refImg_ds, alphaList=[0.2, 0.1, 0.05],
                                              scaleList = 1.0, useMI=True, iterations=100, verbose=True)

affineField = affineToField(affine, field.GetSize(), field.GetSpacing())
fieldComposite = fieldApplyField(field, affineField)

invAffineField = affineToField(affineInverse(affine), invField.GetSize(), invField.GetSpacing())
invFieldComposite = fieldApplyField(invAffineField, invField)

sampleImg_lddmm = imgApplyField(sampleImgReoriented, fieldComposite, size=refImg.GetSize(), spacing=refImg.GetSpacing())

# Saving outputs after LDDMM
sitk.WriteImage(sitk.Cast(sampleImg_lddmm, sitk.sitkUInt16), 'output/sampleImg_lddmm.tif')


# Deform annotation to sample space
SAMPLE_COLLECTION = 's3617_to_ara_coll'
SAMPLE_EXPERIMENT = 's3617_to_ara_exp'
SAMPLE_CHANNEL = 's3617_to_ara_10um_affine'
NEW_CHANNEL_NAME = 'annotation_ara_s3617_test4'
SAMPLE_RESOLUTION = 0
CHANNEL_TYPE = 'annotation'
DATATYPE = 'uint64'

refAnnotationImg_lddmm = imgApplyField(refAnnotationImg, invFieldComposite, size=sampleImgReoriented.GetSize(), spacing=sampleImgReoriented.GetSpacing(), useNearest=True)

# convert the reference image to the same orientation as the input image
refAnnotationImg_lddmm = imgReorient(refAnnotationImg_lddmm, refOrient, sampleOrient)

sitk.WriteImage(sitk.Cast(refAnnotationImg_lddmm, sitk.sitkUInt16), 'output/refAnnotationImg_lddmm.tif')
