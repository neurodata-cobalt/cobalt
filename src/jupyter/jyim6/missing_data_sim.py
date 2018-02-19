import sys, random, ndreg
import numpy as np
import SimpleITK as sitk
import skimage
from ndreg import preprocessor


def random_point(max_height, max_width):
    return random.randint(0, max_height), random.randint(0,max_width)

def random_hyperplane(height, width):
    y_1,x_1 = random_point(height,width)
    y_2,x_2 = random_point(height,width)
    m = (y_2-y_1)/(x_2-x_1)
    b = y_2 - m*x_2
    side_of_the_force = 'light' if random.randint(0,1) == 1 else 'dark'
    missing_data_mask = np.ones((height,width)) if side_of_the_force == 'light' else np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if i - m*j <= b:
                if side_of_the_force == 'light':
                    missing_data_mask[i,j] = 0
                else:
                    missing_data_mask[i,j] = 1
            else:
                break
    return missing_data_mask

def sync_metadata(img, target, dtype='uint16'):
    if type(img) == np.ndarray:
        img = img.astype(dtype)
        img = sitk.GetImageFromArray(img)
    img.SetSpacing(target.GetSpacing())
    #return img

def gen_random_halfspace(depth, height, width):
    hyperplane_2d = random_hyperplane(height, width)
    halfspace = np.zeros((depth,height,width))
    for z in range(depth):
        halfspace[z,:,:] = hyperplane_2d
    return halfspace

def gen_hemisphere_mask(depth, height, width, side='left'):
    depth_lim = depth//2
    missing_data_slice = np.ones((height,width)) if side == 'left'  else np.zeros((height, width))
    # for i in range(height_lim):
    #     for j in range(width_lim):
            # if side_of_the_force == 'light':
            #     missing_data_slice[i,j] = 0
            # else:
            #     missing_data_slice[i,j] = 1
    missing_data_mask = np.zeros((depth, height, width)) if side == 'left'  else np.ones((depth,height,width))
    for z in range(depth_lim):
        missing_data_mask[z,:,:] = missing_data_slice
    return missing_data_mask

def gen_label_mask(img, full_mask, labels):
    missing_data_array = np.copy(sitk.GetArrayFromImage(full_mask))
    img_array = sitk.GetArrayFromImage(img)
    for layer in labels:
        missing_data_array[np.where(img_array == layer)] = 0
    return missing_data_array

def convert_to_image(array, sync_img=None):
    img = sitk.GetImageFromArray(array)
    if sync_img:
        img.CopyInformation(sync_img)
    return img


def mask_img(source, mask):
    if source.GetSize() != mask.GetSize():
        raise Exception("Incompatible sizes")
    elif source.GetPixelIDTypeAsString() != mask.GetPixelIDTypeAsString():
        raise Exception("Incompatible dtype")
    elif source.GetSpacing() != mask.GetSpacing():
        raise Exception("Incompatible spacing")
    return sitk.Mask(source, mask)




