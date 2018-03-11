import sys, random, ndreg
import numpy as np
import SimpleITK as sitk
import skimage
from ndreg import preprocessor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib

def plot_annotations(annotations):
    img = annotations
    fig = plt.figure()
    stats = sitk.StatisticsImageFilter()
    labels_range = [1,-26]
    cmap = 'hsv'
    numSlices = 3
    size = img.GetSize()
    for i in range(img.GetDimension()):
        start = size[2 - i] / (numSlices + 1)
        sliceList = np.linspace(start, size[2 - i] - start, numSlices)
        sliceSize = list(size)
        sliceSize[2 - i] = 0

        for (j, slice) in enumerate(sliceList):
            sliceIndex = [0] * img.GetDimension()
            sliceIndex[2 - i] = int(slice)
            sliceImg = sitk.Extract(img, sliceSize, sliceIndex)
            sliceArray = sitk.GetArrayFromImage(sliceImg)
            labels = np.unique(sliceArray)[labels_range[0]:labels_range[1]]
            vmin = min(labels)
            vmax = max(labels)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

            clr_slice_array = np.zeros(tuple(list(sliceArray.shape)+[4]))
            for v in labels:
                clr = mapper.to_rgba(v)
                clr_slice_array[np.where(sliceArray == v)] = clr
            clr_slice_array[np.where(sliceArray < labels[0])] = (0.0,0.0,0.0,1.0)
            clr_slice_array[np.where(sliceArray > labels[-1])] = (0.0,0.0,0.0,1.0)

            plt.subplot(numSlices, img.GetDimension(),
                     i + img.GetDimension() * j + 1)
            ax = plt.imshow(clr_slice_array, interpolation='bilinear')
            plt.axis('off')

    plt.show()
    
def plot_mse_with_correct_aspect(img1, img2, blend=0.2, color_blend=False, side_img=None):
    height, width = img1.shape
    overlap = np.zeros((height, width)) if not color_blend else np.zeros((height, width, 3))
    if not color_blend:
        for i in range(height):
            for j in range(width):
                grey = img1[i,j]*blend + img2[i,j]*blend
                overlap[i,j] = grey
    else:
        img1 = img1 + abs(np.amin(img1))
        img2 = img2 + abs(np.amin(img2))
        img1_max = np.amax(img1)
        img2_max = np.amax(img2)
        for i in range(height):
            for j in range(width):
                overlap[i,j,:] = (img1[i,j]/img1_max, 
                                  img2[i,j]/img2_max, 
                                  img2[i,j]/img2_max)

    x_errors = []
    for j in range(width):
        error = np.sqrt(np.sum(np.square(img1[:,j] - img2[:,j])))
        x_errors.append(error)
    x_errors = np.log(np.array(x_errors))

    y_errors = []
    for i in range(height):
        error = np.sqrt(np.sum(np.square(img1[i,:] - img2[i,:])))
        y_errors.append(error)
    y_errors = np.log(np.array(y_errors))
    
    grid_height, grid_width = 2, 3 if side_img is not None else 2
    aspect= None #float(height)/width # 'auto'
    gs = gridspec.GridSpec(grid_height, grid_width,width_ratios=[float(width)/4, width, width], height_ratios=[height, height]) 
    
    ax1 = plt.subplot(gs[:(grid_height-1), 0])
    ax2 = plt.subplot(gs[:(grid_height-1), 1:(grid_width//2 + 1)])
    
    # Plot the row wise MSE errors
    ax1.semilogx(y_errors, range(height))
    
    # Axes formatting
    ax1.invert_yaxis()
    ax1.autoscale(tight=True)
    ax1.set_xlabel('Log MSE',fontsize=11)
    ax1.set_ylabel('Row pixel', fontsize=11)
    
    # Tick formatting
    ax1.locator_params(tight=True)
    ax1.tick_params(direction='in')

    # Plot overlay
    ax2.imshow(overlap, aspect=aspect)
    ax2.grid(True)
    ax2.tick_params(direction='in')
    
    # Plot side_img
    if side_img is not None:
        ax4 = plt.subplot(gs[:(grid_height-1), (grid_width//2 + 1):])
        ax4.imshow(side_img, aspect=aspect)
        ax4.grid(True)
        ax4.tick_params(direction='in')
    
def plot_mse(img1, img2, blend=0.2, color_blend=False, side_img=None):
    height, width = img1.shape
    overlap = np.zeros((height, width)) if not color_blend else np.zeros((height, width, 3))
    if not color_blend:
        for i in range(height):
            for j in range(width):
                grey = img1[i,j]*blend + img2[i,j]*blend
                overlap[i,j] = grey
    else:
        img1 = img1 + abs(np.amin(img1))
        img2 = img2 + abs(np.amin(img2))
        img1_max = np.amax(img1)
        img2_max = np.amax(img2)
        for i in range(height):
            for j in range(width):
                overlap[i,j,:] = (img1[i,j]/img1_max, img2[i,j]/img2_max, 0)

    x_errors = []
    for j in range(width):
        error = np.sqrt(np.sum(np.square(img1[:,j] - img2[:,j])))
        x_errors.append(error)
    x_errors = np.log(np.array(x_errors))

    y_errors = []
    for i in range(height):
        error = np.sqrt(np.sum(np.square(img1[i,:] - img2[i,:])))
        y_errors.append(error)
    y_errors = np.log(np.array(y_errors))
    
    gs = gridspec.GridSpec(5, 9, wspace=0.5, hspace=1) if side_img is not None else gridspec.GridSpec(5, 5, wspace=0.5, hspace=1)
    
    # Plot the MSE errors
    ax1 = plt.subplot(gs[:4, 0])
    ax2 = plt.subplot(gs[:4, 1:5])
    ax3 = plt.subplot(gs[4, 1:5])

    ax1.plot(y_errors, range(height))
    ax1.invert_yaxis()
    ax1.set_xlabel('Log MSE',fontsize=11)
    ax1.set_ylabel('Row pixel', fontsize=11)

    ax3.plot(range(width), x_errors)
    ax3.set_xlabel('Column pixel',fontsize=11)
    ax3.set_ylabel('Log MSE', fontsize=11)
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()

    ax2.imshow(overlap, aspect='auto')
    ax2.grid(True)
    
    # Plot side_img
    if side_img is not None:
        ax4 = plt.subplot(gs[:4, 5:])
        ax4.imshow(side_img,aspect='auto')
        ax4.grid(True)
    
    
def pad_img_with_range(img, depth_range=(0,0), height_range=(0,0), width_range=(0,0), return_mask=True):
    ''' User inputs the known padding parameters '''
    img_width, img_height, img_depth = img.GetSize()
    padded_img_width, padded_img_height, padded_img_depth = img_width+sum(width_range), img_height+sum(height_range), img_depth+sum(depth_range)
    padded_array = np.zeros((padded_img_depth, padded_img_height, padded_img_width))

    
    padded_array[depth_range[0]:(padded_img_depth-depth_range[1]), \
               height_range[0]:(padded_img_height-height_range[1]), \
               width_range[0]:(padded_img_width-width_range[1])] = sitk.GetArrayFromImage(img)
    padded_img = sitk.GetImageFromArray(padded_array)
    padded_img.SetSpacing(img.GetSpacing())
    
    if return_mask:
        padded_mask = np.zeros((padded_img_depth, padded_img_height, padded_img_width), dtype=np.uint16)
        padded_mask[depth_range[0]:(padded_img_depth-depth_range[1]), \
               height_range[0]:(padded_img_height-height_range[1]), \
               width_range[0]:(padded_img_width-width_range[1])] = 1
        padded_mask = sitk.GetImageFromArray(padded_mask)
        padded_mask.SetSpacing(img.GetSpacing())
        return padded_img, padded_mask
    return padded_img


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
    missing_data_mask = np.zeros((depth, height, width)) if side == 'left'  else np.ones((depth,height,width))
    for z in range(depth_lim):
        missing_data_mask[z,:,:] = missing_data_slice
    return missing_data_mask


def gen_frac_mask(depth, height, width, frac, axis, side='left'):
    ''' Generates a mask with a fraction of the image exists in either the 0,1,2 axis (depth, height, width) respectively'''
    dimensions = (depth, height, width)
    dim_lim = int(dimensions[axis]*frac)
    slice_dim = tuple([x for i,x in enumerate(dimensions) if i != axis])
    missing_data_slice = np.ones(slice_dim) if side == 'left'  else np.zeros(slice_dim)
    missing_data_mask = np.zeros((depth, height, width)) if side == 'left'  else np.ones((depth,height,width))
    for z in range(dim_lim):
        if axis == 1:
            missing_data_mask[:,z,:] = missing_data_slice
        elif axis == 2:
            missing_data_mask[:,:,z] = missing_data_slice
        else:
            missing_data_mask[z,:,:] = missing_data_slice
    return missing_data_mask



def gen_missing_anterir(depth, height, width, frac):
    height_lim = int(depth*frac)
    missing_data_slice = np.ones((height,width)) if side == 'left'  else np.zeros((height, width))
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




