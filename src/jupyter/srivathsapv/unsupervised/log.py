import sys
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_laplace

img = tiff.imread(sys.argv[1])[:, :, :, 0]
laplace = gaussian_laplace(img, [4, 10, 1])
tiff.imsave(sys.argv[2], laplace.astype(np.uint8))
