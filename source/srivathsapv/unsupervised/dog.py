import sys
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter
import datetime

a = datetime.datetime.now()

img = tiff.imread(sys.argv[1])[:, :, :, 0]
dog = gaussian_filter(img, [4, 10, 1])
tiff.imsave(sys.argv[2], dog.astype(np.uint8))

b = datetime.datetime.now()
print(b-a)
