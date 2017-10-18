import sys
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter
import datetime

def hessian(x):
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

a = datetime.datetime.now()

img = tiff.imread(sys.argv[1])[:, :, :, 0]
h = hessian(img)[0, 0, :, :, :]
tiff.imsave(sys.argv[2], h.astype(np.uint8))

b = datetime.datetime.now()
print(b-a)
