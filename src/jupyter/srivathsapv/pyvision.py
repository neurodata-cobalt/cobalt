import numpy as np
import pylab
import mahotas as mh
cell = mh.imread('input.jpg')
pylab.imshow(cell)
pylab.show()

T = mh.thresholding.otsu(cell)
out_image = (cell > T)
out_image = out_image.astype(np.uint8) * 255

labeled, nr_objects = mh.label(out_image > T)
print nr_objects

pylab.imshow(labeled.astype(np.uint8) * 255)
pylab.jet()
pylab.show()
