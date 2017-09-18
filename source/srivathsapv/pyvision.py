import numpy as np
import pylab
import mahotas as mh
dna = mh.imread('input.jpg')
pylab.imshow(dna)
pylab.show()

T = mh.thresholding.otsu(dna)
out_image = (dna > T)
out_image = out_image.astype(np.uint8) * 255
pylab.imshow(out_image)
pylab.show()

labeled, nr_objects = mh.label(out_image > T)
print nr_objects

pylab.imshow(labeled.astype(np.uint8) * 255)
pylab.jet()
pylab.show()
