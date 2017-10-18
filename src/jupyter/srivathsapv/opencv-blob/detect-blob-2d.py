import numpy as np
import cv2
import pylab

def show_image(img):
  pylab.imshow(img)
  pylab.show()

img = cv2.imread('input.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
show_image(img)

params = params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 200
params.maxThreshold = 256

params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(255-gray)
print "Number of cells:", len(keypoints)

im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_image(im_with_keypoints)
