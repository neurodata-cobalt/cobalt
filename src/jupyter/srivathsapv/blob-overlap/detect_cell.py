import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import color, io
from PIL import Image

def show_image(img):
  imgX = Image.fromarray(img)
  imgX.show()

img = cv2.imread('input.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv2.dilate(opening,kernel,iterations=3)

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 200
params.maxThreshold = 256

params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(255-sure_fg)
im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_image(im_with_keypoints)
