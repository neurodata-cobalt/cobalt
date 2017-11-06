import numpy as np
import csv
import argparse
import sys
import os
from collections import namedtuple
sys.path.append('./../../util/')
sys.path.append('./../srivathsapv/blob-metrics/')
from BlobMetrics import BlobMetrics
from tifffile import imsave, imread
from ImageDrawer import ImageDrawer

EUC_DIST_THRES = 12
OUTPUT_DIR = './farsight_output/'

parser = argparse.ArgumentParser()
parser.add_argument('farsight_centers', type=str)
parser.add_argument('annotated_centers', type=str)
parser.add_argument('results_csv', type=str)
parser.add_argument('cell_fname', type=str)
parser.add_argument('orig_tif', type=str)
args = parser.parse_args()

f_centers_fname, a_centers_fname = args.farsight_centers, args.annotated_centers
results_csv = args.results_csv
cell_fname, orig_fname = args.cell_fname, args.orig_tif
subvolume = f_centers_fname.split("/")[-1][:-12]    # Extract the subvolume name

# Read in the annotated centers
a_centers = []
with open(a_centers_fname, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        a_centers.append(
            (int(float(row[1])), int(float(row[2])), int(float(row[3])))
        )


# Read in the FARSIGHT centers
f_centers = []
with open(f_centers_fname, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        f_centers.append(
            tuple([int(x) for x in row])
        )
# import pdb; pdb.set_trace()

print("{} annotated centers, {} detected centers".format(len(a_centers), len(f_centers)))

# Compute blob metrics
metrics = BlobMetrics(a_centers, f_centers, EUC_DIST_THRES)
accuracy = metrics.accuracy()
precision = metrics.precision()
recall = metrics.recall()
mse = metrics.mean_square_error()
metrics.plot_predictions_with_ground_truth(fname=OUTPUT_DIR+subvolume+"_scatter.png")
metrics.plot_predictions_per_ground_truth(fname=OUTPUT_DIR+subvolume+"_pred_v_gt.png")
metrics.plot_ground_truths_per_prediction(fname=OUTPUT_DIR+subvolume+"_gt_v_pred.png")

with open(results_csv, 'a') as f:
    writer = csv.writer(f)
    writer.writerow([
        subvolume,
        len(a_centers),
        len(f_centers),
        accuracy,
        precision,
        recall,
        mse
    ])

# Draw the detected centers
orig_img = imread(orig_fname)

def min_max(x, minimum, maximum):
    return np.maximum(np.minimum(x, minimum), maximum)

def set_rgb(img, x, y, z, r, g, b):
    img[z, y, x, 0] = r
    img[z, y, y, 1] = g
    img[z, y, x, 2] = b

def draw_square(img, x, y, z, radi, rgb, copy=True, overwrite=True):
    z_range, y_range, x_range, _ = img.shape
    drawn_img = np.copy(img) if copy else img
    r, g, b = rgb
    for i in range(radi):
        for j in range(radi):
            for k in range(radi):
                if overwrite:
                    set_rgb(drawn_img,
                            min_max(x+i, x_range-1, 0),
                            min_max(y+j, y_range-1, 0),
                            min_max(z+k, z_range-1, 0),
                            r,
                            g,
                            b)
                else:
                    add_rgb(drawn_img,
                            min_max(x+i, x_range-1, 0),
                            min_max(y+j, y_range-1, 0),
                            min_max(z+k, z_range-1, 0),
                            r,
                            g,
                            b)
    return drawn_img

def save_tif(img, fname):
    if ".tif" not in fname:
        fname = fname + ".tif"
    save_path = "../img/"+fname if os.path.isdir("../img/") else fname
    save_path = "./img/"+fname if os.path.isdir("./img/") else fname
    imsave(save_path, img.astype(np.uint8))
    print("Saved tif as: ", fname, " at ", save_path)

def draw_centers(img, centers, rgb=None):
    r, g, b = None, None, None
    z_range, y_range, x_range = None, None, None
    if rgb:
        z_range, y_range, x_range, _ = img.shape
        r,g,b = rgb
    else:
        z_range, y_range, x_range = img.shape
    for z,y,x in centers:
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    z_d, y_d, x_d = max(min(z+i, z_range-1), 0), max(min(y+j, y_range-1), 0), max(min(x+k, x_range-1), 0)
                    if rgb:
                        img[z_d, y_d, x_d, 0] = r
                        img[z_d, y_d, x_d, 1] = g
                        img[z_d, y_d, x_d, 2] = b
                    else:
                        img[z_d, y_d, x_d] = 255

draw_centers(orig_img, f_centers, rgb=(255, 0, 0))
draw_centers(orig_img, a_centers, rgb=(0, 255, 0))
save_tif(orig_img, cell_fname+"_drawn_centers")

z_range, y_range, x_range, _ = orig_img.shape
blank_img = np.zeros((z_range, y_range, x_range))
draw_centers(blank_img, f_centers)
save_tif(blank_img, cell_fname+"_f_centers")

blank_img = np.zeros((z_range, y_range, x_range))
draw_centers(blank_img, a_centers)
save_tif(blank_img, cell_fname+"_a_centers")
