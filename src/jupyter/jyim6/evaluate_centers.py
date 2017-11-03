import numpy as np
import csv
import argparse
import sys
from collections import namedtuple
sys.path.append('./../../util/')
sys.path.append('./../srivathsapv/blob-metrics/')
from BlobMetrics import BlobMetrics
from tifffile import imsave, imread
from ImageDrawer import ImageDrawer

EUC_DIST_THRES = 5
OUTPUT_DIR = './farsight_output/'

parser = argparse.ArgumentParser()
parser.add_argument('farsight_centers', type=str)
parser.add_argument('annotated_centers', type=str)
parser.add_argument('results_csv', type=str)
parser.add_argument('drawn_output', type=str)
parser.add_argument('orig_tif', type=str)
args = parser.parse_args()

f_centers_fname, a_centers_fname = args.farsight_centers, args.annotated_centers
results_csv = args.results_csv
drawn_output, orig_fname = args.drawn_output, args.orig_tif
subvolume = f_centers_fname.split("/")[-1][:-12]    # Extract the subvolume name

# Read in the annotated centers
a_centers = []
AnnotatedCenter = namedtuple('AnnotatedCenter', ['id', 'area', 'mean', 'min', 'max', 'x', 'y', 'z', 'counter', 'count'])
with open(a_centers_fname, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        c = AnnotatedCenter(*row)
        a_centers.append(
            (int(float(c.z)), int(float(c.y)), int(float(c.x)))
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

print("{} annotated centers, {} detected centers".format(len(a_centers), len(f_centers)))
import pdb; pdb.set_trace()

# Compute blob metrics
metrics = BlobMetrics(a_centers, f_centers, EUC_DIST_THRES)
accuracy = metrics.accuracy()
precision = metrics.precision()
recall = metrics.recall()
mse = metrics.mean_square_error()
metrics.plot_predictions_with_ground_truth(fname=OUTPUT_DIR+subvolume+"_scatter.png")
metrics.plot_predictions_per_ground_truth(fname=OUTPUT_DIR+subvolume+"_pred_v_gt.png")
metrics.plot_ground_truths_per_prediction(fname=OUTPUT_DIR+subvolume+"_gt_v_pred.png")


with open(results_csv, 'w') as f:
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
ImageDrawer.draw_centers(orig_img, seed_centers, (255,0,0), fname=drawn_output, copy=True)
