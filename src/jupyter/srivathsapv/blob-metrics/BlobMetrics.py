#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import scipy.misc
import time
from tifffile import imread, imsave
import json
import operator

class BlobMetrics(object):

    def __init__(
        self,
        ground_truth_coords,
        predicted_coords,
        euclidean_distance_threshold=0,
        ):
        self.ground_truth_coords = ground_truth_coords
        self.predicted_coords = predicted_coords
        self.edist = euclidean_distance_threshold

        [self.tp, self.tn, self.fp, self.fn] = self._get_submetrics()

    def _get_submetrics(self, edist=None):
        if edist == None:
            edist = self.edist

        tp = self._get_true_positives(edist)
        tn = 0.0

        fp = float(abs(len(self.predicted_coords) - tp))
        fn = float(abs(len(self.ground_truth_coords) - tp - fp))

        return [tp, tn, fp, fn]

    def _euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_true_positives(self, edist=None):
        if edist == None:
            edist = self.edist

        point_map = {}

        for p in self.ground_truth_coords:
            point_map[repr(p)] = []

        for (i, p) in enumerate(self.predicted_coords):
            (nearest_point, dist) = \
                self._find_nearest_point(self.ground_truth_coords, p,
                    edist)
            if nearest_point is not None:
                d = {'point': p, 'edist': dist}
                point_map[repr(nearest_point)].append(d)

        predicted_points_with_match = set()
        for k in point_map:
            dist_info = point_map[k]
            dist_info = sorted(dist_info, key=lambda l: l['edist'])
            if len(dist_info) > 0:
                predicted_points_with_match.add(repr(dist_info[0]['point'
                        ]))

        return float(len(list(predicted_points_with_match)))

    def _find_nearest_point(
        self,
        points,
        point,
        edist=None,
        return_candidate_points=False,
        ):
        if edist == None:
            edist = self.edist

        min_dist = float('inf')
        nearest_point = None
        candidate_points = []

        for p in points:
            euc_dist = self._euclidean_distance(p, point)
            if euc_dist <= edist and euc_dist < min_dist:
                min_dist = euc_dist
                nearest_point = p
                candidate_points.append(p)

        if return_candidate_points:
            return [nearest_point, min_dist, candidate_points]

        return [nearest_point, min_dist]

    def accuracy(self):
        return self.tp / float(len(self.ground_truth_coords)) * 100

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def mean_square_error(self, edist=None):
        if edist == None:
            edist = self.edist

        square_error_sum = 0.0
        n = 0

        for p in self.predicted_coords:
            (nearest_point, dist) = \
                self._find_nearest_point(self.ground_truth_coords, p,
                    edist)
            if nearest_point is not None:
                square_error_sum += self._euclidean_distance(p,
                        nearest_point) ** 2
                n += 1
        if n == 0:
            return 0
        return square_error_sum / n

    def f_measure(self):
        p = self.precision()
        r = self.recall()

        return 2 * p * r / (p + r)

    def g_measure(self):
        p = self.precision()
        r = self.recall()

        return math.sqrt(p * r)

    def plot_predictions_with_ground_truth(self, fname=None):
        x_gt = np.array([i[0] for i in self.ground_truth_coords])
        y_gt = np.array([i[1] for i in self.ground_truth_coords])
        z_gt = np.array([i[2] for i in self.ground_truth_coords])

        x_pr = np.array([i[0] for i in self.predicted_coords])
        y_pr = np.array([i[1] for i in self.predicted_coords])
        z_pr = np.array([i[2] for i in self.predicted_coords])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        gt = ax.scatter(
            x_gt,
            y_gt,
            z_gt,
            c='green',
            marker='o',
            alpha=0.5,
            )
        pr = ax.scatter(
            x_pr,
            y_pr,
            z_pr,
            c='red',
            s=50,
            marker='o',
            alpha=0.2,
            )

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        ax.legend((gt, pr), ('Ground Truth', 'Predicted'),
                  scatterpoints=1, loc='lower left', fontsize=10)
        ax.set_title('Ground Truth vs Predicted', fontsize=12)

        fig.tight_layout()

        if fname:
            plt.savefig(fname)
        else:
            plt.show()

    def plot_accuracy_sensitivity(self):
        edist_range = list(set([abs(i) for i in range(self.edist - 10,
                           self.edist + 11)]))
        accuracies = []

        for dist in edist_range:
            [tp, _, _, _] = self._get_submetrics(dist)
            accuracies.append(tp / float(len(self.ground_truth_coords))
                              * 100)

        (fig, ax) = plt.subplots()
        ax.plot(edist_range, accuracies, marker='o')

        ax.set_xlabel('Euclidean distance threshold', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Euclidean distance threshold vs Accuracy',
                     fontsize=12)

        plt.show()

    def plot_fmeasure_sensitivity(self):
        edist_range = list(set([abs(i) for i in range(self.edist - 10,
                           self.edist + 11)]))
        fmeasures = []

        for dist in edist_range:
            [tp, tn, fp, fn] = self._get_submetrics(dist)

            p = tp / (tp + fp)
            r = tp / (tp + fn)

            fmeasures.append(2 * p * r / (p + r))

        (fig, ax) = plt.subplots()
        ax.plot(edist_range, fmeasures, marker='o')

        ax.set_xlabel('Euclidean distance threshold', fontsize=10)
        ax.set_ylabel('F-measure', fontsize=10)
        ax.set_title('Euclidean distance threshold vs F-measure',
                     fontsize=12)

        plt.show()

    def plot_mean_square_error_sensitivity(self):
        edist_range = list(set([abs(i) for i in range(self.edist - 10,
                           self.edist + 11)]))
        mse = []

        for dist in edist_range:
            mse.append(self.mean_square_error(dist))

        (fig, ax) = plt.subplots()
        ax.plot(edist_range, mse, marker='o')

        ax.set_xlabel('Euclidean distance threshold', fontsize=10)
        ax.set_ylabel('Mean Square Error', fontsize=10)
        ax.set_title('Euclidean distance threshold vs Mean Square Error'
                     , fontsize=12)

        plt.show()

    def plot_predictions_per_ground_truth(self, fname=None):
        counts = {}
        for p in self.ground_truth_coords:
            (_, _, candidate_points) = \
                self._find_nearest_point(self.predicted_coords, p,
                    self.edist, True)
            c = len(candidate_points)
            if c not in counts:
                counts[c] = 1
            else:
                counts[c] += 1

        (fig, ax) = plt.subplots()

    # import pdb; pdb.set_trace()

        ax.bar(list(counts.keys()), list(counts.values()))
        ax.set_xlabel('# predictions', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        plt.xticks(list(counts.keys()), list(counts.keys()))
        ax.set_title('Number of predictions per ground truth label',
                     fontsize=12)
        if fname:
            plt.savefig(fname)
        else:
            plt.show()

    def plot_ground_truths_per_prediction(self, fname=None):
        counts = {}
        for p in self.predicted_coords:
            (_, _, candidate_points) = \
                self._find_nearest_point(self.ground_truth_coords, p,
                    self.edist, True)
            c = len(candidate_points)
            if c not in counts:
                counts[c] = 1
            else:
                counts[c] += 1

        (fig, ax) = plt.subplots()
        ax.bar(list(counts.keys()), list(counts.values()))
        ax.set_xlabel('# ground truths', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        plt.xticks(list(counts.keys()), list(counts.keys()))
        ax.set_title('Number of ground truth labels per prediction',
                     fontsize=12)
        if fname:
            plt.savefig(fname)
        else:
            plt.show()

    def _get_child_nodes_from_ontology(self, node, id2name):
        id2name[node['id']] = node['name']
        # base case
        if node['children'] == []:
            return id2name
        # other case
        for i in range(len(node['children'])):
            id2name = self._get_child_nodes_from_ontology(node['children'][i], id2name)
        return id2name

    def _find_region_for_point(self, point, region2voxel):
        for r in region2voxel.keys():
            if point in region2voxel[r]:
                return r
        return -1

    def get_region_based_count(self, ontology_file, registered_brain_tif):
        ontology_json = json.load(file(ontology_file))
        id2name = self._get_child_nodes_from_ontology(ontology_json, {})
        id2name[32767] = 'background'

        #print(id2name)
        background_region_number = 0

        registered_brain = imread(registered_brain_tif).astype(np.uint64)

        region_numbers = list(np.unique(registered_brain, return_counts=True)[0])

        region2voxel = {}

        for region in region_numbers:
            print('region {}'.format(region))
            voxels = np.where(registered_brain == region)
            region2voxel[region] = map(list, zip(*voxels))

        region_count = {}

        for p in self.predicted_coords:
            print('point {}'.format(p))
            rp = self._find_region_for_point(p, region2voxel)
            if id2name[rp] in region_count:
                region_count[id2name[rp]] += 1
            else:
                region_count[id2name[rp]] = 1

        return region_count

    def plot_region_based_count(self, count_statistics=None):
        count_statistics = sorted(count_statistics.items(), key=operator.itemgetter(1))
        count_statistics.reverse()
        print(count_statistics)

        bar_x = np.arange(5)
        bar_y = [c[1] for c in count_statistics[:5]]
        labels = [c[0] for c in count_statistics[:5]]
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'black', 'gray']

        (fig, ax) = plt.subplots()

        for i,y in enumerate(bar_y):
            ax.bar(i, y, label=labels[i], color=colors[i], width=0.8, align='center', alpha=0.6)

        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        #plt.xticks(list(counts.keys()), list(counts.keys()))
        ax.set_title('Region wise cell count',
                     fontsize=16)
        ax.legend()

        plt.savefig('data/plots/reg.png')
