import numpy as np
import math
from sklearn import decomposition
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import scipy.misc

class BlobMetrics(object):

  def __init__(self, ground_truth_coords, predicted_coords, euclidean_distance_threshold=0):
    self.ground_truth_coords = ground_truth_coords
    self.predicted_coords = predicted_coords
    self.edist = euclidean_distance_threshold

    [self.tp, self.tn, self.fp, self.fn] = self._get_submetrics()

  def _get_submetrics(self, edist=None):
    if edist == None:
      edist = self.edist

    true_positives = []

    for p in self.predicted_coords:
      if self._find_nearest_point(self.ground_truth_coords, p, edist) is not None:
        true_positives.append(p)

    tp = float(len(true_positives))
    tn = 0.0

    fp = float(abs(len(self.predicted_coords) - tp))
    fn = float(abs(len(self.ground_truth_coords) - tp - fp))

    return [tp, tn ,fp, fn]

  def _euclidean_distance(self, p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

  def _find_nearest_point(self, points, point, edist=None):
    if edist == None:
      edist = self.edist

    min_dist = float('inf')
    nearest_point = None

    for p in points:
      euc_dist = self._euclidean_distance(p, point)
      if euc_dist <= edist and euc_dist < min_dist:
        min_dist = euc_dist
        nearest_point = p

    return nearest_point

  def _get_relative_color(self, n, max_n):
    c = (255.0 * n)/max_n

    return [(255-c)/255, (255-c)/255, (255-c)/255, 1]

  def accuracy(self):
    return (self.tp/float(len(self.ground_truth_coords))) * 100

  def precision(self):
    return self.tp/(self.tp + self.fp)

  def recall(self):
    return self.tp/(self.tp + self.fn)

  def mean_square_error(self, edist=None):
    if edist == None:
      edist = self.edist

    square_error_sum = 0.0
    n = 0

    for p in self.predicted_coords:
      nearest_point = self._find_nearest_point(self.ground_truth_coords, p, edist)
      if nearest_point is not None:
        square_error_sum += self._euclidean_distance(p, nearest_point) ** 2
        n += 1

    return square_error_sum/n

  def f_measure(self):
    p = self.precision()
    r = self.recall()

    return (2 * p * r)/(p + r)

  def g_measure(self):
    p = self.precision()
    r = self.recall()

    return math.sqrt(p * r)

  def plot_predictions_with_ground_truth(self):
    pca = decomposition.PCA(n_components=2)
    pca.fit(self.ground_truth_coords)

    gt_2d = pca.transform(self.ground_truth_coords)
    pr_2d = pca.transform(self.predicted_coords)

    gt_2d_x = [p[0] for p in gt_2d]
    gt_2d_y = [p[1] for p in gt_2d]

    pr_2d_x = [p[0] for p in pr_2d]
    pr_2d_y = [p[1] for p in pr_2d]

    fig, ax = plt.subplots()
    gt = ax.scatter(pr_2d_x, pr_2d_y, color='r', s=60, alpha=0.5)
    pr = ax.scatter(gt_2d_x, gt_2d_y, color='g', s=120, alpha=0.2)

    ax.set_xlabel('Principle Component #1', fontsize=10)
    ax.set_ylabel('Principle Component #2', fontsize=10)
    ax.set_title('Ground Truth vs Predicted', fontsize=12)
    ax.legend((gt, pr), ('Ground Truth', 'Predicted'), scatterpoints=1, loc='upper right', fontsize=10)

    fig.tight_layout()

    plt.show()

  def plot_precision_recall_summary(self):
    gt_len = len(self.ground_truth_coords)

    im = np.array(Image.open('img/empty_patch.png'), dtype=np.uint8)

    fig, ax = plt.subplots()
    ax.imshow(im)

    green_cmap = np.array(Image.open('img/green_cmap.png'), dtype=np.uint8)
    green_cmap = scipy.misc.imresize(green_cmap, (15, 250))

    red_cmap = np.array(Image.open('img/red_cmap.png'), dtype=np.uint8)
    red_cmap = scipy.misc.imresize(red_cmap, (15, 250))

    fig.figimage(red_cmap, 10, 20)
    plt.text(-130, 517, 'False positives and negatives', fontsize=8)

    fig.figimage(green_cmap, 330, 20)
    plt.text(270, 517, 'True positives and negatives', fontsize=8)

    fnr_color = [self._get_relative_color(self.fn, gt_len)[0], 0, 0, 1]
    tnr_color = [0, self._get_relative_color(self.tn, gt_len)[0], 0, 1]
    tpr_color = [0, self._get_relative_color(self.tp, gt_len)[0], 0, 1]
    fpr_color = [self._get_relative_color(self.fp, gt_len)[0], 0, 0, 1]

    rect_red = patches.Rectangle((0, 0), 250, 500, linewidth=3, edgecolor='black', facecolor=fnr_color)
    rect_green = patches.Rectangle((250, 0), 250, 500, linewidth=3, edgecolor='black', facecolor=tnr_color)

    left_semi = patches.Wedge((250, 250), 180, 90, 270, linewidth=3, edgecolor='black', facecolor=tpr_color)
    right_semi = patches.Wedge((250, 250), 180, 270, 90, linewidth=3, edgecolor='black', facecolor=fpr_color)

    ax.add_patch(rect_red)
    ax.add_patch(rect_green)
    ax.add_patch(left_semi)
    ax.add_patch(right_semi)

    plt.text(10, 20, 'False negatives={}'.format(self.fn), fontsize=10, color=np.ones(3)-fnr_color[:3])
    plt.text(325, 20, 'True negatives={}'.format(self.tn), fontsize=10, color=np.ones(3)-tnr_color[:3])
    plt.text(80, 250, 'True positives={}'.format(self.tp), fontsize=8, color=np.ones(3)-tpr_color[:3])
    plt.text(270, 250, 'False positives={}'.format(self.fp), fontsize=8, color=np.ones(3)-fpr_color[:3])

    ax.set_title('F Measure summary for # Ground Truth Labels = {}'.format(gt_len))

    fig.tight_layout()

    plt.axis('off')
    plt.show()

  def plot_accuracy_sensitivity(self):
    edist_range = list(set([abs(i) for i in range(self.edist-10, self.edist+11)]))
    accuracies = []

    for dist in edist_range:
      [tp, _, _, _ ] = self._get_submetrics(dist)
      accuracies.append((tp/float(len(self.ground_truth_coords))) * 100)

    fig, ax = plt.subplots()
    ax.plot(edist_range, accuracies, marker='o')

    ax.set_xlabel('Euclidean distance tolerance', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title('Euclidean distance tolerance vs Accuracy', fontsize=12)

    plt.show()

  def plot_fmeasure_sensitivity(self):
    edist_range = list(set([abs(i) for i in range(self.edist-10, self.edist+11)]))
    fmeasures = []

    for dist in edist_range:
      [tp, tn, fp, fn] = self._get_submetrics(dist)

      p = tp/(tp + fp)
      r = tp/(tp + fn)

      fmeasures.append((2 * p * r)/(p + r))

    fig, ax = plt.subplots()
    ax.plot(edist_range, fmeasures, marker='o')

    ax.set_xlabel('Euclidean distance tolerance', fontsize=10)
    ax.set_ylabel('F-measure', fontsize=10)
    ax.set_title('Euclidean distance tolerance vs F-measure', fontsize=12)

    plt.show()

  def plot_mean_square_error_sensitivity(self):
    edist_range = list(set([abs(i) for i in range(self.edist-10, self.edist+11)]))
    mse = []

    for dist in edist_range:
      mse.append(self.mean_square_error(dist))

    fig, ax = plt.subplots()
    ax.plot(edist_range, mse, marker='o')

    ax.set_xlabel('Euclidean distance tolerance', fontsize=10)
    ax.set_ylabel('Mean Square Error', fontsize=10)
    ax.set_title('Euclidean distance tolerance vs Mean Square Error', fontsize=12)

    plt.show()
