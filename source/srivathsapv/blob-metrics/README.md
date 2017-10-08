# BlobMetrics

## Python package to evaluate the results of a blob detector.

Given ground truth values (list) and the predicted values (list) the package computes the following metrics

  - Accuracy - `accuracy()`
  - Precision - `precision()`
  - Recall - `recall()`
  - F-Measure - `f_measure()`
  - G-Measure - `g_measure()`
  - Mean Square Error - `mean_square_error()`

The package also plots a few interesting plots.

### Prediction vs Ground Truth Coordinates:

Uses PCA to reduce the data into 2 dimensions and plots an overlay of ground truths and predictions

```
plot_predictions_with_ground_truth()
```

### Plot precision recall summary

Plots the values of True Positives, False Positives, True Negatives and False Negatives in the precision-recall curve format.

```
plot_precision_recall_summary()
```

### Plot accuracy sensitivity vs euclidean distance threshold

Plots how the value of accuracy changes with the change in euclidean distance threshold

```
plot_accuracy_sensitivity()
```

### Plot F-Measure sensitivity vs euclidean distance threshold

Plots how the value of F-Measure changes with the change in euclidean distance threshold

```
plot_fmeasure_sensitivity()
```

### Plot Mean Square error sensitivity vs euclidean distance threshold

Plots how the value of Mean Square Error changes with the change in euclidean distance threshold

```
plot_mean_square_error_sensitivity()
```
