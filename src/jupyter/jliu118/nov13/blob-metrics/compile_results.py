import matplotlib.pyplot as plt
import numpy as np
# s3617, Atenolol2, Iso1

######## Accuracy ######

hdog_accuracies = [83, 84, 31]
farsight_accuracies = [60, 52, 27]

fig, ax = plt.subplots()

index = np.arange(3)
bar_width = 0.35
opacity = 0.5

hdog = plt.bar(index, hdog_accuracies, bar_width, color='b', label='HDoG', alpha=opacity)
farsight = plt.bar(index + bar_width, farsight_accuracies, bar_width, color='r', label='FARSIGHT', alpha=opacity)

plt.xlabel('Experiment Name')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of HDoG and FARSIGHT')
plt.xticks(index + bar_width/2, ('s3617', 'Atenolol2', 'Iso1'))
plt.legend()

plt.tight_layout()
plt.savefig('./data/plots/accuracy.png')

######## F-Measure ######

hdog_fmeasure = [0.224, 0.244, 0.81]
farsight_fmeasure = [0.68, 0.0002, 0.001]

fig, ax = plt.subplots()

index = np.arange(3)
bar_width = 0.35
opacity = 0.5

hdog = plt.bar(index, hdog_fmeasure, bar_width, color='b', label='HDoG', alpha=opacity)
farsight = plt.bar(index + bar_width, farsight_fmeasure, bar_width, color='r', label='FARSIGHT', alpha=opacity)

for i, v in enumerate(hdog_fmeasure):
    plt.annotate(str(v), (i-0.1, v))

for i, v in enumerate(farsight_fmeasure):
    plt.annotate(str(v), (i+0.25, v))

plt.xlabel('Experiment Name')
plt.ylabel('F-Measure')
plt.title('F-Measure of HDoG and FARSIGHT')
plt.xticks(index + bar_width/2, ('s3617', 'Atenolol2', 'Iso1'))
plt.legend()

plt.tight_layout()
plt.savefig('./data/plots/fmeasure.png')

######## Ground Truth per Predictions HDoG ######
fig, ax = plt.subplots()

index = np.arange(12)
bar_width = 0.35
opacity = 0.5

s3617_values = [125, 80, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
aten_values = [145, 85, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
iso_values = [0, 0, 0, 7, 4, 10, 3, 3, 4, 0, 1, 2]

s3617 = plt.bar(index, s3617_values, bar_width, color='b', label='s3617', alpha=opacity)
aten = plt.bar(index + bar_width, aten_values, bar_width, color='r', label='Atenolol2', alpha=opacity)
iso = plt.bar(index + (2*bar_width), iso_values, bar_width, color='g', label='Iso1', alpha=opacity)

plt.xlabel('# ground truths')
plt.ylabel('Count')
plt.title('Number of ground truth per prediction for HDoG')
plt.xticks(index + bar_width/2, index)
plt.legend()

plt.tight_layout()
plt.savefig('./data/plots/gtpr_hdog.png')

######## Predictions per ground truth HDoG ######
fig, ax = plt.subplots()

index = np.arange(11)
bar_width = 0.2
opacity = 0.5

s3617_values = [7, 35, 15, 2, 0, 0, 0, 0, 0, 0, 0]
aten_values = [10, 35, 17, 5, 0, 0, 0, 0, 0, 0, 0]
iso_values = [0, 1, 3, 12, 10, 8, 16, 9, 12, 8, 7]

s3617 = plt.bar(index, s3617_values, bar_width, color='b', label='s3617', alpha=opacity)
aten = plt.bar(index + bar_width, aten_values, bar_width, color='r', label='Atenolol2', alpha=opacity)
iso = plt.bar(index + (2*bar_width), iso_values, bar_width, color='g', label='Iso1', alpha=opacity)

plt.xlabel('# predictions')
plt.ylabel('Count')
plt.title('Number of predictions per ground truth label for HDoG')
plt.xticks(index + bar_width/2, index)
plt.legend()

plt.tight_layout()
plt.savefig('./data/plots/prgt_hdog.png')

######## Ground Truth per Predictions FARSIGHT ######
fig, ax = plt.subplots()

index = np.arange(4)
bar_width = 0.35
opacity = 0.5

s3617_values = [32, 50, 0, 0]
aten_values = [65000, 1000, 0, 0]
iso_values = [79000, 1000, 0, 0]

s3617 = plt.bar(index, s3617_values, bar_width, color='b', label='s3617', alpha=opacity)
aten = plt.bar(index + bar_width, aten_values, bar_width, color='r', label='Atenolol2', alpha=opacity)
iso = plt.bar(index + (2*bar_width), iso_values, bar_width, color='g', label='Iso1', alpha=opacity)

plt.xlabel('# ground truths')
plt.ylabel('Count')
plt.title('Number of ground truth per prediction for FARSIGHT')
plt.xticks(index + bar_width/2, index)
plt.legend()

plt.tight_layout()
plt.savefig('./data/plots/gtpr_fs.png')

######## Predictions per ground truth FARSIGHT ######
fig, ax = plt.subplots()

index = np.arange(8)
bar_width = 0.2
opacity = 0.5

s3617_values = [4, 50, 2, 0, 0, 0, 0, 0]
aten_values = [0, 7, 10, 27, 15, 4, 3, 2]
iso_values = [0, 0, 9, 14, 14, 9, 1, 0]

s3617 = plt.bar(index, s3617_values, bar_width, color='b', label='s3617', alpha=opacity)
aten = plt.bar(index + bar_width, aten_values, bar_width, color='r', label='Atenolol2', alpha=opacity)
iso = plt.bar(index + (2*bar_width), iso_values, bar_width, color='g', label='Iso1', alpha=opacity)

plt.xlabel('# predictions')
plt.ylabel('Count')
plt.title('Number of predictions per ground truth label for FARSIGHT')
plt.xticks(index + bar_width/2, index)
plt.legend()

plt.tight_layout()
plt.savefig('./data/plots/prgt_fs.png')
