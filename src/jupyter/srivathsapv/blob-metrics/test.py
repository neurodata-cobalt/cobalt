
# coding: utf-8

# # BlobMetrics
#
# ### This notebook explains the usage of the BlobMetrics package with a few samples

# ** Import the necessary packages **

# In[1]:


import csv
from BlobMetrics import BlobMetrics
import numpy as np
import sys
import json
import operator
import matplotlib.pyplot as plt

# ** Define method to get co-ordinates given the CSV file path **

# In[2]:


def get_coords_from_csv(filepath):
  with open(filepath) as infile:
    rows = csv.reader(infile)
    coords = [[float(r[0]), float(r[1]), float(r[2])] for r in list(rows)]

  return coords


# ** Load ground truth and predicted co-ordinates of blurred_147_cells data**
#

# In[3]:

gt_path = sys.argv[1]
pr_path = sys.argv[2]
name_prefix = sys.argv[3]

ground_truth_coords = get_coords_from_csv(gt_path)
predicted_coords = get_coords_from_csv(pr_path)

ontology_file = './data/ara_structure_ontology.json'
registered_brain_tif = './data/reg.tiff'

# ** Sample of the CSV file... **

# In[4]:


print(ground_truth_coords[0])


# ** Define the euclidean distance threshold - euclidean distance between ground truth blob location and predicted blob location upto which it will considered as a match**

# In[5]:


euclidean_distance_threshold = 12


# ** Define the instance of BlobMetrics **

# In[6]:


modified_predicted_coords = []

for c in predicted_coords:
    z, y, x = [int(c[0]/2), int(c[1]/2), int(c[2]/2)]
    modified_predicted_coords.append([z,y,x])

metrics = BlobMetrics(ground_truth_coords, modified_predicted_coords, euclidean_distance_threshold)

#region_based_count = metrics.get_region_based_count(ontology_file, registered_brain_tif)

region_based_count = {
    "Primary somatosensory area, upper limb, layer 6a": 2,
    "Agranular insular area, posterior part, layer 6a": 1,
    "Lateral posterior nucleus of the thalamus": 1,
    "Central amygdalar nucleus, medial part": 1,
    "Caudoputamen": 47,
    "fimbria": 1,
    "Endopiriform nucleus, dorsal part": 2,
    "Lateral amygdalar nucleus": 1,
    "Nucleus of the posterior commissure": 1,
    "Primary somatosensory area, upper limb, layer 2/3": 1,
    "Triangular nucleus of septum": 1,
    "Gustatory areas, layer 6b": 1,
    "Gustatory areas, layer 6a": 5,
    "Endopiriform nucleus, ventral part": 1,
    "Anterior cingulate area, dorsal part, layer 6a": 1,
    "external capsule": 3,
    "Supplemental somatosensory area, layer 6a": 21,
    "anterior commissure, temporal limb": 1,
    "Striatum": 3,
    "Primary somatosensory area, trunk, layer 6b": 1,
    "fiber tracts": 2,
    "Primary somatosensory area, mouth, layer 5": 10,
    "Primary somatosensory area, mouth, layer 4": 4,
    "alveus": 2,
    "Central amygdalar nucleus, lateral part": 1,
    "background": 15,
    "Primary somatosensory area, nose, layer 4": 14,
    "Primary somatosensory area, nose, layer 5": 24,
    "Primary somatosensory area, barrel field, layer 6a": 3,
    "Central amygdalar nucleus, capsular part": 3,
    "Claustrum": 1,
    "Primary somatosensory area, barrel field, layer 5": 14,
    "Primary somatosensory area, barrel field, layer 4": 2,
    "Orbital area, lateral part, layer 5": 1,
    "Globus pallidus, external segment": 1,
    "Supplemental somatosensory area, layer 5": 26,
    "Field CA1": 6,
    "Field CA3": 1,
    "Basolateral amygdalar nucleus, anterior part": 9,
    "root": 1,
    "Visceral area, layer 6a": 2
}

metrics.plot_region_based_count(region_based_count)

intensity_data = json.loads(open('intensity.json').read())

intensity_data = sorted(intensity_data.items(), key=operator.itemgetter(1))
intensity_data.reverse()
print(intensity_data[:5])

bar_x = np.arange(5)
bar_y = [c[1] for c in intensity_data[1:6]]
labels = [c[0] for c in intensity_data[1:6]]
colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'black', 'gray']

(fig, ax) = plt.subplots()

for i,y in enumerate(bar_y):
    ax.bar(i, y, label=labels[i], color=colors[i], width=0.8, align='center', alpha=0.6)

ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Intensity', fontsize=12)
#plt.xticks(list(counts.keys()), list(counts.keys()))
ax.set_title('Region wise intensity sum',
             fontsize=16)
ax.legend()

plt.savefig('data/plots/reg-intensity.png')

# ### Numeric Metrics

# ** Accuracy **

# In[7]:


# accuracy = metrics.accuracy()
# print('Accuracy: {}'.format(accuracy))
#
#
# # ** Precision **
#
# # In[8]:
#
#
# precision = metrics.precision()
# print('Precision: {}'.format(precision))
#
#
# # ** Recall **
#
# # In[9]:
#
#
# recall = metrics.recall()
# print('Recall: {}'.format(recall))
#
#
# # ** F-Measure **
#
# # In[10]:
#
#
# fmeasure = metrics.f_measure()
# print('F-Measure: {}'.format(fmeasure))
#
#
# # ** G-Measure **
#
# # In[11]:
#
#
# gmeasure = metrics.g_measure()
# print('G-Measure: {}'.format(gmeasure))
#
#
# # ** Mean Square Error **
#
# # In[12]:
#
#
# mse = metrics.mean_square_error()
# print('Mean Square Error: {}'.format(mse))
#
#
# # ### Visual Plots
#
# # ** Predictions vs Ground Truth Labels **
#
# # In[13]:
#
#
# metrics.plot_predictions_with_ground_truth('./data/plots/{}-prgt.png'.format(name_prefix))
#
#
# # ** Number of predictions per ground truth label **
#
# # In[14]:
#
#
# metrics.plot_predictions_per_ground_truth('./data/plots/{}-prwgt.png'.format(name_prefix))
#
#
# # ** Number of ground truth labels per prediction **
#
# # In[15]:
#
#
# metrics.plot_ground_truths_per_prediction('./data/plots/{}-gtwpr.png'.format(name_prefix))


# ** Accuracy Sensitivity to Euclidean Distance Threshold **

# In[16]:


# metrics.plot_accuracy_sensitivity()
#
#
# # ** F-Measure Sensitivity to Euclidean Distance Threshold **
#
# # In[17]:
#
#
# metrics.plot_fmeasure_sensitivity()
#
#
# # ** Mean Square Error Sensitivity to Euclidean Distance Threshold **
#
# # In[18]:
#
#
# metrics.plot_mean_square_error_sensitivity()


# In[ ]:
