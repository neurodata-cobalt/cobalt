
# coding: utf-8

# # BlobMetrics
#
# ### This notebook explains the usage of the BlobMetrics package with a few samples

# ** Import the necessary packages **

# In[1]:


import csv
from BlobMetrics import BlobMetrics
import ipyvolume as ipv
import numpy as np


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


ground_truth_coords = get_coords_from_csv('./data/s3617_manual.csv')
predicted_coords = get_coords_from_csv('./data/s3617_predicted_nocluster.csv')


# ** Sample of the CSV file... **

# In[4]:


print(ground_truth_coords[0])


# ** Define the euclidean distance threshold - euclidean distance between ground truth blob location and predicted blob location upto which it will considered as a match**

# In[5]:


euclidean_distance_threshold = 12


# ** Define the instance of BlobMetrics **

# In[6]:


metrics = BlobMetrics(ground_truth_coords, predicted_coords, euclidean_distance_threshold)


# ### Numeric Metrics

# ** Accuracy **

# In[7]:


accuracy = metrics.accuracy()
print('Accuracy: {}'.format(accuracy))


# ** Precision **

# In[8]:


precision = metrics.precision()
print('Precision: {}'.format(precision))


# ** Recall **

# In[9]:


recall = metrics.recall()
print('Recall: {}'.format(recall))


# ** F-Measure **

# In[10]:


fmeasure = metrics.f_measure()
print('F-Measure: {}'.format(fmeasure))


# ** G-Measure **

# In[11]:


gmeasure = metrics.g_measure()
print('G-Measure: {}'.format(gmeasure))


# ** Mean Square Error **

# In[12]:


mse = metrics.mean_square_error()
print('Mean Square Error: {}'.format(mse))


# ### Visual Plots

# ** Predictions vs Ground Truth Labels **

# In[13]:


metrics.plot_predictions_with_ground_truth()


# ** Number of predictions per ground truth label **

# In[14]:


metrics.plot_predictions_per_ground_truth()


# ** Number of ground truth labels per prediction **

# In[15]:


metrics.plot_ground_truths_per_prediction()


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
