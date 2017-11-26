import numpy as np
import csv

saved_centroids = np.genfromtxt('cell_detection_bunch.csv', delimiter=',', dtype=np.dtype(int))

centroids_ordered = saved_centroids[np.argsort(saved_centroids[:,0])]
# z slices are 1 indexed
for centroid in centroids_ordered:
    centroid[0] -= 1

split_centroids = [[], [], [], []]
for centroid in centroids_ordered:
    if centroid[0] < 100:
        split_centroids[0].append(centroid)
    elif centroid[0] < 200:
        split_centroids[1].append(centroid)
    elif centroid[0] < 300:
        split_centroids[2].append(centroid)
    elif centroid[0] < 400:
        split_centroids[3].append(centroid)

print(split_centroids[0])

np.savetxt('cell_detection_0.csv', split_centroids[0], delimiter=',', dtype=np.dtype(int))

# with open("cell_detection_0.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(split_centroids[0])
# with open("cell_detection_1.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(split_centroids[1])
# with open("cell_detection_2.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(split_centroids[2])
# with open("cell_detection_3.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(split_centroids[3])
