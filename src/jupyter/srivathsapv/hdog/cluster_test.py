import csv
from sklearn.cluster import KMeans, SpectralClustering, spectral_clustering
import numpy as np
from tifffile import imsave, imread
import math

from image_processing import draw_square, neighboring_pixels
from sklearn.feature_extraction import image
from sklearn.metrics import silhouette_score

orig_img = imread('./img/test_slice.tif')[:, :, :, 0]

def distance(p1, p2):
    z1, y1, x1 = p1
    z2, y2, x2 = p2

    #edist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    edist = abs(np.linalg.norm(np.array(p1) - np.array(p2)))

    exp = np.exp( - 0.0075 * (edist ** 2))
    return (exp/edist) if edist != 0 else exp

def edist(p1, p2):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def is_coord_range(p, p1, p2):
    min_x = min(p1[0], p2[0])
    min_y = min(p1[1], p2[1])
    min_z = min(p1[2], p2[2])

    max_x = max(p1[0], p2[0])
    max_y = max(p1[1], p2[1])
    max_z = max(p1[2], p2[2])

    return p[0] in range(min_x, max_x+1) and p[1] in range(min_y, max_y+1) and p[2] in range(min_z, max_z+1)

def intensity_between_points(p1, p2, orig_img):
    img = orig_img[:, :, :, 0]
    dist = edist(p1, p2)
    p_temp = p2

    points = [p2]
    for n in range(0, 2*int(dist)):
        min_dist = math.inf
        min_point = None

        for i,j,k in neighboring_pixels(p_temp[0], p_temp[1], p_temp[2]):
            pc = [i,j,k]
            pc_dist = edist(pc, p1)
            cond = (pc_dist < dist and is_coord_range(pc, p1, p2) and pc_dist < min_dist)

            if cond:
                min_dist = pc_dist
                min_point = pc

        if min_dist == math.inf:
            break

        p_temp = min_point
        draw_square(orig_img, p_temp[0], p_temp[1], p_temp[2], 2, [0, 0, 255], copy=False)

        points.append(p_temp)

    points.append(p1)

    avg_intensity = sum([img[p[0], p[1], p[2]] for p in points])/len(points)
    if dist == 0.0:
        return 100
    return avg_intensity/dist

with open('output_centers.csv') as csv_file:
    reader = csv.reader(csv_file)
    blob_candidates = [[int(l[0]), int(l[1]), int(l[2])] for l in list(reader)]

img = np.zeros((100, 150, 150, 3))

colors = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 255],
    [150, 0, 0],
    [0, 150, 0],
    [0, 0, 150],
    [150, 150, 0],
    [150, 0, 150],
    [0, 150, 150],
    [255, 150, 0]
]

distance_matrix = np.zeros((len(blob_candidates), len(blob_candidates)))

for i, p1 in enumerate(blob_candidates):
    for j, p2 in enumerate(blob_candidates):
        distance_matrix[i, j] = distance(p1, p2)

for n in range(10,20):
    print('n={}'.format(n))
    spectral = SpectralClustering(n_clusters=n, eigen_solver='arpack', affinity='precomputed')
    spectral.fit(distance_matrix)
    cluster_labels = spectral.labels_
    s_score = silhouette_score(blob_candidates, cluster_labels)
    print(s_score)

# for i, c in enumerate(blob_candidates):
#     draw_square(img, c[2], c[1], c[0], 2, colors[cluster_labels[i]], copy=False)
#
# imsave('cluster_test_clustered.tif', img.astype(np.uint8))
