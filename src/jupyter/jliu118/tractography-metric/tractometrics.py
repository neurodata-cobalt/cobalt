import pandas as pd
import numpy as np
import tifffile as tiff

import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def compute_features(curves, angle_threshold=(np.pi/10), debug=False):
    source = curves[0][0]
    segments = []
    angles = []
    points = set()
#     print('curves')
#     print(curves)
    prev_p = None

    for curve in curves:
        if len(curve) < 2:
                continue
#         print('CURVE')
#         print(curve)
#         print('prev_p')
#         print(prev_p)
        if prev_p is not None:
            prev_v = prev_p - source
            first_curr_v = curve[1] - curve[0] # prev_p and curve[0] should be the same thing
            angle = angle_between(-prev_v, first_curr_v)
            angles.append(angle)
#             print('in prev_v')
#             print('curve[0]', curve[0])
#             print('curve[1]', curve[1])
#             print('angle:', angle)
            if angle > angle_threshold:
                    segments.append([source, prev_p])
                    source = prev_p
#                     print('UPDATING SOURCE:', source)
                    prev_p = curve[1]
        for i in range(1, len(curve) - 1):
#             print('in main loop')
            p1 = curve[i-1]
            p2 = curve[i]
            p3 = curve[i+1]
#             print('source:', source)
#             print('p2:', p2)
#             print('p3:', p3)
            v1 = p2 - source
            v2 = p3 - p2
            angle = angle_between(-v1, v2)
            angles.append(angle)
#             print('angle:', angle)
#             if i == 1 and angle > angle_threshold:
#                 segments.append([p1, p2])
#                 source = p2
#                 prev_v = v2
#             el
            if angle > angle_threshold:
                segments.append([source, p2])
#                 print('UPDATING SOURCE:', p2)
                source = p2
            prev_p = p3
    segments.append([source, prev_p])
            
#     print('SEGMENTS')
#     print(segments)
#     print('ANGLES')
#     print(angles)
#     print('asdf')
    
    return segments, angles


def compute_length_vector(segments, debug=False, show_histogram=False):
    # getting the segment lengths
    lengths = []
    for segment in segments:
        # getting segment lengths
        length = np.linalg.norm(segment[0] - segment[1])
        lengths.append(length)
    

    if debug:
        print('lengths')
        print(lengths)
        tmp1, tmp2, tmp3 = plt.hist(lengths, density=True, bins=20)
        plt.show()
    elif show_histogram:
        print('length histogram')
        tmp1, tmp2, tmp3 = plt.hist(lengths, density=True, bins=20)
        plt.show()
        
#     length_vector, bins, patches = plt.hist(lengths, density=True, bins=20)
    hist, bin_edges = np.histogram(lengths, bins=20, density=True)
    length_vector = hist * np.diff(bin_edges)

    return length_vector


def compute_angle_vector(angles, angle_threshold=(np.pi/10), debug=False, show_histogram=False):
    seg_angles = np.array([angle for angle in angles if angle > angle_threshold]) 
#     angle_vector, bins, patches = plt.hist(seg_angles, range=(angle_threshold, np.pi/2), density=True, bins=20)

    if debug:
        print('seg_angles')
        print(seg_angles)
        tmp, tmp2, tmp3 = plt.hist(seg_angles, density=True, bins=20)
        plt.show()
    elif show_histogram:
        print('angle histogram')
        tmp, tmp2, tmp3 = plt.hist(seg_angles, density=True, bins=20)
        plt.show()
    
    hist, bin_edges = np.histogram(seg_angles, range=(angle_threshold, np.pi), bins=20, density=True)
#     hist, bin_edges = np.histogram(seg_angles, bins=20, density=True)
    angle_vector = hist * np.diff(bin_edges)
    return angle_vector


def compute_feature_vector(curves, angle_threshold=(np.pi/10), debug=False, show_histogram=False):
    segments, angles = compute_features(curves, angle_threshold)
#     print('segments:')
#     print(segments)
    length_vector = compute_length_vector(segments, debug=debug, show_histogram=show_histogram)
    angle_vector = compute_angle_vector(angles, angle_threshold, debug=debug, show_histogram=show_histogram)
    feature_vector = np.concatenate([length_vector, angle_vector])
    return feature_vector