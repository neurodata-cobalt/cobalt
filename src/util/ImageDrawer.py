import numpy as np
import sys
import argparse
from helper import (
    save_tif,
    min_max,
    random_3D,
    bound_check,
    random_val,
    gauss_3D,
    set_rgb,
    add_rgb
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from decorators import save_img


class ImageDrawer:
    x_range = 1000
    y_range = 1000
    z_range = 100
    channels = 3

    @classmethod
    @save_img
    def draw_centers(cls, img, centers, rgb, copy=False, overwrite=True, print_level=0):
        x_range, y_range, z_range, chans = img.shape
        cls.x_range, cls.y_range, cls.z_range = x_range, y_range, z_range
        drawn_img = np.copy(img) if copy else img
        i = 1
        for (z, y, x) in centers:
            if (i % 2500 == 0) and print_level:
                print("{} centers drawn".format(i))
            cls.draw_square(drawn_img, x, y, z, 3, rgb, overwrite=overwrite)
        return drawn_img

    @classmethod
    @save_img
    def draw_square(cls, img, x, y, z, radi, rgb, copy=False, overwrite=True):
        drawn_img = np.copy(img) if copy else img
        r, g, b = rgb
        for i in range(radi):
            for j in range(radi):
                for k in range(radi):
                    if overwrite:
                        set_rgb(drawn_img,
                                min_max(x+i, cls.x_range-1, 0),
                                min_max(y+j, cls.y_range-1, 0),
                                min_max(z+k, cls.z_range-1, 0),
                                r,
                                g,
                                b)
                    else:
                        add_rgb(drawn_img,
                                min_max(x+i, cls.x_range-1, 0),
                                min_max(y+j, cls.y_range-1, 0),
                                min_max(z+k, cls.z_range-1, 0),
                                r,
                                g,
                                b)
        return drawn_img
