import numpy as np
import sys
from helper import (
    save_tif, 
    min_max, 
    random_3D, 
    bound_check,
    random_val,
    gauss_3D
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

THETA_SAMPLE_SIZE = 500
GRAY_SCALE_MAX = 255

def save_img(func):
    def gen_image(self, *args, **kwargs):
        fname = kwargs.get('fname')
        img = None
        if fname:
            del kwargs['fname']
        if kwargs:
            img = func(self, *args, **kwargs)
        else:
            img = func(self, *args)
        if fname:
            save_tif(img, fname)
    return gen_image

class ImageGenerator:
    def __init__(self, x_range, y_range, z_range):
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range

    @save_img
    def make_ellipsoidal_image(
                            self,
                            x_radi,
                            y_radi,
                            z_radi,
                            x_margin,
                            y_margin,
                            z_margin,
                            fade = False,
                            intensity = 1.0,
                            random_center = False,
                            randomized_intensity = False,
                            overlap = False,
                            ):
        img = np.zeros((self._z_range, self._y_range, self._x_range, 3))
        z_n = self._z_range // (z_radi + z_margin)
        x_n = self._x_range // (x_radi + x_margin)
        y_n = self._y_range // (y_radi + y_margin)

        # Compute the centers
        cell_count = (z_n-1) * (x_n-1) * (y_n-1)
        centers = []
        if random_center:
            if random_center == 'gauss':
                x_mean, y_mean, z_mean = self._x_range/2, self._y_range/2, self._z_range/2
                x_sigma, y_sigma, z_sigma = self._x_range/5, self._y_range/5, self._z_range/5
                for _ in range(cell_count):
                    centers.append(gauss_3D(x_mean, y_mean, z_mean, x_sigma, y_sigma, z_sigma)) 
            else:
                for _ in range(cell_count):
                    centers.append(random_3D((0, self._x_range), (0, self._y_range), (0, self._z_range))) 
        else:
            for k in range(1, z_n):
                z = k*(z_radi + z_margin)-1
                for j in range(1, y_n):
                    y = j*(y_radi + y_margin)-1
                    for i in range(1, x_n):
                        x = i*(x_radi + x_margin)-1
                        centers.append([x,y,z])
        print("{} cells will be drawn on a {} (x) by {} (y) by {} (z) image".format(cell_count, self._x_range, self._y_range, self._z_range))

        # Sample theta
        theta_pos = np.linspace(np.pi/2,  1.5* np.pi, THETA_SAMPLE_SIZE)
        theta_neg = np.linspace(1.5* np.pi,  2.5* np.pi, THETA_SAMPLE_SIZE)

        # Compute elipsoid with given radi
        ellipsoid = [] # list of points (x,y,z)
        for z in range(-z_radi, z_radi + 1):
            ro = np.arccos(float(z)/z_radi)
            y = y_radi * np.sin(theta_pos) * np.sin(ro)
            x_pos = x_radi * np.cos(theta_pos) * np.sin(ro)
            x_neg = x_radi * np.cos(theta_neg) * np.sin(ro)
            xx_coord = zip(x_pos, x_neg)
            xy_coord = zip(xx_coord,y)
            for x_pack,y in xy_coord:
                x_pos = x_pack[0]
                x_neg = x_pack[1]
                for x in range(int(x_pos), int(x_neg)):
                    ellipsoid.append((x,y,z))

        # Draw the ellipsoids
        for center in centers:
            x_c, y_c, z_c = center
            for x,y,z in ellipsoid:
                fade_level = 1 if not fade else compute_fade_level(x, y, z, x_radi, y_radi, z_radi)
                intensity_level = intensity if not randomized_intensity else random_val(intensity)
                gray_val = GRAY_SCALE_MAX * intensity * fade_level
                # print("fade: {}, gray: {}, intensity: {}".format(fade_level, gray_val, intensity))
                x_d, y_d, z_d = x + x_c, y + y_c, z + z_c
                if bound_check(x_d, 0, self._x_range) and bound_check(y_d, 0, self._y_range) and bound_check(z_d, 0, self._z_range):
                    set_rgb(img, x_d, y_d, z_d, gray_val, gray_val, gray_val, overlap=overlap)
        return img
    
def compute_fade_level(x,y,z,x_radi, y_radi, z_radi):
    return 1.0 - (x**2/x_radi**2 + y**2/y_radi**2 + z**2/z_radi**2)

def set_rgb(img, x, y, z, r, g, b, overlap=False):
    if overlap:
        img[int(z),int(y),int(x),0] = min(r+img[int(z),int(y),int(x),0], GRAY_SCALE_MAX)
        img[int(z),int(y),int(x),1] = min(g+img[int(z),int(y),int(x),1], GRAY_SCALE_MAX)
        img[int(z),int(y),int(x),2] = min(b+img[int(z),int(y),int(x),2], GRAY_SCALE_MAX)
    else:
        img[int(z),int(y),int(x),0] = r
        img[int(z),int(y),int(x),1] = g
        img[int(z),int(y),int(x),2] = b

def generate_validation_suite():
    img_gen = ImageGenerator(1000,1000,100)

    # Most basic, easiest image with solid well separated cells
    img_gen.make_ellipsoidal_image(
        50,
        50,
        10,
        200,
        200,
        10,
        fname = "36_cell_array",
    )

    # Harder gaussian distributed centers with borderless overlap and fading blobs
    img_gen.make_ellipsoidal_image(
        50,
        50,
        10,
        200,
        200,
        10,
        fade = True,
        random_center = 'gauss',
        randomized_intensity = True,
        fname = "36_cell_fade_gauss_overlap",
        overlap = True
    )

def test():
    img_gen = ImageGenerator(1000,1000,100)
    img_gen.make_ellipsoidal_image(
        50,
        50,
        10,
        200,
        200,
        10,
        fade = True,
        random_center = 'gauss',
        randomized_intensity = True,
        fname = "ellipsoid_gen_test",
        overlap = True
    )
if __name__ == "__main__":
    # generate_validation_suite()
    test()
                
