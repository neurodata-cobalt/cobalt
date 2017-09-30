import numpy as np
import sys
import argparse
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
from scipy import ndimage
from decorators import save_img

THETA_SAMPLE_SIZE = 500
GRAY_SCALE_MAX = 255

class ImageGenerator:

    def __init__(self, x_range, y_range, z_range):
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range

    def _gauss_blur(self, img, sigma = 2):
        return ndimage.filters.gaussian_filter(img, sigma=sigma)

    def _gaussian_noise(self, mean, sigma, dim):
        if type(dim) != list:
            dim = list(dim)
        return np.random.normal(mean, sigma, dim)

    def _compute_random_transforms(self, 
                                    max_scale = 3, 
                                    max_rotation = np.pi*2, 
                                    max_shear = 3,
                                    ):
        P = np.zeros((4, 4))
        for i in range(4):
            P[i, i] = 1
        
        # Compute scale
        s_x, s_y, s_z = random_val(max_scale), random_val(max_scale), random_val(max_scale)
        for i, s in enumerate([s_x, s_y, s_z]):
            P[i, i] = s

        # Compute rotation
        theta = random_val(max_rotation)
        R_x = np.zeros((4, 4))
        R_x[0,0] = 1
        R_x[1,1] = np.cos(theta)
        R_x[2,2] = np.cos(theta)
        R_x[1,2] = -np.sin(theta)
        R_x[2,1] = np.sin(theta)

        R_y = np.zeros((4, 4))
        R_y[0,0] = np.cos(theta)
        R_y[0,2] = np.sin(theta)
        R_y[1,1] = 1
        R_y[2,0] = -np.sin(theta)
        R_y[2,2] = np.cos(theta)

        R_z = np.zeros((4, 4))
        R_z[0,0] = np.cos(theta)
        R_z[0,1] = -np.sin(theta)
        R_z[1,0] = np.sin(theta)
        R_z[1,1] = np.cos(theta)
        R_z[2,2] = 1

        P = np.dot(np.dot(np.dot(P, R_x), R_y), R_z)

        # Compute Shear
        sh_x_y, sh_x_z, sh_y_x, sh_y_z, sh_z_x, sh_z_y = random_val(max_shear), random_val(max_shear), \
                        random_val(max_shear), random_val(max_shear), random_val(max_shear), random_val(max_shear)
        Sh = np.zeros((4, 4))
        for i in range(4):
            Sh[i, i] = 1
        Sh[0, 1] = sh_x_y
        Sh[0, 2] = sh_x_z
        Sh[1, 0] = sh_y_x
        Sh[1, 2] = sh_y_z
        Sh[2, 0] = sh_z_x
        Sh[2, 1] = sh_z_y

        P = np.dot(P, Sh)

        # TODO: Add Homographies and prospective projections

        return P

    def _apply_transform(self, P, x, y, z):
        homo_coord = np.array([[x], [y], [z], [1]])
        coord = np.dot(P, homo_coord)
        x,y,z = np.nan_to_num(coord[0]//coord[3])[0], np.nan_to_num(coord[1]//coord[3])[0], np.nan_to_num(coord[2]//coord[3])[0]
        return (x,y,z)

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
                            random_center_sigma = 5,
                            randomized_intensity = False,
                            overlap = False,
                            add_cell_noise = False,
                            add_img_noise = False,
                            blur = False,
                            blur_sigma = 5,
                            affine_transform = False
                            ):

        z_n = self._z_range // (z_radi + z_margin)
        x_n = self._x_range // (x_radi + x_margin)
        y_n = self._y_range // (y_radi + y_margin)

        # Compute the centers
        cell_count = (z_n-1) * (x_n-1) * (y_n-1)
        centers = []
        if random_center:
            if random_center == 'gauss':
                x_mean, y_mean, z_mean = self._x_range/2, self._y_range/2, self._z_range/2
                x_sigma, y_sigma, z_sigma = self._x_range/random_center_sigma, self._y_range/random_center_sigma, self._z_range/random_center_sigma  # TODO: Get rid of the arbitrary variances
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
        ellipsoid = set() # set of points (x,y,z)
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
                    point = (int(x),int(y),int(z))
                    if point not in ellipsoid:
                        ellipsoid.add(point)

        # Draw the ellipsoids
        img = np.zeros((self._z_range, self._y_range, self._x_range, 3))
        for center in centers:
            noise = list(self._gaussian_noise(0, (GRAY_SCALE_MAX*intensity)**1/2 * 1/16, [len(ellipsoid)])) if add_cell_noise else None
            intensity_level = intensity if not randomized_intensity else random_val(intensity)
            x_c, y_c, z_c = center
            P = self._compute_random_transforms() if affine_transform else None
            for x,y,z in ellipsoid:
                fade_level = 1 if not fade else compute_fade_level(x, y, z, x_radi, y_radi, z_radi)
                gray_val = GRAY_SCALE_MAX * intensity_level * fade_level
                # print("fade: {}, gray: {}, intensity: {}".format(fade_level, gray_val, intensity))
                x_d, y_d, z_d = x + x_c, y + y_c, z + z_c
                if bound_check(x_d, 0, self._x_range) and bound_check(y_d, 0, self._y_range) and bound_check(z_d, 0, self._z_range):
                    noise_val = noise.pop() if noise else 0
                    if P is not None:
                        x_d, y_d, z_d = self._apply_transform(P, x_d, y_d, z_d)
                    set_rgb(img, x_d, y_d, z_d, gray_val+noise_val, gray_val+noise_val, gray_val+noise_val, overlap=overlap)
        
        # Add gaussian blur
        if blur:
            img = self._gauss_blur(img, sigma = blur_sigma)

        # Add image noise
        if add_img_noise:
            noise = self._gaussian_noise(0, GRAY_SCALE_MAX * 0.001, [self._z_range, self._y_range, self._x_range])
            for i in range(3):
                img[:, :, :, i] = img[:, :, :, i] + noise

        return img, centers
    
def compute_fade_level(x,y,z,x_radi, y_radi, z_radi):
    return 1.0 - (x**2/x_radi**2 + y**2/y_radi**2 + z**2/z_radi**2)

def set_rgb(img, x, y, z, r, g, b, overlap=False):
    if overlap:
        # print("set gray: {}, gray: {}, prev: {}".format(min(r+img[int(z),int(y),int(x),0], GRAY_SCALE_MAX), r, img[int(z),int(y),int(x),0]))
        img[int(z),int(y),int(x),0] = min(r+img[int(z),int(y),int(x),0], GRAY_SCALE_MAX)
        img[int(z),int(y),int(x),1] = min(g+img[int(z),int(y),int(x),1], GRAY_SCALE_MAX)
        img[int(z),int(y),int(x),2] = min(b+img[int(z),int(y),int(x),2], GRAY_SCALE_MAX)
    else:
        img[int(z),int(y),int(x),0] = r
        img[int(z),int(y),int(x),1] = g
        img[int(z),int(y),int(x),2] = b

def test():
    img_gen = ImageGenerator(1000,1000,100)
    img_gen.make_ellipsoidal_image(
        25,
        25,
        10,
        150,
        150,
        10,
        # fade = True,
        random_center = 'gauss',
        # randomized_intensity = True,
        fname = "ellipsoid_gen_test",
        overlap = True,
        # blur = True,
        # add_cell_noise = True,
        # add_img_noise = True,
        # affine_transform = True,
    )
if __name__ == "__main__":
    # args = arg_parser()
    test()

                
