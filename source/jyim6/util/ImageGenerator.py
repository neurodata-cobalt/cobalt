import numpy as np
import sys
from helper import save_tif
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

THETA_SAMPLE_SIZE = 500
GRAY_SCALE_MAX = 255

class ImageGenerator:
    def __init__(self, x_range, y_range, z_range):
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range

    def make_spherical_image(
                            self,
                            x_radi,
                            y_radi,
                            z_radi,
                            x_margin,
                            y_margin,
                            z_margin,
                            fname = None,
                            ):
        img = np.zeros((self._z_range, self._y_range, self._x_range, 3))
        z_n = self._z_range //(z_radi + z_margin)
        x_n = self._x_range //(x_radi + x_margin)
        y_n = self._y_range //(y_radi + y_margin)

        # Compute the centers
        centers = []
        for k in range(1, z_n):
            z = k*(z_radi + z_margin)-1
            for j in range(1, y_n):
                y = j*(y_radi + y_margin)-1
                for i in range(1, x_n):
                    x = i*(x_radi + x_margin)-1
                    centers.append([x,y,z])

        # Sample theta
        theta_pos = np.linspace(np.pi/2,  1.5* np.pi, THETA_SAMPLE_SIZE)
        theta_neg = np.linspace(1.5* np.pi,  2.5* np.pi, THETA_SAMPLE_SIZE)

        # Draw the spheres
        for center in centers:
            x_c, y_c, z_c = center
            for z_offset in range(-z_radi, z_radi + 1):
                z = z_c + z_offset
                ro = np.arccos(float(z_offset)/z_radi)

                y = y_radi * np.sin(theta_pos) * np.sin(ro) + y_c
                x_pos = x_radi * np.cos(theta_pos) * np.sin(ro) + x_c
                x_neg = x_radi * np.cos(theta_neg) * np.sin(ro) + x_c
                xx_coord = zip(x_pos, x_neg)
                xy_coord = zip(xx_coord,y)

                for x_pack,y in xy_coord:
                    x_pos = x_pack[0]
                    x_neg = x_pack[1]
                    for x in range(int(x_pos), int(x_neg)):
                        img[z,int(y),x,0] = GRAY_SCALE_MAX
                        img[z,int(y),x,1] = GRAY_SCALE_MAX
                        img[z,int(y),x,2] = GRAY_SCALE_MAX
        if fname:
            save_tif(img, fname)
        return img

def test():
    img_gen = ImageGenerator(1000,1000,100)
    img_gen.make_spherical_image(
        50,
        50,
        10,
        200,
        200,
        10,
        "36_cell_array_img"
    )
if __name__ == "__main__":
    test()
                
