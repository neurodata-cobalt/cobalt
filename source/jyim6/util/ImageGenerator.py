import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class ImageGenerator:
    def __init__(self, x_range, y_range, z_range):
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range

    def make_image_spherical(
                            self,
                            radius, 
                            fade,
                            cell_count,
                            sheer = 0,
                            rotation = 0,
                            center = None,
                            ):
        img = np.zeros((self._x_range, self._y_range, self._z_range))

        # centers weren't specificed
        # so they will be assigned based on other parameters
        if not center:
            max_cells = self._x_range//(2 * radius) * \
                        self._y_range//(2 * radius) * \
                        self._z_range//(2 * radius)
            if cell_count > max_cells:
                print("WARNING: number of cells in image exceeds space.", file=sys.stderr)

            # Amount of cells that can fit in each dim
            z_cells = self._z_range//(2 * radius) # TODO: better formula for determining how many cells in z?
            xy_cells = cell_count // z_cells
            yx_ratio = self._y_range/self._x_range
            x_cells = int((1-yx_ratio)*xy_cells)
            y_cells = int(yx_ratio*xy_cells)
            centers = []

            for k in range(1, z_cells*2, 2):
                for i in range(1, x_cells*2, 2):
                    for j in range(1, y_cells*2, 2):
                        centers.append([i*r, j*r, k*r])
            print(centers)
        return centers

def test():
    pass
if __name__ == "__main__":
    test()
                
