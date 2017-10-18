import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_plot(func):
    def create_plot(cls, *args, **kwargs):
        show_plot = True if kwargs.get('show_plot') else False
        if show_plot:
            del kwargs['show_plot']
        if kwargs:
            func(cls, *args, **kwargs)
        else:
            func(cls, *args)
        if show_plot:
            if kwargs['show_plot']:
                cls.show_plot()
    return create_plot

class Grapher:
    figure = plt.figure()
    current_subplot = 111

    @classmethod
    @show_plot
    def graph_2D(cls, xs, ys):
        pass


    @classmethod
    @show_plot
    def scatter_3D(cls, xs, ys, zs):
        ax = cls.figure.add_subplot(cls.current_subplot, projection='3d')
        ax.scatter(xs,ys,zs)

    @classmethod
    def show_plot(cls):
        plt.show()

    @classmethod
    def next_subplot(cls, next=None):
        if not next:
            cls.current_subplot += 1
        else:
            cls.current_subplot = next

    @classmethod
    def show_image(cls, img, color='gray'):
        plt.imshow(img, color)


def test():
    pass


if __name__ == "__main__":
    test()

