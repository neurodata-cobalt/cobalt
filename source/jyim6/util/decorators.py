from helper import save_tif, write_csv

def save_img_and_centers(func):
    def gen_image_and_centers(self, *args, **kwargs):
        fname = kwargs.get('fname')
        img = None
        if fname:
            del kwargs['fname']
        if kwargs:
            img, centers = func(self, *args, **kwargs)
        else:
            img, centers = func(self, *args)
        if fname:
            save_tif(img, str(fname))
            write_csv(centers, fname+'_centers')
        return img, centers
    return gen_image_and_centers

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
            save_tif(img, str(fname))
        return img
    return gen_image