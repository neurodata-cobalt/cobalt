from helper import save_tif

def save_img(func):
    def gen_image(self, *args, **kwargs):
        fname = str(kwargs.get('fname'))
        img = None
        # import pdb; pdb.set_trace()
        if fname:
            del kwargs['fname']
        if kwargs:
            img, centers = func(self, *args, **kwargs)
        else:
            img, centers = func(self, *args)
        if fname:
            save_tif(img, fname)
        return img, centers
    return gen_image