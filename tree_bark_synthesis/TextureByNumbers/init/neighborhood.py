import numpy as np

def _flatten_by_one(x):
    if len(x.shape) <= 1:
        return x
    new_shape = tuple([x.shape[0]*x.shape[1]] + list(x.shape[2:]))
    return x.reshape(new_shape)

def neighborhood_lmr(image, x, y, size):
    radius = (size - 1) // 2
    slice = image[max(0,x-radius):x+radius+1, max(0,y-radius):y+radius+1]
    return _flatten_by_one(slice)

def neighborhood_lm(image, x, y, size):
    slice = neighborhood_lmr(image, x, y, size)

    split = (size**2 - 1) // 2
    return slice[:split+1]

def neighborhood_mr(image, x, y, size):
    slice = neighborhood_lmr(image, x, y, size)

    split = (size**2 - 1) // 2
    return slice[split:]

def neighborhood_init(image, x, y, size):
    x = np.clip(x, 2, image.shape[0]-3)
    y = np.clip(y, 2, image.shape[1]-3)

    radius = (size - 1) // 2
    slice = image[max(0,x-radius):x+radius+1, max(0,y-radius):y+radius+1]
    return slice.flatten()
