import numpy as np

from neighborhood import *

def _surround_with_border(array_2d, size, fill_value):
    row = np.full([size, array_2d.shape[1]+2*size], fill_value)
    col = np.full([array_2d.shape[0], size], fill_value)
    return np.block([[row],[col,array_2d,col],[row]])

def chamfer_distance_transformation(image):
    """ DT3,4 sequential algorithm [Fan citation [20]]"""
    image = _surround_with_border(image, 1, 255)

    coefs = np.array((4,3,4,3,0))
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            image[i,j] = np.min(neighborhood_lm(image,i,j,3) + coefs)

    coefs = np.array((0,3,4,3,4))
    for i in range(image.shape[0]-2, 0, -1):
        for j in range(image.shape[1]-2, 0, -1):
            image[i,j] = np.min(neighborhood_mr(image,i,j,3) + coefs)

    image = image[1:-1, 1:-1]
    return image
