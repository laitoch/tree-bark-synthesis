import sys
import regex as re
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
from scipy.ndimage import imread
from scipy.misc import imsave
import scipy
import numpy_indexed as npi

def color_region_label(ai, k, log_dir=None, output_file=None, nolonely_file=None):
    """ Create label image a from an input image ai.
    Segments image into large regions of similar colors, grayscales them and
    adds a separate region at all edges 3 pixels wide.

    @param k Number of regions to be created (excluding edge region).
    """
    denoised = denoise(ai)
    quant = quantize(denoised, k)
    no_lonely = erode_tiny_regions(quant)
    edges = add_eges_grayscale(no_lonely)
    big_edges = increase_edge_size(edges)
    a = big_edges

    if log_dir is not None:
        imsave(log_dir + 'denoise.png', denoised)
        imsave(log_dir + 'quantize.png', quant)
        imsave(log_dir + 'quant_no_lonely.png', no_lonely)
        imsave(log_dir + 'edges.png', edges)
        imsave(log_dir + 'big_edges.png', big_edges)

    if nolonely_file is not None:
        imsave(nolonely_file, no_lonely)
    if output_file is not None:
        imsave(output_file, a)
    return a

def denoise(image):
    """ Denoise.
    Remove noise (small local perturbations) in the image.
    """
    return cv2.fastNlMeansDenoisingColored(image,None,6,6,7,21)

def quantize(image, k):
    """ Quantize.
    Reduces the number of colors to k.
    Uses k-means in L*a*b* color space.
    """
    (h, w, z) = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image = image.reshape((h*w, -1))
    clt = MiniBatchKMeans(n_clusters = k)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, z))
    return cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)

def erode_tiny_regions(image):
    """ Erode. Remove tiny regions.
    Each pixel becomes the mode of its most common neighbor.
    """
    no_lonely = image.copy()
    for x,y in np.ndindex(image.shape[:2]):
        n = image[max(0,x-1):x+2, max(0,y-1):y+2]
        no_lonely[x,y] = npi.mode(n.reshape(-1,3))
    return no_lonely

def add_eges_grayscale(image):
    """ Edge detect.
    Keep original image grayscale value where no edge.
    """
    greyscale = rgb2gray(image)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    edges = scipy.ndimage.filters.correlate(greyscale, laplacian)
    for index,value in np.ndenumerate(edges):
        edges[index] = 255-greyscale[index] if value == 0 else 0
    return edges

def increase_edge_size(image):
    """ Increase border size.
    Each pixel next to an edge becomes an edge.
    """
    big_edges = image.copy()
    for x,y in np.ndindex(image.shape[:2]):
        if np.min(image[max(0,x-1):x+2, max(0,y-1):y+2]) == 0:
            big_edges[x,y] = 0
    return big_edges

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
