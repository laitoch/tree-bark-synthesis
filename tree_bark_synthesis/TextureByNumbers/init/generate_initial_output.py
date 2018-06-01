import sys

from scipy.ndimage import imread
from PIL import Image

from edge_detector import *
from chamfer_distance_transformation import *
from initial_output import *

import numpy as np

def _rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def generate_initial_output(ai, a, b, debug_dir=None):
    assert((a is None) == (b is None))

    if a is None:
        a = np.zeros(ai.shape)
        b = np.zeros(ai.shape)

    ida = _rgb2gray(a)
    idb = _rgb2gray(b)

    a_edges = edge_detector(ida)
    b_edges = edge_detector(idb)

    da = chamfer_distance_transformation(a_edges)
    db = chamfer_distance_transformation(b_edges)

    init_b, color_init_b, color_init_b_indices = initial_output(ai, ida, idb, da, db)

    if debug_dir is not None:
        Image.fromarray(a_edges.astype(np.uint8)).save(debug_dir + 'a_edges.png')
        Image.fromarray(b_edges.astype(np.uint8)).save(debug_dir + 'b_edges.png')
        Image.fromarray(da.astype(np.uint8)).save(debug_dir + 'da.png')
        Image.fromarray(db.astype(np.uint8)).save(debug_dir + 'db.png')
        Image.fromarray(init_b.astype(np.uint8)).save(debug_dir + 'init_b.png')
        Image.fromarray(color_init_b.astype(np.uint8)).save(debug_dir + 'color_init_b.png')
        np.save(debug_dir + 'color_init_b_indices.npy', color_init_b_indices)

    return color_init_b, color_init_b_indices
