#!/usr/bin/env python3

from shape_synthesis import *
from border import *

import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave
from PIL import Image
import cv2
import math
from collections import Counter, defaultdict

def resize(image, ratio):
    new_shape = (int(image.shape[1]*ratio), int(image.shape[0]*ratio))
    type = image.dtype
    return cv2.resize(
            image.astype(float),
            new_shape,
            interpolation=cv2.INTER_NEAREST
            ).astype(type)

def downsample(image, pyramid_size, log):
    """ Downsize a B&W image to half its width and height each subsequent
    layer.
    @param image B&W image to be downsampled into a pyramid.
    @param pyramid_size Count of layers including original image.
    @return [pyramid_layer] from smallest to largest.
    """
    result = []
    for i in range(pyramid_size):
        result.append(image)
        log(image, 'downsample' + str(i))
        image = resize(image, 0.5)
    return list(reversed(result))

def label_map_to_layer_map(label_map_filename):
    """ Arbitrary labels to layers starting from 0 going up.
    @return (layer_count, layer_map)
    """
    result = imread(label_map_filename, mode='L')
    rgb = imread(label_map_filename)
    n = 0
    d = {}
    inverse = {}
    for i,val in np.ndenumerate(result):
        if val not in d:
            d[val] = n
            inverse[n] = rgb[i]
            n += 1
        result[i] = d[val]
    return (n, inverse, result)

def random_init(exemplar, width_ratio=1, height_ratio=1):
    """ Randomly initialize np.array matching the shape and histogram of
    an exemplar np.array. """
    new_shape = (int(exemplar.shape[0]*height_ratio), int(exemplar.shape[1]*width_ratio))

    init = cv2.resize(
        np.copy(exemplar).astype(float),
        new_shape,
        interpolation=cv2.INTER_NEAREST
        ).astype(np.bool)

    init = init.reshape(-1)
    np.random.shuffle(init)
    return init.reshape(new_shape)

def extract_layer(layer_map, i):
    """ @return shape of the ith layer in layer_map """
    return np.vectorize(lambda x : x >= i)(layer_map)

def matching_to_mask(inner_matching, prev_layer, n, gfo, log, i):
    w = weights(inner_matching, prev_layer, prev_layer, n, gfo, False)
    log(w, 'w-' + str(i))
    mask = np.vectorize(lambda x: x > 0)(w)
    return mask

def matching_to_seed(inner_matching, prev_layer, ratio, n, gfo, log, ij):
    """ Initialize layer using these seeded values. """
    w = weights(inner_matching, prev_layer, prev_layer, n, gfo, False)

    log(w, 'w-seed' + ij)

    w_list = sorted(val for _,val in np.ndenumerate(w) if val > 0)
    desired_count = int(len(w_list) * ratio)
    th = w_list[desired_count]

    seed = np.vectorize(lambda x: x <= th)(w)
    log(seed, 'w-seed-bef' + ij)
    seed = np.logical_and(seed, prev_layer)
    log(seed, 'w-seed-aft' + ij)

    return seed

def layer_pixel_ratios(image):
    vals = list(Counter(image.flatten()).values())
    def ith(i):
        return 1.0 * sum(vals[i+1:]) / sum(vals[i:])
    return [ith(i) for i in range(len(vals)-1)]

def create_ordering(exemplar):
    """
    Use heuristic to order the individual layers.

    Necessary because this ordering matters for computing the result. (Is this
    an error in the algorithm or not?)

    1st layer: The layer most present at the image border.
    Subsequent layer: The layer most present next to the previous layer.
    """
    counts = defaultdict(Counter)
    border_counts = Counter()
    for (i,j),val in np.ndenumerate(exemplar):
        counts[val] += Counter(exemplar[i-1:i+2, j-1:j+2].flatten())
        if i == 0 or j == 0 or i == exemplar.shape[0] or j == exemplar.shape[1]:
            border_counts[val] += 1

    for i in counts.keys():
        del counts[i][i]

    result = [border_counts.most_common(1)[0][0]]
    del counts[result[-1]]
    while counts != {}:
        result += [Counter({i:x[result[-1]] for i,x in counts.items()}).most_common(1)[0][0]]
        del counts[result[-1]]

    return dict(enumerate(result))

def control_map(exemplar_filename, pyramid_size, n_size, log, width_ratio, height_ratio, ordering, rot):
    """ Layered shape synthesis - Rosenberger """

    n,gfo = gaussian_falloff_function(n_size)
    layer_count, inverse, exemplar = label_map_to_layer_map(exemplar_filename)

    if layer_count == 1:
        h,w = exemplar.shape
        result = np.empty([int(h*height_ratio), int(w*width_ratio), 3])
        for i in np.ndindex(result.shape[:2]):
            result[i] = inverse.values()[0]
        return result

    if ordering is None:
        ordering = create_ordering(exemplar)

    # ordering = {1:0,0:1,2:2}
    if ordering is not None:
        exemplar = np.vectorize(lambda x : ordering[x])(exemplar)
        tmp = {}
        for key,val in ordering.items():
            tmp[key] = inverse[val]
        inverse = tmp

    exemplars = [np.vectorize(lambda x: x <= i)(exemplar)
                 for i in range(layer_count-2, -1, -1)]
    exemplar_pyramids = [downsample(x, pyramid_size, log) for x in exemplars]

    ratios = layer_pixel_ratios(exemplar)

    for i in range(pyramid_size):
        if i == 0:
            h,w = exemplar_pyramids[0][0].shape
            synth = np.zeros((int(h*height_ratio), int(w*width_ratio)))
            log(synth, 'init')
        else:
            log(synth, 'synth_pre_' + str(i))
            synth = resize(synth, 2)
            log(synth, 'synth_post_' + str(i))

            synth_layers = [extract_layer(synth, k) for k in range(layer_count)]
            synth = np.zeros(synth.shape)

        exemplar_layers = [x[i] for x in exemplar_pyramids]
        for j,exemp_lay in enumerate(exemplar_layers):
            if i == 0:
                if j == 0:
                    layer = random_init(exemp_lay, width_ratio, height_ratio)
                    mask = np.full(layer.shape, True, np.bool)
                    log(layer, 'random')
                else:
                    mask = layer
                    ij = str(i) + '_' + str(j)
                    layer = matching_to_seed(inner_matching, layer, ratios[j], n, gfo, log, ij)
                    log(layer, 'seed' + str(j))
            else:
                if j == 0:
                    mask = np.full(synth_layers[j].shape, True, np.bool)
                if j != 0:
                    mask = layer
                layer = np.logical_and(mask, synth_layers[j+1])


            log(mask, 'mask' + str(i) + '-' + str(j))
            log(exemp_lay, 'exemp' + str(i) + '-' + str(j))
            log(layer, 'layer_before' + str(i) + '-' + str(j))

            if i > 0 or j == 0:
                (layer, inner_matching) = shape_synthesis(exemp_lay, layer, n_size, mask, log, '--'+str(j), rot=rot)
            synth += layer

            log(layer, 'layer_after' + str(i) + '-' + str(j))
            log(synth, 'synth_after' + str(i))

    synth = (layer_count-1) - synth
    result = np.empty(synth.shape + (3,))
    for i,val in np.ndenumerate(synth):
        result[i] = inverse[val]
    return result

def shape_optimization(exemplar_filename,
                       pyramid_size=6,
                       n_size = 13,
                       output_file='./output_p/out.png',
                       log_dir='./output_p/',
                       width_ratio=1,
                       height_ratio=1,
                       ordering=None,
                       rot="90",
                       ):
    def log(image, name):
        if log_dir is not None:
            imsave(log_dir + name + '.png', image.astype(float))

    out = control_map(exemplar_filename, pyramid_size, n_size, log, width_ratio, height_ratio, ordering, rot)
    Image.fromarray(out.astype(np.uint8)).save(output_file)
