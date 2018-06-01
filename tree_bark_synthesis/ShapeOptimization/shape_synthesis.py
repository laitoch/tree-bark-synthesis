#!/usr/bin/env python3

from scipy.ndimage import imread
from scipy.misc import imsave
from collections import Counter
import random
import math
import numpy as np
from pyflann import *
import weave

def is_boundary_pixel(shape, index, n_size):
    """ Pixel is True and at least one 4-neighbor is False. """
    code = """
            return_val = false;
            if (x > r && y > r && x < nx-r && y < ny-r)
                if (shape(x,y)) {
                    if (x-1 >= 0 && !shape(x-1,y))
                        return_val = true;
                    else if (y-1 >= 0 && !shape(x,y-1))
                        return_val = true;
                    else if (x+1 < nx && !shape(x+1,y))
                        return_val = true;
                    else if (y+1 < ny && !shape(x,y+1))
                        return_val = true;
                }
           """
    nx,ny = shape.shape
    x,y = index
    r = n_size // 2
    return weave.inline(code, ['nx','ny','x','y','shape','r'],
            type_converters=weave.converters.blitz)

def is_close_to_boundary(shape, index):
    """ Pixel is X and at least one 8-neighbor is not X. """
    code = """
            return_val = false;
            if (x-1 >= 0 && shape(x,y) != shape(x-1,y))
                return_val = true;
            else if (y-1 >= 0 && shape(x,y) != shape(x,y-1))
                return_val = true;
            else if (x+1 < nx && shape(x,y) != shape(x+1,y))
                return_val = true;
            else if (y+1 < ny && shape(x,y) != shape(x,y+1))
                return_val = true;
            else if (x-1 >= 0 && y-1 >= 0 && shape(x,y) != shape(x-1,y-1))
                return_val = true;
            else if (x+1 < nx && y+1 < ny && shape(x,y) != shape(x+1,y+1))
                return_val = true;
            else if (x+1 < nx && y-1 >= 0 && shape(x,y) != shape(x+1,y-1))
                return_val = true;
            else if (x-1 >= 0 && y+1 < ny && shape(x,y) != shape(x-1,y+1))
                return_val = true;
           """
    nx,ny = shape.shape
    x,y = index
    return weave.inline(code, ['nx','ny','x','y','shape'],
            type_converters=weave.converters.blitz)

def boundary(shape, n_size, n):
    """ Shape boundaries & their neighborhoods
    @param shape 2D_bool_numpy_array: True if pixel in shape
    @return {index: neighborhood}
        index: 2D_int_tuple = index of neighborhood center in shape
        neighborhood: 2D_bool_numpy_array of size n_size

    Boundaries are shape pixels inside the shape having 1 or more 4-neighbors
    outside the shape.
    """
    return {i: shape[n(i)]
            for i in np.ndindex(shape.shape)
            if is_boundary_pixel(shape,i,n_size)}

def exemplar_patch_counts(exemplar_boundary, synthesized_boundary):
    """ Determine counts for each e.
    @return [[2D_e_index, count]]

    Assuming: |Bs| = Q|Be| + R.
    We include each patch from Be Q times, and randomly select R additional
    patches from Be.

    In this way we ensure that all the boundary features in the exemplar
    shape get an equal chance to be represented in the synthesized shape.

    """
    quotient = len(synthesized_boundary) // len(exemplar_boundary)
    remainder = len(synthesized_boundary) % len(exemplar_boundary)
    patches = set(exemplar_boundary.keys())
    counter =  Counter({x:quotient for x in patches}) \
             + Counter(random.sample(patches, remainder))
    return [[x,y] for x,y in counter.items()]

def boundary_patch_matching(exemplar_boundary, synthesized_boundary, rot):
    """ Matching boundary patches
    Algorithm:
        Foreach e: Compute n = number of s's a given e should be assigned to.
        Repeat on unassigned e's and s's:
            Assign each e to its nearest s n times.
            Each s keeps only 1 nearest e.

    Each patch in Be is initially assigned to its nearest neighbor in Bs.

    As a result, some patches in Bs may have more than one exemplar patch
    assigned to them, while others may have none.

    In the former case, we keep only the assignment with the smallest L2
    difference, and discard the rest.

    All of the pairs of patches which have been assigned are then removed from
    further consideration, and the process is repeated until every patch in Bs
    has been assigned.

    @return {2D_s_index: (rotated_e_neighborhood, distance)}
    """
    # exemplar_boundary = {2D_e_index -> e_neighborhood}
    # synthesized_boundary = {2D_s_index -> s_neighborhood}
    # patch_counts = [flann_e_index -> [2D_e_index, count_to_be_matched]]
    # s = [flann_s_index -> 2D_s_index]
    # flann_result = ([flann_e_index -> flann_s_index],[flann_e_index -> distance])
    # tmp = {2D_s_index -> (rotated_e_neighborhood, distance)}

    s = list(synthesized_boundary.keys())
    patch_counts = exemplar_patch_counts(exemplar_boundary, synthesized_boundary)
    matching = {}
    while s != []:
        # Find 1-nearest_neighbors
        r0 = [synthesized_boundary[x].flatten() for x in s]
        if rot == "no":
            r = zip(r0)
            k_rot = lambda x: 0
            r_num = 1
        else:
            r2 = [np.rot90(synthesized_boundary[x],2).flatten() for x in s]
            if rot == "vert":
                r = zip(r0,r2)
                k_rot = lambda x: (x%2)*2
                r_num = 2
            else:
                assert(rot == "yes")
                r1 = [np.rot90(synthesized_boundary[x],1).flatten() for x in s]
                r3 = [np.rot90(synthesized_boundary[x],3).flatten() for x in s]
                r = zip(r0,r1,r2,r3)
                k_rot = lambda x: 4-(x%4)
                r_num = 4
        dataset = np.array(sum(r,())).astype(float)

        testset = np.array([exemplar_boundary[x].flatten()
                            for x,_
                            in patch_counts]).astype(float)

        flann_result = FLANN().nn(dataset, testset, 1)

        # 1_e:1_s -> 1_s:n_e -> 1_s:first_e
        tmp = {}
        for flann_e,(flann_s,distance) in enumerate(zip(*flann_result)):
            si = s[flann_s/r_num]
            if si not in tmp or tmp[si][1] > distance:
                ei = patch_counts[flann_e][0]
                rot_e = np.rot90(exemplar_boundary[ei], k=k_rot(flann_s))
                tmp[si] = (rot_e, distance, flann_e)

        # Remove assigned patches from consideration
        s = set(s)
        for si,(rot_e,distance,flann_e) in tmp.items():
            matching[si] = (rot_e, distance)
            s.remove(si)
            patch_counts[flann_e][1] -= 1
        s = list(s)

        patch_counts = [[x,y] for x,y in patch_counts if y != 0]

    return matching



def weights(boundary_patch_matching, synthesized, mask, n, falloff, outer=True):
    """ Create weights for pixels near boundaries of the synthesized image that
    determine a pixel change during this iteration. """
    a = np.zeros(synthesized.shape)
    b = np.zeros(synthesized.shape)

    for si,(rot_e,distance) in boundary_patch_matching.items():
        x = np.vectorize(lambda x : 1 if x else -1)
        a[n(si)] += x(rot_e) * falloff * (1/(1+distance))
        b[n(si)] += falloff * (1/(1+distance))

    result = np.vectorize(lambda a,b: a if b < 1 else a/b)(a,b)

    for i in np.ndindex(result.shape):
        # apply layer mask
        if not mask[i]:
            result[i] = 0

        if outer:
            weight = result[i]
            value = synthesized[i]
            if (value == True and weight > 0) or (value == False and weight < 0):
                result[i] = 0 # Weight value indicates no change

            # Only change pixels with at least one different 8-neighbor
            if not is_close_to_boundary(synthesized, i):
                result[i] = 0

    return result

def added_pixel_count(weights, threshold):
    """ Number of pixels to be added to shape when using given weights and
    threshold. (Negative when removing pixels.)"""
    code = """
            int result = 0;
            for (int i=0; i<nx; i++)
                for (int j=0; j<ny; j++)
                    if (std::abs(weights(i,j)) > (double)threshold)
                        result += (weights(i,j) > 0) - (weights(i,j) < 0);
            return_val = result;
           """
    nx,ny = weights.shape
    return weave.inline(code, ['nx','ny','threshold','weights'],
            type_converters=weave.converters.blitz)

def threshold(weights, delta_size):
    """ Sample for threshold minimizing pixel changes. """
    return min((abs(added_pixel_count(weights,i) + delta_size), i)
                for i in np.linspace(10**-2,10**-7,40))[1]

def adjust_shape(synthesized, weights, threshold):
    """ Create new shape by changing pixels when weight is above the
    threshold. """
    x = np.vectorize(lambda x,y : y if abs(x) < threshold else not y)
    return x(weights,synthesized)

def gaussian_falloff_function(n_size):
    def n(center):
        """ Neighborhood slice()
        @param center 2-tuple of coords of neighborhood center
        """
        x,y = center
        r = n_size//2
        return (slice(x-r,x+r+1), slice(y-r,y+r+1))

    sigma = n_size / 2
    x, y = np.mgrid[n((0,0))]
    return (n, np.exp(-((x**2 + y**2)/(2.0*sigma**2))))

def shape_synthesis(exemplar, synthesized, n_size, mask, log, s="", rot="yes"):
    """ Perform shape synthesis at one pyramid level. """
    n, falloff = gaussian_falloff_function(n_size)

    x,y = synthesized.shape

    log(exemplar, 'e' + str(x) + s)
    log(synthesized, 's' + str(x) + s)

    exemplar_boundary = boundary(exemplar, n_size, n)
    delta_size = 0
    stop_magic_number = 1 + math.ceil(1.0*x*y / 10000 * 0.7)

    for i in range(8):
        synthesised_boundary = boundary(synthesized, n_size, n)
        matching = boundary_patch_matching(exemplar_boundary, synthesised_boundary, rot)
        w = weights(matching, synthesized, mask, n, falloff)
        th = threshold(w,delta_size)
        added_pixels = added_pixel_count(w,th)
        synthesized = adjust_shape(synthesized, w, th)

        log(w, 'w' + str(x) + "_" + str(i) + s)
        log(synthesized, 'out' + str(x) + "_" + str(i) + s)

        delta_size += added_pixels

    return (synthesized, matching)
