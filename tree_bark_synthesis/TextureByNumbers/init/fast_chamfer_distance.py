import numpy as np
from operator import itemgetter
from collections import defaultdict
from collections import Counter

def _index2coord(index):
    return np.array([index // neighborhood_size, index % neighborhood_size])

neighborhood_size = 5
_cache = {}

def chamfer_distance(a,b):
    x = _fast_assymetric_chamfer_distance(a,b)
    y = _fast_assymetric_chamfer_distance(b,a)
    return x + y

def _gen_paths():
    """
    Compute search order for each input pixel position.

    Search from closest to farthest, so that search can be stopped at
    success.
    """
    def path(i,j):
        return (max(abs(_index2coord(i)-_index2coord(j))), j)

    r = range(neighborhood_size**2)
    return [sorted([path(i,j) for j in r], key=itemgetter(0)) for i in r]

_paths = _gen_paths()

def _fast_assymetric_chamfer_distance(a,b):
    key = (tuple(a),tuple(b))

    if key in _cache:
        return _cache[key]

    set_a = set(a)
    result = 0
    for i,x in enumerate(b):
        if x not in set_a:
            result += 2 * neighborhood_size
        else:
            for dist,j in _paths[i]:
                if a[j] == x:
                    result += dist
                    break

    _cache[key] = result
    return result
