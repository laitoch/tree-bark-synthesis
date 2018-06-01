from scipy.ndimage import imread
from scipy.misc import imsave
from pyflann import *
import numpy as np
import math
import weave
from collections import Counter
import random

from neighbors import *

from init.Floodfill import *

class TBN():
    def __init__(self, aihm, fdm, a, b, max_iter_time, neighborhoods, wrap, log_dir):
        self.max_iter_time = max_iter_time
        self.neighborhoods = neighborhoods
        self.wrap = wrap
        self.log_dir = log_dir

        def np_to_list(x):
            if x is not None:
                if len(np.array(x).shape) == 2:
                    return x.reshape(-1, 1).tolist()
                if len(np.array(x).shape) == 3:
                    return x.reshape(-1, x.shape[-1]).tolist()
            return None

        self.a_shape = aihm.shape[:2]

        self.b = np_to_list(b)

        self.aihm = np_to_list(aihm)
        self.aifdm = self.concat([self.aihm, np_to_list(fdm)])
        self.aaifdm = self.concat([np_to_list(a), self.aifdm])

        self.n_lr_aaifdm = {}
        self.nn = {}
        for n_size in neighborhoods:
            self.n_lr_aaifdm[n_size] = n_lr(self.a_shape, self.aaifdm, n_size, 'no')

            data = np.array(self.n_lr_aaifdm[n_size])
            k = 3
            self.nn[n_size] = FLANN().nn(data, data, k)[0].tolist()

    def optimize(self, init_b):
        self.b_shape = init_b.shape[:2]
        return self.compute(list(init_b.flatten()))

    def concat(self, lists):
        return np.concatenate([x for x in lists if x is not None], axis=1)

    def rgb(self, reflist):
        return [self.aihm[ref] for ref in reflist]

    def color(self, reflist):
        return [self.aifdm[ref] for ref in reflist]

    def candidate_features(self, candidates):
        return [[self.aaifdm[ref] for ref in set] for set in candidates]

    def candidate_feature_neighborhoods(self, candidates, n_size):
        """
        Find coherence candidates for a subset X+ of the image Bi.
        @return List of NaNs and candidate lists.

        NaN -> -1
        """
        shape = self.b_shape
        last = 0
        result = []
        w = 1.0*n_size/4 if n_size >= 4 else 1
        for y in np.ceil(np.linspace(0,shape[1]-w,1.0*shape[1]/w)).astype(int):
            for x in np.ceil(np.linspace(0,shape[0]-w,1.0*shape[0]/w)).astype(int):
                this = x+shape[0]*y
                for _ in range(last+1,this):
                    result.append(-1)
                result.append([self.n_lr_aaifdm[n_size][ref] for ref in candidates[this]])
                last = this
        for _ in range(last+1,len(candidates)):
            result.append(-1)
        return result

    # This is a working simpler version computing all pixels, not just a
    # subset:
    #
    # def candidate_feature_neighborhoods(self, candidates, n_size):
    #     return [[self.n_lr_aaifdm[n_size][ref] for ref in set] for set in candidates]

    def coherence_candidates(self, bi, n_size):
        candidates = []
        for lmr in n_lmr(self.b_shape, bi, n_size, self.wrap):
            r = []
            for y in forward_shift_neighborhood(lmr, self.a_shape, self.wrap):
                r += self.nn[n_size][y] + [y]
            candidates.append(set(r))
        return list(candidates)

    def k_coherence_search(self, candidates, cand_vals, targets, bi=None):
        """ For each set of candidates, choose the one with its value closest
        to its target.
        @return Reference pixel. (the best candidate)
        """
        def l2sq(a,b):
            code =  """
                    int result = 0;
                    for(int i = 0; i < N; i++) {
                        int c = a[i];
                        int d = b[i];
                        if (!(c == -1 || d == -1)) {
                            int x = c-d;
                            result += x*x;
                        }
                    }
                    return_val = result;
                    """
            N = len(a)
            res = weave.inline(code, ['a', 'b', 'N'])
            return res

        return [min([(l2sq(v,target),i) for i,v in zip(indices,values)])[1]
                if isinstance(values, list)
                else -1
                for indices, values, target
                in zip(candidates, cand_vals, targets)]

    def m_step(self, candidates, bi, n_size):
        """ Find the closest neighborhood in the exemplar to every
        neighborhood from a subset of the synthesized image. """
        targets = n_lr(self.b_shape, self.concat([self.b, self.color(bi)]), n_size, self.wrap)

        cand_vals = self.candidate_feature_neighborhoods(candidates, n_size)

        matchL = self.k_coherence_search(candidates, cand_vals, targets)
        return matchL

    def e_step(self, candidates, matchL, bi, n_size):
        """ Create the next iteration of the synthesized image by minimizing
        the energy function.

        bi needed as param for improved version of energy function.
        """
        average_color = lambda x: np.nanmean(np.array(x).reshape([-1, self.aifdm.shape[1]]), axis=0)
        target_colors = [average_color(self.color(forward_shift_neighborhood(lmr, self.a_shape, self.wrap)))
                         for lmr
                         in n_lmr(self.b_shape, matchL, n_size, self.wrap)]
        targets = self.concat([self.b, target_colors])

        cand_vals = self.candidate_features(candidates)

        bi = self.k_coherence_search(candidates, cand_vals, targets, bi)
        return bi

    def image(self, x):
        return np.array(self.rgb(x)).reshape(self.b_shape + (-1,))

    def histogram(self, bi):
        c = Counter(bi)
        return np.array([c[x] for x in range(len(bi))]).reshape(self.a_shape)

    def region_map(self, bi):
        class _Region_map(Floodfill, object):
            def __init__(self, a_shape, bi):
                self.region_number = 1
                self.region_map = np.zeros(a_shape)
                self.bi = bi

            def floodfill(self, x, y):
                try:
                    if self.region_map[x,y] != 0:
                        return False

                    self.region_map[x,y] = self.region_number
                    super(_Region_map, self).floodfill(x, y)
                    self.region_number += 1

                except (IndexError, KeyError):
                    pass

            def _is_in_region(self, x, y):
                try:
                    if self.region_map[x,y] != 0:
                        return False

                    if self.region_map[x-1,y] == self.region_number:
                        if self.bi[x-1,y] + 1 ==  self.bi[x,y]:
                            return True

                    if self.region_map[x+1,y] == self.region_number:
                        if self.bi[x+1,y] - 1 ==  self.bi[x,y]:
                            return True

                    if self.region_map[x,y-1] == self.region_number:
                        if self.bi[x,y-1] + self.a_shape[0] ==  self.bi[x,y]:
                            return True

                    if self.region_map[x,y+1] == self.region_number:
                        if self.bi[x,y+1] - self.a_shape[0] ==  self.bi[x,y]:
                            return True

                    return False

                except (IndexError, KeyError, TypeError):
                    return False

            def _fill(self, x, y):
                self.region_map[x,y] = self.region_number

        ff = _Region_map(self.a_shape, bi)

        s = list(np.ndindex(self.a_shape))
        random.shuffle(s)
        for i,j in s:
            ff.floodfill(i,j)

        return ff.region_map

    def compute(self, bi):
        if self.log_dir is not None:
            imsave(self.log_dir + 'init_b.png', self.image(bi))
        for n_size in self.neighborhoods:
            for i in range(self.max_iter_time):
                candidates = self.coherence_candidates(bi, n_size)
                matchL = self.m_step(candidates, bi, n_size)
                bi = self.e_step(candidates, matchL, bi, n_size)
                self.log_dir = 'log/'
                if self.log_dir is not None:
                    imsave(self.log_dir + 'matchL_'+str(n_size)+str(i)+'.png', self.image(matchL))
                    imsave(self.log_dir + 'bi'+str(n_size)+str(i)+'.png', self.image(bi))

                    histogram = self.histogram(bi)
                    imsave(self.log_dir + 'histogram'+str(n_size)+str(i)+'.png', histogram)

                    imsave(self.log_dir + 'regionmap'+str(n_size)+str(i)+'.png', self.region_map(histogram))
        return (bi, self.image(bi))

def optimize(ai, init_b, fdm=None, a=None, b=None, max_iter_time=5,
        neighborhoods=[7,5,3], wrap='no', log_dir=None):
    assert type(ai) == np.ndarray and ai.dtype == 'float'
    assert len(ai.shape) == 3 and 3 <= ai.shape[2] <= 4

    assert type(init_b) == np.ndarray and init_b.dtype == 'int64'
    assert len(init_b.shape) == 2

    if a is None or b is None:
        assert a == b == None
    else:
        assert type(a) == np.ndarray
        assert a.dtype == 'float'
        assert a.shape[:2] == ai.shape[:2]

        assert type(b) == np.ndarray
        assert b.dtype == 'float'

    if not (fdm is None):
        assert type(fdm) == np.ndarray
        assert fdm.dtype == 'float'
        assert fdm.shape[:2] == ai.shape[:2]

    assert(wrap in ['yes', 'horizontal', 'vertical', 'no'])

    tbn = TBN(ai, fdm, a, b, max_iter_time, neighborhoods, wrap, log_dir)
    return tbn.optimize(init_b)
