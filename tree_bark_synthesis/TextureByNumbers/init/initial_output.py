import numpy as np
import random

from Floodfill import *
from neighborhood import neighborhood_init
from fast_chamfer_distance import *

class _RegionGrowthFloodfill(Floodfill, object):
    """floodfill implementation performing region growth"""

    def __init__(self, ai, ida, idb, da, db):
        Floodfill.__init__(self)

        self.ai = ai
        self.ida = ida
        self.idb = idb
        self.da = da
        self.db = db

        self.shape = db.shape

        self.region_number = 1

        self.init_b = np.full(self.shape, 0)
        self.color_init_b = np.full(self.shape+(ai.shape[-1],), 0)
        self.color_init_b_indices = np.full(self.shape, 0)

        self._cache = {}

        def init_neighborhood(image):
            return {(x,y): neighborhood_init(image, x, y, neighborhood_size)
                    for (x,y),_
                    in np.ndenumerate(image)}

        self._da_neighborhoods = init_neighborhood(self.da)
        self._db_neighborhoods = init_neighborhood(self.db)
        self._ida_neighborhoods = init_neighborhood(self.ida)
        self._idb_neighborhoods = init_neighborhood(self.idb)

    def floodfill(self, x, y):
        try:
            if self.init_b[x,y] != 0:
                return False

            self.pbx = x
            self.pby = y
            self.__find_pa()
            super(_RegionGrowthFloodfill, self).floodfill(x, y)
            self.region_number += 1

        except (IndexError, KeyError):
            pass

    def _is_in_region(self, x, y):
        try:
            if self.init_b[x,y] != 0:
                return False

            dist = self.__region_growth_distance(self._ai_coords(x,y), (x, y))
            return dist < self.threshold

        except (IndexError, KeyError, TypeError):
            return False

    def _fill(self,x,y):
        self.init_b[x,y] = self.region_number
        self.color_init_b[x,y] = self.ai[self._ai_coords(x,y)]

        aix, aiy = self._ai_coords(x,y)
        index = self.ai.shape[1] * aix + aiy
        self.color_init_b_indices[x,y] = index

    def __find_pa(self):
        self.pax, self.pay = np.inf, None
        min_distance = np.inf

        for (x,y),_ in np.ndenumerate(self.da):
            chamfer_a = self._ida_neighborhoods[x,y]
            chamfer_b = self._idb_neighborhoods[self.pbx,self.pby]
            if tuple(chamfer_a) == tuple(chamfer_b):
                distance = self.__region_growth_distance((x, y), (self.pbx, self.pby))
                if distance < min_distance:
                    self.pax, self.pay, min_distance = x, y, distance

        if min_distance > 90:
            for (x,y),_ in np.ndenumerate(self.da):
                distance = self.__region_growth_distance((x, y), (self.pbx, self.pby))
                if distance < min_distance:
                    self.pax, self.pay, min_distance = x, y, distance
                    # hack to dramatically decrease runtime
                    # if min_distance < 1500:
                    #     break

        self.threshold = (min_distance + 5) * 1.25

    def __region_growth_distance(self, a, b):
        pax, pay = a
        pbx, pby = b
        if (pax,pay,pbx,pby) in self._cache:
            return self._cache[(pax,pay,pbx,pby)]

        w = 90
        chamfer = chamfer_distance(self._ida_neighborhoods[pax,pay], self._idb_neighborhoods[pbx,pby])
        l2_norm = np.linalg.norm(self._da_neighborhoods[pax,pay] - self._db_neighborhoods[pbx,pby])

        result = w*chamfer + l2_norm
        self._cache[(pax,pay,pbx,pby)] = result
        return result

    def _ai_coords(self, x, y):
        aix = self.pax + (x-self.pbx)
        aiy = self.pay + (y-self.pby)

        self.ai[aix,aiy] # raises IndexError if necessary
        return (aix,aiy)

def initial_output(ai, ida, idb, da, db):
    ff = _RegionGrowthFloodfill(ai, ida, idb, da, db)

    s = list(np.ndindex(db.shape))
    random.shuffle(s)
    for i,j in s:
        ff.floodfill(i,j)

    return (ff.init_b, ff.color_init_b, ff.color_init_b_indices)
