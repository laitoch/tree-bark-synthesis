class Floodfill:
    """abstract floodfill class, must specify _is_in_region() condition and
    _fill() operation"""
    def __init__(self):
        pass

    def floodfill(self, x, y):
        Q = []
        if not self._is_in_region(x,y):
            return
        Q.append([x,y])
        while Q != []:
            w,z = Q.pop(0)
            e = w
            while self._is_in_region(w-1,z):
                w-=1
            while self._is_in_region(e+1,z):
                e+=1
            for n in range(w, e+1):
                self._fill(n,z)
                if self._is_in_region(n,z+1):
                    Q.append([n,z+1])
                if self._is_in_region(n,z-1):
                    Q.append([n,z-1])

    def _is_in_region(self, x, y):
        pass

    def _fill(self, x, y):
        pass
