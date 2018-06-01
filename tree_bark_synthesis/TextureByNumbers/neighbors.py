import numpy as np
import math
from functools import partial
import weave

def _neighborhood(n_type, nan, shape, list, n_size, wrap):
    radius = (n_size - 1) // 2
    image = _wrap(shape, nan, list, radius, wrap)
    feature_count = len(list) // (shape[0]*shape[1])
    assert(feature_count*shape[0]*shape[1] ==  len(list))

    return [n_type(image, (i,j), radius, feature_count)
            for i in range(radius, image.shape[0]-radius)
            for j in range(radius, image.shape[1]-radius)]

def _flat_lr(image, index, radius, feature_count):
    slice = _neighborhood_flat(image, index, radius)
    beg = list(slice[:(len(slice)-feature_count)/2])
    end = list(slice[(len(slice)-feature_count)/2+feature_count:])
    return np.array(beg+end).astype(float)

def _flat_lmr(image, index, radius, feature_count):
    slice = _neighborhood_flat(image, index, radius)
    return slice.astype(float)

def _neighborhood_flat(image, index, radius):
    x,y = index
    slice = image[x-radius:x+radius+1, y-radius:y+radius+1]
    return slice.flatten()

def _wrap(shape, nan, list, radius, wrap):
    assert(radius <= min(shape[0], shape[1]))

    image = np.array(list).reshape(shape + (-1,))

    corner = np.full((radius,radius,image.shape[2]), nan)
    vert = np.full((image.shape[0],radius,image.shape[2]), nan)
    horiz = np.full((radius,image.shape[1],image.shape[2]), nan)

    if wrap == 'yes':
        return np.block([
            [[image[-radius:,-radius:]], [image[-radius:]], [image[-radius:,:radius]] ],
            [[image[:,-radius:]],        [image],           [image[:,:radius]]        ],
            [[image[:radius,-radius:]],  [image[:radius]],  [image[:radius ,:radius]] ]
            ])
    elif wrap == 'vertical':
        return np.block([
            [[corner], [image[-radius:]], [corner] ],
            [[vert]  , [image],           [vert]   ],
            [[corner], [image[:radius]],  [corner] ]
            ])
    elif wrap == 'horizontal':
        return np.block([
            [[corner],            [horiz], [corner]           ],
            [[image[:,-radius:]], [image], [image[:,:radius]] ],
            [[corner],            [horiz], [corner]           ]
            ])
    elif wrap == 'no':
        return np.block([
            [[corner], [horiz], [corner] ],
            [[vert],   [image], [vert]   ],
            [[corner], [horiz], [corner] ]
            ])
    elif wrap == 'impossible':
        return image
    else:
        assert(False)

def forward_shift_neighborhood(lmr, shape, wrap):
    """
    For each referenced pixel in the input neighborhood, compute an
    appropriately forward shifted reference and return in the list.

    @param lmr Flat lmr neighborhood in image of references.
    @param shape Shape (x,y) of the referenced image.
    @param wrap Wrap option - one of:
        ['yes', 'horizontal', 'vertical', 'no'].
    @return List of forward shifted references.
        Includes multiples. Excludes NaNs.

    Example visualization:
        lmr: abc  referenced image: 12345  for i->9, return 3
             def                    67890  for a->1, return 7
             ghi                    uvwxy
        for f->6 and no wrap, should be NaN, but add nothing to result
        for f->6 and wrap, return 0
    """
    code = """
           int index = -1;
           for (int j = -radius; j <= radius; j++) {
               for (int i = -radius; i <= radius; i++) {
                   index++;
                   int shift_x, shift_y;
                   int x = lmr[index];
                   if (x == -1)
                       continue;

                   int d_x;
                   if (x-i < 0)
                       d_x = 1;
                   else
                       d_x = x / width - (x-i) / width;

                   if (d_x == 0)
                       shift_x = -i;
                   else if (horizontal == 1)
                       shift_x = -i + d_x * width;
                   else
                       continue;

                   int s = width * height;
                   int d_y;
                   if (x-j*width >= s)
                       d_y = -1;
                   else if (x-j*width < 0)
                       d_y = 1;
                   else
                       d_y = 0;

                   if (d_y == 0)
                       shift_y = -j*width;
                   else if (vertical == 1)
                       shift_y = -j*width + d_y * s;
                   else
                       continue;

                   result.append(x + shift_x + shift_y);
               }
           }
           """
    result = []
    size = int(math.sqrt(len(lmr)))
    radius = size // 2
    width = shape[1]
    height = shape[0]
    horizontal = 1 if wrap == 'yes' or wrap == 'horizontal' else 0
    vertical = 1 if wrap == 'yes' or wrap == 'vertical' else 0
    weave.inline(code, ['lmr', 'radius', 'width', 'height', 'horizontal',
        'vertical', 'result'])
    return result

n_lr = _foo = partial(_neighborhood, _flat_lr, -1)
n_lmr = partial(_neighborhood, _flat_lmr, -1)
