from scipy.ndimage import imread
import numpy as np
import math
import cv2

def pyramid_sizes(init_size, a_output_size, b_output_size):
    """
    Determine appropriate resolutions for all levels of an image pyramid.

    @param init_size Max width/height of the lowest resolution of the pyramid.
    @param a_output_size Size (w,h) of the exemplar image (ai) of the highest
    resolution of the pyramid.
    @param b_output_size Size (w,h) of the synthesized image (bi) of the
    highest resolution of the pyramid.
    @return Sizes of all pyramid levels from lowest to highest resolution.

    Examples:
        >>> pyramid_sizes(32, (158,239), (345,287))
        [((21, 32), (45, 38)), ((42, 64), (90, 76)), ((84, 128),
         (180, 152)), ((158, 239), (345, 287))]
    """
    if a_output_size[0] < a_output_size[1]:
        y = init_size
        x = int(math.floor(init_size * a_output_size[0] / a_output_size[1]))
    else:
        x = init_size
        y = int(math.floor(init_size * a_output_size[1] / a_output_size[0]))

    a = int(math.ceil(1.0 * b_output_size[0] * x / a_output_size[0]))
    b = int(math.ceil(1.0 * b_output_size[1] * y / a_output_size[1]))
    result = []

    while x < a_output_size[0] and y < a_output_size[1]:
        result.append(((x,y), (a,b)))
        x *= 2
        y *= 2
        a *= 2
        b *= 2

    assert(a >= b_output_size[0])
    assert(b >= b_output_size[1])
    assert(a//2 < b_output_size[0])
    assert(b//2 < b_output_size[1])

    result.append((a_output_size, b_output_size))
    return result

def tbn_pyramid_level(sizes, ai, fdm, a, b):
    """
    Downsize an image to appropriate sizes.

    A discrete interpolation method is used for images with discrete vs
    continuous values.
    """
    result = []
    result.append(downsize(ai, sizes[0]))

    if fdm is not None:
        result.append(downsize(fdm, sizes[0]))
    else:
        result.append(None)

    if a is not None:
        result.append(downsize(a, sizes[0]))
        result.append(downsize(b, sizes[1]))
    else:
        result.append(None)
        result.append(None)

    return result

def upsample(bi, old_size, new_size):
    x = resize_reference(bi, old_size[0], new_size[0])
    return enlarge(x.reshape(old_size[1]), new_size[1], False, new_size[0][1])

def enlarge(image, size, nan=False, width=None):
    """
    Make an matrix bigger by evenly adding rows and columns of NaN.
    Always add last row and last column.

    @param image 2D np.array to enlarge.
    @param x Desired width, must be >= image width.
    @param y Desired height, must be >= image height.
    @param nan If True: new array elements are np.nan
               If False: new columns are previous column value plus 1
                         new rows are previous row value plus x

    Examples:
        >>> enlarge(np.array([[0, 1, 2], [3, 4, 5]]), (4,4), True)
        array([[  0.,   1.,   2.,  nan],
            [ nan,  nan,  nan,  nan],
            [  3.,   4.,   5.,  nan],
            [ nan,  nan,  nan,  nan]])
        enlarge(np.array([[0, 1, 2], [3, 4, 5]]), (4,4), 4)
        array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 3,  4,  5,  6],
            [ 7,  8,  9, 10]])
    """
    x, y = size
    oldx, oldy = image.shape[:2]

    assert(x > oldx)
    assert(y > oldy)

    rows = np.floor(np.linspace(0,oldx-0.5,x)).astype(int)
    rows = np.insert(rows[1:]!=rows[:-1], 0, True)

    cols = np.floor(np.linspace(0,oldy-0.5,y)).astype(int)
    cols = np.insert(cols[1:]!=cols[:-1], 0, True)

    def swap_first_False_with_last_element(arr):
        if arr[-1] != False:
            arr[-1] = False
            for i in range(len(arr)):
                if arr[i] == False:
                    arr[i] = True
                    return
    swap_first_False_with_last_element(rows)
    swap_first_False_with_last_element(cols)

    if nan:
        new = np.full((x,y), np.nan)
        new[np.ix_(rows,cols)] = image
    else:
        new = np.zeros((x,y)).astype(int)
        new[np.ix_(rows,cols)] = image

        cols = np.where(cols == False)[0]
        new[:,cols] = new[:,cols-1] + 1

        rows = np.where(rows == False)[0]
        new[rows] = new[rows-1] + width

    return new

def downsize(image,size):
    if size[0]*2 < image.shape[0]:
        image = downsize(image, (size[0]*2,size[1]*2))

    x, y = size
    oldx, oldy = image.shape[:2]

    if x == oldx and y == oldy:
        return image

    assert(x <= oldx)
    assert(y <= oldy)

    rows = np.floor(np.linspace(0,x-0.5,oldx)).astype(int)
    rows = np.insert(rows[1:]!=rows[:-1], 0, True)

    cols = np.floor(np.linspace(0,y-0.5,oldy)).astype(int)
    cols = np.insert(cols[1:]!=cols[:-1], 0, True)

    def swap_first_False_with_last_element(arr):
        if arr[-1] != False:
            arr[-1] = False
            for i in range(len(arr)):
                if arr[i] == False:
                    arr[i] = True
                    return
    swap_first_False_with_last_element(rows)
    swap_first_False_with_last_element(cols)

    rows = (rows == False).nonzero()
    cols = (cols == False).nonzero()

    image = np.delete(image, cols, axis=1)
    image = np.delete(image, rows, axis=0)

    return image

def resize_reference(ref, old_size, new_size):
    """
    When enlarging an image that is being pointed to by a reference image, the
    reference image breaks. This function repairs that problem.

    @param ref List or numpy array of any shape.
    """
    oldx, oldy = old_size
    old = np.arange(oldx*oldy).reshape((oldx,oldy))
    large = enlarge(old, new_size, True)

    key = {}
    for x,y in enumerate(large.reshape(-1).tolist()):
        if not np.isnan(y):
            key[int(y)] = x

    return np.vectorize(lambda t: key[t])(ref)
