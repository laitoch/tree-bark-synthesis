import numpy as np

def add_border(array_2d, size, fill_value):
    row = np.full([size, array_2d.shape[1]+2*size], fill_value)
    col = np.full([array_2d.shape[0], size], fill_value)
    return np.block([[row],[col,array_2d,col],[row]])

def trim_border(image, trim_val=False):
    """ Remove empty outer rows and columns.
    Remove multiple rows from top and multiple columns from left and right
    until a non-empty row/column is found.
    """
    def first_nonzero(arr, axis, invalid_val=float('inf')):
        mask = arr!=trim_val
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

    def last_nonzero(arr, axis, invalid_val=float('-inf')):
        mask = arr!=trim_val
        val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
        return np.where(mask.any(axis=axis), val, invalid_val)

    x1 = int(min(first_nonzero(image, 0)))
    x2 = int(max(last_nonzero(image, 0)))
    y1 = int(min(first_nonzero(image, 1)))
    y2 = int(max(last_nonzero(image, 1)))

    return image[x1:x2+1,y1:y2+1]
