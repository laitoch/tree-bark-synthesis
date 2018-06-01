import numpy as np
import scipy

def edge_detector(image):
    """ 255 = edge; 0 = no edge"""

    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered = scipy.ndimage.filters.correlate(image, laplacian)

    for index,value in np.ndenumerate(filtered):
        filtered[index] = 255 if value == 0 else 0

    return filtered
