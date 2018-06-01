from scipy.misc import imsave
import cv2

def blur(ai, output_file=None):
    """ Create simple FDM.
    Apply Gaussian blur params=(11,11).
    """
    fdm = cv2.GaussianBlur(ai, (11,11), 0)

    if output_file is not None:
        imsave(output_file, fdm)
    return fdm

