from PIL import Image
import cv2

def flip(a, output_file):
    """ Rotate image my 180 degrees. """
    (h, w) = a.shape[:2]
    center = (w/2, h/2)

    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    b = cv2.warpAffine(a, M, (w, h))

    if output_file is not None:
        Image.fromarray(output_file.astype(np.uint8)).save(b)
    return b
