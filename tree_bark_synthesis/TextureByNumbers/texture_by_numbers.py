import numpy as np
from scipy.ndimage import imread
from PIL import Image

from image_pyramid import *
from optimize import *
from init.generate_initial_output import *
from init.random_init import *

def tbn(ai
      , height_map=None
      , fdm=None
      , a=None
      , b=None
      , output_size=None
      , max_iter_time=4
      , neighborhoods=[7,5,3]
      , wrap='no'
      , init_size=64
      , init='random'
      , init_log_dir=None
      , log_dir='./log/'
      , optimize_log_dir=None
      , save_output=True
    ):
    assert((output_size is None) ^ (b is None))
    assert((a is None) == (b is None))
    assert(init in ('random', 'smart'))

    if height_map is not None:
        assert(ai.shape[:2] == height_map.shape[:2])
        height_map = height_map.reshape(height_map.shape + (1,))
        ai = np.concatenate([ai,height_map], axis=2)

    if fdm is not None:
        assert(ai.shape[:2] == fdm.shape[:2])
    if a is not None:
        assert(a.shape[:2] == ai.shape[:2])

    if output_size is None:
        if b is not None:
            output_size = b.shape[:2]
        else:
            output_size = ai.shape[:2]

    assert(init_size <= output_size[0] and init_size <= output_size[1])

    def log(filename, image):
        if log_dir is not None:
            Image.fromarray(image.astype(np.uint8)).save(log_dir + filename)

    sizes = pyramid_sizes(init_size, ai.shape[:2], output_size)
    _ai, _fdm, _a, _b = tbn_pyramid_level(sizes[0], ai, fdm, a, b)

    if init == 'random':
        color_init_b, init_b = random_init(_ai, sizes[0][1])
    elif init == 'smart':
        color_init_b, init_b = generate_initial_output(_ai,_a,_b,init_log_dir)
    log('init_b.png', color_init_b)

    bi, bi_rgb = optimize(_ai, init_b, _fdm, _a, _b,
            max_iter_time, neighborhoods, wrap, optimize_log_dir)
    old_size = sizes[0]

    for new_size in sizes[1:]:
        log('bi' + str(old_size[1]) + '.png', bi_rgb)
        bi = upsample(bi, old_size, new_size)
        _ai, _fdm, _a, _b = tbn_pyramid_level(new_size, ai, fdm, a, b)
        bi, bi_rgb = optimize(_ai, bi, _fdm, _a, _b,
                max_iter_time, neighborhoods, wrap, optimize_log_dir)
        old_size = new_size

    if save_output:
        Image.fromarray(bi_rgb[:,:,:3].astype(np.uint8)).save('./bi.png')
        if height_map is not None:
            Image.fromarray(bi_rgb[:,:,3].astype(np.uint8)).save('./bhm.png')

    return bi_rgb
