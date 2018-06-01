#!/usr/bin/env python2

import sys
sys.path.append('../../..')
from tree_bark_synthesis import generate_tree_bark

generate_tree_bark(
    ai_filename='ai.png',
    height_map=False,
    k=1,
    pyramid_size=3,
    width_ratio=2,
    height_ratio=6,

    init = 'smart',
    init_size = 16,

    coefs = [1,4,1,1],
    neighborhoods = [11,7,5,3],
    max_iter_time = 4,

    rot='vert',
    )
