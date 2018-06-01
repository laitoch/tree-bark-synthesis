#!/usr/bin/env python2

import sys
sys.path.append('../../..')
from tree_bark_synthesis import generate_tree_bark

generate_tree_bark(
    ai_filename=sys.argv[1],
    height_map=True,
    k=2,
    pyramid_size=3,
    width_ratio=2,
    height_ratio=8,

    init = 'smart',
    init_size = 64,

    coefs = [1,8,3,3],
    neighborhoods = [5,3],
    max_iter_time = 7,

    rot='vert',
    )
