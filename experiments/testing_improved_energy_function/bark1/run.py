#!/usr/bin/env python2

import sys
sys.path.append('../../..')
from tree_bark_synthesis import generate_tree_bark

generate_tree_bark(
    ai_filename=sys.argv[1],
    k=3,
    pyramid_size=4,
    width_ratio=1,
    height_ratio=1,

    tbn_only=True
    )
