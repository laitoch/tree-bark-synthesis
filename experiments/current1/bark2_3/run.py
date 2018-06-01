#!/usr/bin/env python2

import sys
sys.path.append('../../..')
from tree_bark_synthesis import generate_tree_bark

generate_tree_bark(
    ai_filename=sys.argv[1],
    k=4,
    pyramid_size=4,
    width_ratio=1.3,
    height_ratio=0.8
    )
