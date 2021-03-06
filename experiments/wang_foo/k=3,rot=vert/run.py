#!/usr/bin/env python2

import sys
sys.path.append('../../..')
from tree_bark_synthesis import generate_tree_bark

generate_tree_bark(
    ai_filename=sys.argv[1],
    k=3,
    pyramid_size=3,
    width_ratio=2,
    height_ratio=8,
    rot="vert",
    )
