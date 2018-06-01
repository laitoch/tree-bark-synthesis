#!/usr/bin/env python2

from scipy.ndimage import imread

import numpy as np
from scipy.misc import imsave

import sys
import shutil
sys.path.append('.')
from TextureByNumbers import tbn
from LabelGeneration import color_region_label, blur, add_eges_grayscale, increase_edge_size
from ShapeOptimization import shape_optimization

def generate_tree_bark(
        ai_filename,
        height_map = False,
        k = 3,
        pyramid_size = 4,
        n_size = 13,
        width_ratio = 1,
        height_ratio = 1,
        rot = "yes",
        use_big_edges = False,

        # texture-by-numbers
        coefs = [1,8,3,3],
        init_size = 16,
        neighborhoods = [5,3],
        max_iter_time = 7,
        wrap = 'no',
        init = 'random',
        init_log_dir = None,

        tbn_only = False,
    ):
    """
    Run the whole bark generation pipeline.
    """
    ai = imread(ai_filename, mode='RGB')

    if not tbn_only:
        blur(ai, 'fdm.png')
        color_region_label(ai, k, None, 'a.png', 'quant_no_lonely.png')

        shape_optimization(
            exemplar_filename = 'quant_no_lonely.png',
            pyramid_size = pyramid_size,
            n_size = n_size,
            output_file = './output_p/out.png',
            log_dir = './output_p/',
            width_ratio = width_ratio,
            height_ratio = height_ratio,
            rot = rot,
        )

        if use_big_edges:
            b = imread('output_p/out.png', mode='RGB')
            edges = add_eges_grayscale(b)
            big_edges = increase_edge_size(edges)
            imsave('b.png', big_edges)
        else:
            shutil.copy('output_p/out.png', 'b.png')
            shutil.copy('quant_no_lonely.png', 'a.png')

    tbn(
        ai = ai.astype(float) * coefs[0],
        height_map = imread('ahm.png', mode='L').astype(float) if height_map else None,
        fdm = imread('fdm.png', mode='L').astype(float) * coefs[1],
        a = imread('a.png', mode='RGB').astype(float) * coefs[2],
        b = imread('b.png', mode='RGB').astype(float) * coefs[3],
        init_size = init_size,
        neighborhoods = neighborhoods,
        max_iter_time = max_iter_time,
        wrap = wrap,
        init = init,
        init_log_dir = init_log_dir,
        log_dir = './log/',
        optimize_log_dir = None,
        save_output = True,
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ai_filename", default='ai.png', type=str, help="Input image filename.")
    args = parser.parse_args()

    generate_tree_bark(args.ai_filename)
