import math
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch

from mmsr.data.transforms import totensor

def paired_paths_from_ann_file(folders, keys, ann_file):
    """Generate paired paths from an anno file.

    Annotation file is a txt file listing all paths of pairs.
    Each line contains the relative lq and gt paths (or input and ref paths),
    separated by a white space.

    Example of an annotation file:
    ```
    lq/0001_x4.png gt/0001.png
    lq/0002_x4.png gt/0002.png
    ```

    Args:
        folders (list): A list of folder root path. The order of list should be:
            [input_folder, ref_folder] or [lq_folder, gt_folder]
        keys (list): A list of keys identifying folders. The order should be in
            consistent with folders. Here are examples: ['input', 'ref'] or
            ['lq', 'gt']
        ann_file (str): Path for annotation file.

    Returns:
        list: Returned path list.
    """
    paths = []
    input_folder = folders[0]  # e.g., input, lq
    ref_folder = folders[1]  # e.g., ref, gt
    input_key = keys[0]
    ref_key = keys[1]

    with open(ann_file, 'r') as fin:
        for line in fin:
            input_path, ref_path = line.strip().split(' ')
            input_path = osp.join(input_folder, input_path)
            ref_path = osp.join(ref_folder, ref_path)
            paths.append(
                dict([(f'{input_key}_path', input_path),
                      (f'{ref_key}_path', ref_path)]))
    return paths