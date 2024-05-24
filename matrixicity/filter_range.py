import argparse
import os
import json
import imageio
from tqdm import tqdm
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="filter and merge matrixcity aerial & street pose")

    parser.add_argument("--aerial_path",
                        type=str,
                        default='data/openxlab_matrix/street/pose')
    
    parser.add_argument("--input_transforms",
                        type=str,
                        default='transform_71_6_1K.json')

    parser.add_argument("--aabb",
                        nargs='+',
                        type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    