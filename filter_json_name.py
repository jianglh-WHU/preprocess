import argparse
import os
import json
import imageio
from tqdm import tqdm
import numpy as np
from kornia import create_meshgrid
from typing import NamedTuple
import torch
from plyfile import PlyData, PlyElement
import os
import csv
import pdb

def parse_args():
    parser = argparse.ArgumentParser(
        description="normalize and center json.")

    
    parser.add_argument("--input_path",
                        type=str,
                        default='data/shimao/')
    
    parser.add_argument("--input_json",
                        type=str,
                        default='transforms_5.json')

    parser.add_argument("--name",
                        type=str,
                        default='DJ')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    INPUT_PATH=args.input_path
    # OUTPUT_PATH=args.output_path
    OUTPUT_PATH = INPUT_PATH

    with open(os.path.join(INPUT_PATH,f"{args.input_json}"), "r") as f:
        meta = json.load(f)

    transforms = {
            "camera_model": "SIMPLE_PINHOLE",
            "orientation_override": "none",
            "frames": []
        }
    all_frames=[]

    frames = meta['frames']
    
    for i,frame in tqdm(enumerate(frames)):
        if args.name not in frame["file_path"]:
            continue
        all_frames.append(frame)
            
    for i,frame in enumerate(all_frames):
        transforms['frames'].append(frame)

    with open(os.path.join(INPUT_PATH,'hr', f"transforms_5.json"),"w") as outfile:
        json.dump(transforms, outfile, indent=2)
