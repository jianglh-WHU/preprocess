import argparse
import os
import json
import imageio
import numpy as np

def list_of_strings(arg):
    return arg.split(',')

def parse_args():
    parser = argparse.ArgumentParser(
        description="merge n jsons to 1.")

    parser.add_argument('--json_list', type=list_of_strings)
    
    parser.add_argument("--output_json",
                        type=str,
                        default='transforms_4K.json')
    
    parser.add_argument("--input_path",
                        type=str,
                        default='zhangjiang/pose')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    json_list = args.json_list
    
    transforms = {
            "frames": []
        }
    
    for json_path in json_list:
        with open(os.path.join(args.input_path,args.json_path), "r") as f:
            tj = json.load(f)
        frames = tj['frames']
    
        for idx, frame in enumerate(frames):
            transforms['frames'].append(frame)

    with open(os.path.join(args.input_path,args.output_json), 'w') as f:
        json.dump(transforms, f, indent=4)