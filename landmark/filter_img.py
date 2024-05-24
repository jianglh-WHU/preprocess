import os
import numpy as np
import argparse
from typing import NamedTuple
from xml.dom import minidom
import numpy as np
import open3d as o3d
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement
import os
import csv
import pdb

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    pdb.set_trace()
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.zeros((positions.shape[0], 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def parse_args():
    parser = argparse.ArgumentParser(
        description="filter images based on pc range")

    parser.add_argument("--input_json",
                        type=str,
                        default='transforms_5.json')
    parser.add_argument("--input_pcd",
                        type=str,
                        default='shimao_500.ply')
    parser.add_argument("--input_path",
                        type=str,
                        default='data/shimao/all')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    pcd = fetchPly(os.path.join(args.input_path, args.input_pcd))
    xmin,ymin = np.min(pcd.points,0)[:-1]
    xmax,ymax = np.max(pcd.points,0)[:-1]
    print(xmin,ymin,xmax,ymax)
    
    frames_filter = []
    with open(os.path.join(args.input_path, args.input_json)) as json_file:
        contents = json.load(json_file)
        transforms_filter = {
                "camera_model": contents["camera_model"],
                "frames": []
            }
        

        frames = contents["frames"]
        
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        for i, frame in enumerate(frames):
            c2w = np.array(frame["transform_matrix"])
            # import pdb;pdb.set_trace()
            if c2w[:,-1][0]>xmin and c2w[:,-1][0]<xmax and c2w[:,-1][1]>ymin and c2w[:,-1][1]<ymax:
                frame_dict = {
                    'fl_x':frame["fl_x"],
                    'fl_y':frame["fl_y"],
                    'cx':frame["cx"],
                    'cy':frame["cy"],
                    'w':frame["w"],
                    'h':frame["h"],
                    'file_path':frame["file_path"],
                    'depth_path':[],
                    'transform_matrix':c2w.tolist(),
                    }
                frames_filter.append(frame_dict)
    for i,frame in enumerate(frames_filter):
        transforms_filter['frames'].append(frame)
    output_transforms = args.input_json.split(".")[0]+"_filter.json"
    with open(os.path.join(args.input_path, output_transforms),"w") as outfile:
        json.dump(transforms_filter, outfile, indent=2)
        
        
    
    
    
    

