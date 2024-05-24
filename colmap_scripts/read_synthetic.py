import argparse
import os
import json
import imageio
import numpy as np
from tqdm import tqdm

def read_pose(input_path):
    with open(os.path.join(input_path, "transforms_train.json"), 'r') as f:
       frames = json.load(f)["frames"]
    with open(os.path.join(input_path, "transforms_val.json"), 'r') as f:
        frames+= json.load(f)["frames"]
    with open(os.path.join(input_path, "transforms_test.json"), 'r') as f:
        frames+= json.load(f)["frames"]
    poses=[]
    img_names=[]
    W = H = 800
    fx = fy = 0.5*800/np.tan(0.5*0.6194058656692505)

    K = np.float32([[fx, 0, W/2],
                    [0, fy, H/2],
                    [0,  0,   1]])

    for frame in tqdm(frames):
        c2w = np.array(frame['transform_matrix'])
        c2w[:, 1:3] *= -1 # opengl -> opencv
        w2c = np.linalg.inv(c2w)
        poses.append(w2c)
        name = str(frame['file_path']).split('/')[1]+'-'+str(frame['file_path']).split('/')[-1]+'.png'
        img_names.append(os.path.join(str(frame['file_path']).split('/')[1],name))
    poses = np.stack(poses)
    NUM_IMAGES=poses.shape[0]
    return K, poses, H, W, NUM_IMAGES, img_names