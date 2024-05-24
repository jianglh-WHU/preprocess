import argparse
import os
import json
import imageio
import numpy as np


def read_all_pose(input_path):
    SCALE = 100
    poses=[]
    img_names = []
    for s in ["train", "test"]:
        with open(os.path.join(input_path,f'transforms_{s}.json'), "r") as f:
            tj = json.load(f)
            
        img_0_path = os.path.normpath(os.path.join(input_path, tj['frames'][0]["file_path"]))
        img_0 = imageio.imread(img_0_path)
        angle_x = tj['camera_angle_x']
        w = float(img_0.shape[1])
        h = float(img_0.shape[0])
        fx = float(.5 * w / np.tan(.5 * angle_x))
        fy = fx
        cx = w/2
        cy = h/2
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    
    
        frames = tj['frames']
        bottom =np.array([0,0,0,1])
        # import pdb;pdb.set_trace()
        for i, frame in enumerate(frames):
            c2w = np.array(frame['transform_matrix'])
            c2w[:, 1:3] *= -1 # opengl -> opencv
            # blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # c2w = c2w @ blender2opencv
            # R = np.linalg.inv(w2c[:3, :3])
            # T = -np.matmul(R, w2c[:3, 3])
            # a= np.linalg.inv(c2w)
            # import pdb;pdb.set_trace()
            # b = np.concatenate([np.concatenate([R, T], 1), bottom], 0)
            # poses.append(np.concatenate([np.concatenate([R, T], 1), bottom], 0))
            
            w2c = np.linalg.inv(c2w) # c2w -> w2c
            # import pdb;pdb.set_trace()
            poses.append(w2c)
            # file_path_list = frame['file_path'].split('/')
            # img_name = file_path_list[-3]+'-'+file_path_list[-1]
            # img_name = os.path.normpath(os.path.join(input_path, frame['file_path']))
            img_name = frame['file_path']
            img_names.append(img_name)
            # import pdb;pdb.set_trace()
            # poses.append(c2w)
    
    poses = np.stack(poses)
    H = h
    W = w
    NUM_IMAGES = poses.shape[0]
    # import pdb;pdb.set_trace()
    return K, poses, H, W, NUM_IMAGES, img_names

if __name__ == '__main__':
    # K, poses, H, W, NUM_IMAGES, names=read_all_pose('data/small_city_img/new_high/all/')
    K, poses, H, W, NUM_IMAGES, names=read_all_pose('data/blockdemo')
    print(NUM_IMAGES)
    import pdb;pdb.set_trace()
    
        
        