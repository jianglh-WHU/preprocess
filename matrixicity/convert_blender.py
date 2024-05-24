import argparse
import os
import json
import imageio
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="convert matrixcity transforms_origin.json to nerfstudio format.")
    
    parser.add_argument("--input_path",
                        type=str,
                        default='/cpfs01/shared/pjlab_lingjun_landmarks/data/openxlab_matrix/small_city/street/train/small_city_road_down')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(os.path.join(args.input_path,"transforms_origin.json"), "r") as f:
        tj = json.load(f)
    
    SCALE = 100 # fix ue bug
    img_0_path = os.path.join(args.input_path,str(tj['frames'][0]['frame_index']).zfill(4)+'.png')
    img_0 = imageio.imread(img_0_path)
    
    angle_x = tj['camera_angle_x']
    frames = tj['frames']
    w = float(img_0.shape[1])
    h = float(img_0.shape[0])
    fl_x = float(.5 * w / np.tan(.5 * angle_x))
    fl_y = fl_x
    cx = w / 2
    cy = h / 2
    
    all_frames=[]
    for i, frame in enumerate(frames):
        file_path = os.path.abspath(os.path.join(args.input_path,str(frame['frame_index']).zfill(4)+'.png'))
        depth_path = os.path.abspath(os.path.join(args.input_path,'../small_city_road_down_depth',str(frame['frame_index']).zfill(4)+'.exr'))
        c2w = np.array(frame['rot_mat'])
        c2w[:3,:3] *= 100
        c2w[:3,3] /= SCALE
        all_frames.append({'file_path':file_path, 'depth_path':depth_path, 'transform_matrix':c2w.tolist()})
    
    # import pdb;pdb.set_trace()
    all_json = {
            "camera_angle_x": angle_x,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "frames": []
        }

    for i,frame in enumerate(all_frames):
        all_json['frames'].append(frame)
    
    with open(os.path.join(args.input_path, 'transforms_.json'),"w") as outfile:
        json.dump(all_json, outfile, indent=2)
