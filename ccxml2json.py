import argparse
import os
import json
import imageio
import numpy as np
import torch
# from utils import auto_orient_and_center_poses
from read_xml import read_xml

def parse_args():
    parser = argparse.ArgumentParser(
        description="convert cc xml to transforms.json for nerfstudio")

    # parser.add_argument("--output_path",
    #                     type=str,
    #                     default='china_museum')
    parser.add_argument("--input_path",
                        type=str,
                        default='.')
    parser.add_argument("--input_transforms",
                        type=str,
                        default='71_6.json')
    parser.add_argument("--input_xml",
                        type=str,
                        required=True,
                        default='shizi.xml')
    
    parser.add_argument("--output_transforms",
                        type=str,
                        default='shizi.json')
    
    parser.add_argument("--same_intri",action='store_true', default=False)

    parser.add_argument("--downsample", type=int, default=1)
    
    parser.add_argument("--orientation_method", type=str, default='none',
                        choices=["pca", "up", "vertical", "none"],
                        help='The method to use for orientation')
    
    parser.add_argument("--center_method", type=str, default='poses',
                        choices=["poses", "focus", "none"],
                        help='The method to use to center the poses')

    parser.add_argument("--auto_scale_poses", action='store_false', default=False)
    
    parser.add_argument("--train_skip",
                        type=int,
                        default=10,
                        help="index%train_skip==9 -> test")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    INPUT_PATH=args.input_path
    INPUT_XML = args.input_xml
    # OUTPUT_PATH=args.output_path
    OUTPUT_PATH = INPUT_PATH
    DOWNSAMPLE = args.downsample
    TRAIN_SKIP = args.train_skip
    tj = read_xml(input_xml=INPUT_XML) 

    # with open(os.path.join(INPUT_PATH,f"{args.input_transforms}"), "r") as f:
    #     tj = json.load(f)
    
    if args.same_intri:
        frame_1 = tj['0'] 
        rot_mat =np.array(frame_1["rot_mat"])
        
        w = int(rot_mat[1,-1] / DOWNSAMPLE)
        h = int(rot_mat[0,-1] / DOWNSAMPLE)
        fl_x = rot_mat[2,-1] / DOWNSAMPLE
        fl_y = rot_mat[2,-1] / DOWNSAMPLE
        
        cx = w / 2 
        cy = h / 2 
        k1 = 0
        k2 = 0
        k3 = 0
        p1 = 0
        p2 = 0
        angle_x = 2* np.arctan(w/(2*fl_x))
            
        transforms = {
                "camera_angle_x": angle_x,
                "camera_model": "SIMPLE_PINHOLE",
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "p1": p1,
                "p2": p2,
                "frames": []
            }
    
        # train_json = {
        #         "camera_angle_x": angle_x,
        #         "camera_model": "SIMPLE_PINHOLE",
        #         "fl_x": fl_x,
        #         "fl_y": fl_y,
        #         "cx": cx,
        #         "cy": cy,
        #         "w": w,
        #         "h": h,
        #         "k1": k1,
        #         "k2": k2,
        #         "k3": k3,
        #         "p1": p1,
        #         "p2": p2,
        #         "frames": []
        #     }
        
        # test_json = {
        #         "camera_angle_x": angle_x,
        #         "camera_model": "SIMPLE_PINHOLE",
        #         "fl_x": fl_x,
        #         "fl_y": fl_y,
        #         "cx": cx,
        #         "cy": cy,
        #         "w": w,
        #         "h": h,
        #         "k1": k1,
        #         "k2": k2,
        #         "k3": k3,
        #         "p1": p1,
        #         "p2": p2,
        #         "frames": []
        #     }
        
        all_frames=[]
        # import pdb; pdb.set_trace()
        
        c2ws=[]
        for i in range(len(tj)):
            frame = tj[f'{i}']
            c2w = np.array(frame['rot_mat'])
            c2w = c2w[...,:-1]
            c2ws.append(c2w)
        
        c2ws=np.array(c2ws) 
        
        poses = torch.from_numpy(np.array(c2ws).astype(np.float32))
        # poses, transform_matrix = auto_orient_and_center_poses(
        #     poses, # type: ignore
        #     method=args.orientation_method,
        #     center_method=args.center_method,
        # )

        # Scale poses
        scale_factor = 1.0
        # scale_factor = 0.67
        if args.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3]))) 

        # poses[:, :3, 3] *= scale_factor
        
        c2ws = poses.numpy()
        
        for i in range(len(tj)):
            frame = tj[f'{i}']
            file_path = frame['path']
            suffix = 'images' if DOWNSAMPLE==1 else f'images_{DOWNSAMPLE}'
            file_path = os.path.join(f'{suffix}',file_path)
            # import pdb; pdb.set_trace()
            
            c2w = np.concatenate((c2ws[i],np.array([[0,0,0,1]])),axis=0)
            all_frames.append({'file_path':file_path,'transform_matrix':c2w.tolist()})

        for i,frame in enumerate(all_frames):
            transforms['frames'].append(frame)
        #     if i % TRAIN_SKIP == 9:
        #         test_json['frames'].append(frame)
        #     else:
        #         train_json['frames'].append(frame)
                
        # val_json = test_json
    else:
        transforms = {
                "camera_model": "SIMPLE_PINHOLE",
                "orientation_override": "none",
                "frames": []
            }
        all_frames=[]
        # with open(os.path.join(INPUT_PATH,f"{args.input_transforms}"), "r") as f:
        #     tj = json.load(f)
        keys = tj.keys()
        c2ws = []
        
        for key in keys:
            frame = tj[key]
            c2w = np.array(frame['rot_mat'])
            c2w = c2w[...,:-1]
            c2ws.append(c2w)
            
        poses = torch.from_numpy(np.array(c2ws).astype(np.float32))
        # import pdb;pdb.set_trace()
        # poses[...,:3,3] -=torch.mean(poses[...,:3,3], dim=0)
        # poses, transform_matrix = auto_orient_and_center_poses(
        #     poses, # type: ignore
        #     method=args.orientation_method,
        #     center_method=args.center_method,
        # )
        # Scale poses
        scale_factor = 1.0
        if args.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3]))) 
        # import pdb;pdb.set_trace()
        print('scale factor:', scale_factor)
        poses[:, :3, 3] *= scale_factor
        
        c2ws = poses.numpy()
        
        for i,key in enumerate(keys):
            frame = tj[key]
            file_path = frame['path']
            rot_mat = np.array(frame['rot_mat'])
            
            w = int(rot_mat[1,-1] / DOWNSAMPLE)
            h = int(rot_mat[0,-1] / DOWNSAMPLE)
            fl_x = rot_mat[2,-1] / DOWNSAMPLE
            fl_y = rot_mat[2,-1] / DOWNSAMPLE
            
            cx = w / 2 
            cy = h / 2 
            
            suffix = 'images' if DOWNSAMPLE==1 else f'images_{DOWNSAMPLE}'
            file_path = os.path.join(f'{suffix}',file_path)
            c2w = np.concatenate((c2ws[i],np.array([[0,0,0,1]])),axis=0)
            frame_dict = {
                'fl_x':fl_x,
                'fl_y':fl_y,
                'cx':cx,
                'cy':cy,
                'w':w,
                'h':h,
                'file_path':file_path,
                'transform_matrix':c2w.tolist(),
                }
            all_frames.append(frame_dict)
            # import pdb; pdb.set_trace()

        for i,frame in enumerate(all_frames):
            transforms['frames'].append(frame)
        
            
    #TODO: save
    with open(os.path.join(OUTPUT_PATH, args.output_transforms),"w") as outfile:
        json.dump(transforms, outfile, indent=2)
    
    # with open(os.path.join(INPUT_PATH, 'transforms_train_pca.json'),"w") as outfile:
    #     json.dump(train_json, outfile, indent=2)
    # with open(os.path.join(INPUT_PATH, 'transforms_test_pca.json'),"w") as outfile:
    #     json.dump(test_json, outfile, indent=2)
    # with open(os.path.join(INPUT_PATH, 'transforms_val_pca.json'), "w") as outfile:
    #     json.dump(val_json, outfile, indent=2)
        