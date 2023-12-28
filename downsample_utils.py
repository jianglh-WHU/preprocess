
import json
import numpy as np
from read_xml import read_xml
import os
from tqdm import tqdm


def get_tasks_xml(input_path, input_xml, downsample, args):
    results = read_xml(input_xml=os.path.join(input_path, input_xml))
    tj = results['photo_results']
    keys = tj.keys()
    
    tasks=[]
    depth_tasks=[]
    for key in keys:
        frame = tj[key]
        rot_mat =np.array(frame["rot_mat"])
        w = rot_mat[1,-1]
        h = rot_mat[0,-1]
        if not args.is_depth:
            if 'images' in frame['path']:
                file_path = os.path.join(input_path,frame['path'])
            else:
                file_path = os.path.join(input_path,'images',frame['path'])
            tasks.append((file_path,w,h,downsample))
        
        if args.is_depth:
            if 'images' in frame['path']:
                depth_path = os.path.join(input_path,'depth',frame['path'])
                if 'indoor' in input_path:
                    file_path = os.path.join(input_path,'rgb','JPG',frame['path'])
                else:
                    file_path = os.path.join(input_path,'rgb',frame['path'])
            else:
                depth_path = os.path.join(input_path,'images','depth',frame['path']+'.depth.exr')
                if 'indoor' in input_path:
                    file_path = os.path.join(input_path,'images','rgb','JPG',frame['path'])
                else:
                    file_path = os.path.join(input_path,'images','rgb',frame['path'])
            tasks.append((file_path,w,h,downsample))
            if not os.path.exists(depth_path):
                continue
            depth_tasks.append((depth_path,w,h,downsample))
    return tasks, depth_tasks

def get_tasks_json(input_path, input_xml, downsample, args):
    with open(os.path.join(input_path,f"{args.input_transforms}"), "r") as f:
        tj = json.load(f)
    tasks=[]
    depth_tasks=[]
    frames = tj['frames']
    
    for i,frame in tqdm(enumerate(frames), desc="Processing"):
        # import pdb;pdb.set_trace()
        w = frame['w']
        h = frame['h']
        if not args.is_depth:
            if 'images' in frame['file_path']:
                file_path = os.path.join(input_path,frame['file_path'])
            else:
                file_path = os.path.join(input_path,'images',frame['file_path'])
            tasks.append((file_path,w,h,downsample))
            
        if args.is_depth:
            if 'images' in frame['file_path']:
                depth_path = os.path.join(input_path, frame['depth_path'])
                file_path = os.path.join(input_path, frame['file_path'])
            else:
                depth_path = os.path.join(input_path,'images', frame['depth_path'])
                file_path = os.path.join(input_path,'images', frame['file_path'])
            tasks.append((file_path,w,h,downsample))
            if not os.path.exists(depth_path):
                continue
            depth_tasks.append((depth_path,w,h,downsample))
    return tasks, depth_tasks