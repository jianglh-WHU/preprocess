
import json
import numpy as np
from read_xml import read_xml
import os
from tqdm import tqdm
from PIL import Image


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

def get_tasks_json(input_path, downsample, args):
    json_path = input_path if args.json_path is None else args.json_path
    with open(os.path.join(json_path, f"{args.input_transforms}"), "r") as f:
        tj = json.load(f)
    tasks=[]
    depth_tasks=[]
    frames = tj['frames']
    
    for i,frame in tqdm(enumerate(frames), desc="Processing"):
        # import pdb;pdb.set_trace()
        w = frame['w']
        h = frame['h']
        if not args.is_depth:
            if input_path in frame['file_path']:
                file_path = frame['file_path']
            else:
                if 'images' in frame['file_path']:
                    file_path = os.path.join(input_path,frame['file_path'])
                else:
                    file_path = os.path.join(input_path,'images',frame['file_path'])
            try:
                # if already have been ds 
                Image.open(file_path)
                continue
            except:
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
    # import pdb;pdb.set_trace()
    return tasks, depth_tasks

def get_tasks_all(input_path, downsample, args):
    
    tasks=[]
    depth_tasks=[]
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(input_path, filename)

            with Image.open(file_path) as img:
                # 计算下采样后的尺寸
                new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
                # 下采样图片
                downsampled_img = img.resize(new_size, Image.ANTIALIAS)
                # 保存下采样后的图片
                downsampled_img.save(file_path)
                print(f"Downsampled {filename} to {new_size}")

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