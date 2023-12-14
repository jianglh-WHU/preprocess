from PIL import Image
import os
import json
import mmcv
import OpenEXR
import Imath
import argparse
import numpy as np
from read_xml import read_xml

def parse_args():
    parser = argparse.ArgumentParser(
        description="downsample images")

    # parser.add_argument("--output_path",
    #                     type=str,
    #                     default='china_museum')
    parser.add_argument("--input_path",
                        type=str,
                        default='sh_city/3')
    parser.add_argument("--input_transforms",
                        type=str,
                        default='wukang_tiejin_up.json')
    parser.add_argument("--input_xml",
                        type=str,
                        default='shizi.xml')
    parser.add_argument("--is_depth",
                        action='store_true',
                        default=False)

    parser.add_argument("--downsample", type=int, default=10)
    
    args = parser.parse_args()
    return args

def downsample(tasks):
    
    img_path,w,h,reso = tasks
    index = img_path.find('images/')
    img_name = img_path[index+len('images/'):]
    input_path = img_path[:index]
    # img_name = os.path.join(*img_path.split('/')[2:])
    # input_path = img_path.split('/')[0]

    img = Image.open(img_path)
    img_wh=(int(w/reso), int(h/reso))

    im_resize = img.resize(img_wh, resample=Image.LANCZOS)
    img_path=os.path.join(input_path, f'images_{reso}',img_name)

    im_resize.save(img_path)

def downsample_depth(tasks):
    
    depth_path,w,h,reso = tasks
    
    index = depth_path.find('images/')
    img_name = depth_path[index+len('images/'):]
    input_path = depth_path[0:index]

    exr = OpenEXR.InputFile(depth_path)
    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    ystr = exr.channel('Y', pt)

    y = np.frombuffer(ystr, dtype=np.float32)
    y.shape = (size[1], size[0])  
    img = Image.fromarray(y, mode='F')
    img_wh=(int(w/reso), int(h/reso))
    # print(img_wh[0], img_wh[1])

    im_resize = img.resize(img_wh, resample=Image.LANCZOS)
    img_path=os.path.join(input_path, f'images_{reso}',img_name)
    # print(im_resize.size)
    
    # im_resize.save(img_path)
    header = OpenEXR.Header(img_wh[0], img_wh[1])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'Y': half_chan}

    exr_save = OpenEXR.OutputFile(img_path, header)

    y_data = np.array(im_resize,dtype=np.float32).tobytes()
    exr_save.writePixels({'Y': y_data})
    exr_save.close()

def traverse_mkdir_folder(input_path,output_path,downsample,dirname=''):
    folder_path = os.path.join(input_path,'images',dirname)
    for filepath,dirnames,filenames in os.walk(folder_path):
        # import pdb;pdb.set_trace()
        for dir in dirnames:
            full_path = os.path.join(input_path,'images',dirname, dir)
            if os.path.isdir(full_path):
                os.makedirs(os.path.join(output_path,f"images_{downsample}",dirname,dir),exist_ok=True)
                dirname_new = os.path.join(dirname, dir)
                if dirname_new == dirname:
                    continue
                traverse_mkdir_folder(input_path,output_path,downsample,dirname_new)
            
if __name__ == "__main__":
    args = parse_args()
    
    INPUT_PATH=args.input_path
    # OUTPUT_PATH=args.output_path
    OUTPUT_PATH = INPUT_PATH
    DOWNSAMPLE = args.downsample
    INPUT_XML = args.input_xml
    
    results = read_xml(input_xml=os.path.join(INPUT_PATH,INPUT_XML))
    tj = results['photo_results']
    # with open(os.path.join(INPUT_PATH,f"{args.input_transforms}"), "r") as f:
    #     tj = json.load(f)
    # import pdb;pdb.set_trace()
    os.makedirs(os.path.join(OUTPUT_PATH,f"images_{DOWNSAMPLE}"),exist_ok=True)

    keys = tj.keys()
    traverse_mkdir_folder(INPUT_PATH,OUTPUT_PATH,DOWNSAMPLE)
    
    tasks=[]
    depth_tasks=[]
    for key in keys:
        frame = tj[key]
        # if "tiejin" not in frame['path']:
            # continue
        # if "huanrao" not in frame['path']:
            # continue
        rot_mat =np.array(frame["rot_mat"])
        w = rot_mat[1,-1]
        h = rot_mat[0,-1]
        if not args.is_depth:
            if 'images' in frame['path']:
                file_path = os.path.join(INPUT_PATH,frame['path'])
            else:
                file_path = os.path.join(INPUT_PATH,'images',frame['path'])
            tasks.append((file_path,w,h,DOWNSAMPLE))
        
        if args.is_depth:
            if 'images' in frame['path']:
                file_path = os.path.join(INPUT_PATH,'rgb',frame['path'])
                depth_path = os.path.join(INPUT_PATH,'depth',frame['path'])
            else:
                file_path = os.path.join(INPUT_PATH,'images','rgb',frame['path'])
                depth_path = os.path.join(INPUT_PATH,'images','depth',frame['path']+'.depth.exr')
            tasks.append((file_path,w,h,DOWNSAMPLE))
            depth_tasks.append((depth_path,w,h,DOWNSAMPLE))
    # import pdb;pdb.set_trace()
    
    mmcv.track_parallel_progress(downsample,tasks,nproc=64)
    if args.is_depth:
        mmcv.track_parallel_progress(downsample_depth,depth_tasks,nproc=64)
    
    