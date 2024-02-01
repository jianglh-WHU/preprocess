from PIL import Image
import os
import json
from downsample_utils import get_tasks_json, get_tasks_xml
import mmcv
import OpenEXR
import Imath
import argparse
import numpy as np
from read_xml import read_xml

def parse_args():
    parser = argparse.ArgumentParser(
        description="downsample images")

    parser.add_argument("--json_path",
                        type=str,
                        default=None)
    parser.add_argument("--input_path",
                        type=str,
                        default='sh_city/3')
    parser.add_argument("--input_transforms",
                        type=str,
                        default='wukang_tiejin_up.json')
    parser.add_argument("--input_xml",
                        type=str,
                        default='shizi.xml')
    parser.add_argument("--input_csv",
                        type=str,
                        default='zao.csv')
    parser.add_argument("--type",type=str,choices=['xml','json','all'])
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
    
    os.makedirs(os.path.join(OUTPUT_PATH,f"images_{DOWNSAMPLE}"),exist_ok=True)
    traverse_mkdir_folder(INPUT_PATH,OUTPUT_PATH,DOWNSAMPLE)

    if args.type=='json':
        tasks,depth_tasks=get_tasks_json(INPUT_PATH, DOWNSAMPLE, args)
    
    elif args.type=='xml':
        tasks,depth_tasks=get_tasks_xml(INPUT_PATH, INPUT_XML, DOWNSAMPLE, args)
        
    mmcv.track_parallel_progress(downsample,tasks,nproc=64)
    if args.is_depth:
        mmcv.track_parallel_progress(downsample_depth,depth_tasks,nproc=64)
    
    