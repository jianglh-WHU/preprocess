import argparse
import os
import json
import imageio
import numpy as np
from pyproj import CRS

def parse_args():
    parser = argparse.ArgumentParser(
        description="convert cc xml to transforms.json for nerfstudio")

    parser.add_argument("--input_path",
                        type=str,
                        default='.')
    parser.add_argument("--input_transforms",
                        type=str,
                        default='71_6.json')
    parser.add_argument("--input_prj",
                        type=str,
                        required=True,
                        default='data/kongsan/48_58.prj')
    
    parser.add_argument("--output_transforms",
                        type=str,
                        default='xuhui.json')
    
    parser.add_argument("--same_intri",action='store_true', default=False)

    parser.add_argument("--downsample", type=int, default=1)
    
    parser.add_argument("--orientation_method", type=str, default='none',
                        choices=["pca", "up", "vertical", "none"],
                        help='The method to use for orientation')
    
    parser.add_argument("--center_method", type=str, default='poses',
                        choices=["poses", "focus", "none"],
                        help='The method to use to center the poses')

    parser.add_argument("--auto_scale_poses", action='store_false', default=False)
    
    
    args = parser.parse_args()
    return args

def list_tif(input_path):
    tif_list=[]
    for root, dirs, files in os.walk(input_path):
        for file_name in files:
            if file_name.endswith(".tif"):  
                tif_list.append(file_name)
    return tif_list

def read_prj(input_prj, img_list):
    c2ws=[]
    focal=None
    cx=None
    cy=None
    w=None
    h=None
    is_reading_ext_ori = False
    is_true_tif = False
    line_id = 0
    intri_line_id = 0
    is_read_intr = False
    all_frames = []
    with open(input_prj, "r") as f:
        for line in f:
            parts = line.strip()
            if parts.startswith("$PHOTO_FILE"): 
                photo_parts = parts.split(":")  
                file_path=photo_parts[1].split('\\')[-1]
                if file_path in img_list:
                    is_true_tif = True
                        
            if parts.startswith("$EXT_ORI") and is_true_tif:
                is_reading_ext_ori = True
                c2w = np.identity(4)
                
            if parts.startswith("$PHOTO_FOOTPRINT"):
                is_true_tif = False
                is_reading_ext_ori = False
                line_id = 0
                c2w = np.identity(4)
                
            if is_true_tif and is_reading_ext_ori and not parts.startswith("$EXT_ORI"):
                field = line.split()
                if line_id == 0:
                    c2w[0,3] = field[1]
                    c2w[1,3] = field[2]
                    c2w[2,3] = field[3]
                    
                if line_id > 0:
                    c2w[line_id-1,0] = field[0]
                    c2w[line_id-1,1] = field[1]
                    c2w[line_id-1,2] = field[2]
                
                if line_id == 3:
                    c2ws.append(c2w)
                    all_frames.append({'file_path':file_path,'transform_matrix':c2w.tolist()})
                line_id+=1
            
            if parts.startswith("$CCD_INTERIOR_ORIENTATION"):
                is_read_intr = True
                continue
            
            if is_read_intr:
                field = line.split()
                # import pdb;pdb.set_trace()
                if intri_line_id == 0:
                    cx = float(field[-1])
                elif intri_line_id == 1:
                    cy = float(field[-1])
                intri_line_id+=1
                
            if parts.startswith("$FOCAL_LENGTH"):
                is_read_intr=False
                focal_parts = parts.split(":") 
                focal = float(focal_parts[1].strip())
            if parts.startswith("$CCD_COLUMNS"):
                ccd_columns = parts.split(":")
                h = float(ccd_columns[1].strip())
            if parts.startswith("$CCD_ROWS"):
                ccd_rows = parts.split(":") 
                w = float(ccd_rows[1].strip())

    poses = np.array(c2ws).astype(np.float32)
    print(poses.shape)
    
    transforms = {
                "focal": focal,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "frames": []
            }
    for i,frame in enumerate(all_frames):
        transforms['frames'].append(frame)
    print(all_frames)
    return transforms
 
if __name__ == "__main__":
    args = parse_args()
    INPUT_PATH=args.input_path
    INPUT_PRJ = args.input_prj
    # OUTPUT_PATH=args.output_path
    OUTPUT_PATH = INPUT_PATH
    DOWNSAMPLE = args.downsample
    img_list = list_tif(INPUT_PATH)
    print(img_list)
    transforms = read_prj(INPUT_PRJ, img_list)
    
    with open(os.path.join(OUTPUT_PATH, args.output_transforms),"w") as outfile:
        json.dump(transforms, outfile, indent=2)
    