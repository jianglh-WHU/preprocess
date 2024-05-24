import argparse
import csv
import os
import json
import pdb
from PIL import Image
import cv2
import numpy as np
# from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="convert reality capture csv to transforms.json for nerf")

    parser.add_argument("--input_path",
                        type=str,
                        default='.')
    parser.add_argument("--output_path",
                        type=str,
                        default=None)
    parser.add_argument("--images_path",
                        type=str,
                        default=None)
    parser.add_argument("--input_csv",
                        type=str,
                        required=True,
                        default='shizi.xml')
    
    parser.add_argument("--file_name",
                        type=str,
                        default='transforms.json')
    
    parser.add_argument("--output_transforms",
                        type=str,
                        default=None)
    parser.add_argument("--input_img_list",
                        nargs='+',
                        default=None)
    
    parser.add_argument("--is_depth",action='store_true', default=False)

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
    parser.add_argument("--RAW",
                        action='store_true',
                        default=False)
    parser.add_argument("--recur",
                        action='store_true',
                        default=False,
                        help="Whether you need to recursively find images")

    args = parser.parse_args()
    return args

def _get_rotation_matrix(yaw, pitch, roll):
    """Returns a rotation matrix given euler angles."""

    s_yaw = np.sin(np.deg2rad(yaw))
    c_yaw = np.cos(np.deg2rad(yaw))
    s_pitch = np.sin(np.deg2rad(pitch))
    c_pitch = np.cos(np.deg2rad(pitch))
    s_roll = np.sin(np.deg2rad(roll))
    c_roll = np.cos(np.deg2rad(roll))

    rot_x = np.array([[1, 0, 0], [0, c_pitch, -s_pitch], [0, s_pitch, c_pitch]])
    rot_y = np.array([[c_roll, 0, s_roll], [0, 1, 0], [-s_roll, 0, c_roll]])
    rot_z = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]])

    return rot_z @ rot_x @ rot_y

def recur_find_file(path, filename):
    for root, dirs, files in os.walk(path):
        if dirs == [] and filename in files:
            file_path = os.path.join(root, filename)
            return file_path 
        if filename not in files:
            for dir in dirs:
                result = recur_find_file(os.path.join(path, dir), filename)
                
                if result is not None:
                    return result  

    return None
        
    
def read_rc_csv(csv_filename, output_dir, args, downsample=1):
    data = {}
    frames = []
    in_img_list, RAW = args.input_img_list, args.RAW
    is_recur = args.recur
    with open(csv_filename, encoding="UTF-8") as file:
        reader = csv.DictReader(file)
        cameras = {}
        for row in reader:
            for column, value in row.items():
                cameras.setdefault(column, []).append(value)
    
    missing_image_data = 0

    # processed_img means that play rc together
    if is_recur:
        for i, name in tqdm(enumerate(cameras["#name"]), desc="Processing"):
            basename = name
            # import pdb;pdb.set_trace()
            file_path = recur_find_file(output_dir, basename)
            if file_path == None:
                print(f'{basename} is not found')
                continue
                
            img = np.array(Image.open(file_path))
            
            frame = {}
            
            height, width, _ = img.shape
            frame["h"] = int(height / downsample)
            frame["w"] = int(width / downsample)
            
            left_part, right_part = file_path.split("images/")
            file_path = file_path if downsample == 1 else os.path.join(left_part, f'images_{downsample}', right_part)
            
            frame['file_path'] = file_path
            # frame["file_path"] = image_filename_map[basename].as_posix()
            frame["fl_x"] = float(cameras["f"][i]) * max(width, height) / 36 / downsample
            frame["fl_y"] = float(cameras["f"][i]) * max(width, height) / 36 / downsample
            # TODO: Unclear how to get the principal point from RealityCapture, here a guess...
            frame["cx"] = float(cameras["px"][i]) / 36.0 + width / 2.0 / downsample
            frame["cy"] = float(cameras["py"][i]) / 36.0 + height / 2.0 / downsample
            # TODO: Not sure if RealityCapture uses this distortion model
            frame["k1"] = cameras["k1"][i]
            frame["k2"] = cameras["k2"][i]
            frame["k3"] = cameras["k3"][i]
            frame["k4"] = cameras["k4"][i]
            frame["p1"] = cameras["t1"][i]
            frame["p2"] = cameras["t2"][i]

            # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
            rot = _get_rotation_matrix(-float(cameras["heading"][i]), float(cameras["pitch"][i]), float(cameras["roll"][i]))

            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = np.array([float(cameras["x"][i]), float(cameras["y"][i]), float(cameras["alt"][i])])

            frame["transform_matrix"] = transform.tolist()
            frames.append(frame)
            
            # if i > 10:
            #     break
        data["frames"] = frames
        return data
               
    processed_img = False if in_img_list==None else True
    if not processed_img:
        if RAW == False:
            image_filename_map = os.listdir(os.path.join(output_dir, 'images', args.images_path))
        else:
            image_filename_map = os.listdir(os.path.join(output_dir, 'images'))
    
    # import pdb;pdb.set_trace()
    for i, name in tqdm(enumerate(cameras["#name"]), desc="Processing"):
        # basename = name.rpartition(".")[0]
        basename = name
        if not processed_img:
            if basename not in image_filename_map:
                print(f"Missing image for camera data {basename}, Skipping")
                missing_image_data += 1
                continue
            if RAW == False:
                # img = np.array(Image.open(os.path.join(output_dir, 'images', args.images_path, basename)))
                img = cv2.imread(os.path.join(output_dir, 'images', args.images_path, basename))
            else:
                img = np.array(Image.open(os.path.join(output_dir, 'images_cp', basename)))
        else:
            # import pdb;pdb.set_trace()
            for img_path in in_img_list:
                missing_index = 0
                image_filename_map = os.listdir(os.path.join(output_dir, img_path, 'images', "rgb"))
                if basename not in image_filename_map:
                    missing_index += 1
                    continue
                else:
                    break

            if missing_index == 2:
                print(f"Missing image for camera data {basename}, Skipping")
                missing_image_data += 1
                continue
            
            img = np.array(Image.open(os.path.join(output_dir, img_path, 'images', "rgb", basename)))
        
        frame = {}
        # import pdb;pdb.set_trace()
        
        height, width, _ = img.shape
        frame["h"] = int(height / downsample)
        frame["w"] = int(width / downsample)
        if not processed_img:
            if RAW == None:
                file_path = os.path.join('images', args.images_path, basename) if downsample==1 else os.path.join(f'images_{downsample}',args.images_path, basename)
            else:
                file_path = os.path.join('images', args.images_path, basename) if downsample==1 else os.path.join(f'images_{downsample}',args.images_path, basename)
        else:
            file_path = os.path.join(output_dir, img_path,'images', "rgb", basename) if downsample==1 else os.path.join(output_dir, img_path, f'images_{downsample}', "rgb", basename)
        frame['file_path'] = file_path
        # frame["file_path"] = image_filename_map[basename].as_posix()
        frame["fl_x"] = float(cameras["f"][i]) * max(width, height) / 36 / downsample
        frame["fl_y"] = float(cameras["f"][i]) * max(width, height) / 36 / downsample
        # TODO: Unclear how to get the principal point from RealityCapture, here a guess...
        frame["cx"] = float(cameras["px"][i]) / 36.0 + width / 2.0 / downsample
        frame["cy"] = float(cameras["py"][i]) / 36.0 + height / 2.0 / downsample
        # TODO: Not sure if RealityCapture uses this distortion model
        frame["k1"] = cameras["k1"][i]
        frame["k2"] = cameras["k2"][i]
        frame["k3"] = cameras["k3"][i]
        frame["k4"] = cameras["k4"][i]
        frame["p1"] = cameras["t1"][i]
        frame["p2"] = cameras["t2"][i]
        if args.is_depth:
            frame["depth_path"] = os.path.join('images', "depth", basename+".depth.exr")

        # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
        rot = _get_rotation_matrix(-float(cameras["heading"][i]), float(cameras["pitch"][i]), float(cameras["roll"][i]))

        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = np.array([float(cameras["x"][i]), float(cameras["y"][i]), float(cameras["alt"][i])])

        frame["transform_matrix"] = transform.tolist()
        frames.append(frame)
        
        # if i>5:
        #     break
    data["frames"] = frames

    return data
                                          
    
if __name__ == "__main__":
    args = parse_args()
    INPUT_PATH=args.input_path
    INPUT_CSV = os.path.join(INPUT_PATH, args.input_csv)
    OUTPUT_PATH = INPUT_PATH if args.output_path == None else args.output_path
    # IMAGES_PATH = OUTPUT_PATH if args.images_path == None else args.images_path
    DOWNSAMPLE = args.downsample
    output_name = args.output_transforms
    # import pdb;pdb.set_trace()
    
    results = read_rc_csv(INPUT_CSV, OUTPUT_PATH, args, DOWNSAMPLE)
    # file_name = "transforms.json" if DOWNSAMPLE == 1 else f"transforms_{DOWNSAMPLE}.json"
    file_name = args.file_name
    file_name = file_name if output_name == None else output_name
    with open(os.path.join(OUTPUT_PATH , file_name), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    # with open(os.path.join(INPUT_PATH , file_name), "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4)