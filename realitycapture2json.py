import argparse
import csv
import os
import json
from PIL import Image
import numpy as np
# from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="convert reality capture csv to transforms.json for nerf")

    parser.add_argument("--input_path",
                        type=str,
                        default='.')
    parser.add_argument("--input_transforms",
                        type=str,
                        default='71_6.json')
    parser.add_argument("--input_csv",
                        type=str,
                        required=True,
                        default='shizi.xml')
    
    parser.add_argument("--output_transforms",
                        type=str,
                        default='shizi.json')
    
    parser.add_argument("--input_img_path",type=str, default='.')
    
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

def read_rc_csv(csv_filename, output_dir, downsample=1, verbose=False):
    data = {}
    frames = []
    with open(csv_filename, encoding="UTF-8") as file:
        reader = csv.DictReader(file)
        cameras = {}
        for row in reader:
            for column, value in row.items():
                cameras.setdefault(column, []).append(value)
    
    missing_image_data = 0

    image_filename_map = os.listdir(os.path.join(output_dir, 'images', "rgb"))
    for i, name in tqdm(enumerate(cameras["#name"]), desc="Processing"):
        # basename = name.rpartition(".")[0]
        basename = name
        if basename not in image_filename_map:
            print(f"Missing image for camera data {basename}, Skipping")
            missing_image_data += 1
            continue
        
        frame = {}
        # import pdb;pdb.set_trace()
        img = np.array(Image.open(os.path.join(output_dir, 'images', "rgb", basename)))
        height, width, _ = img.shape
        frame["h"] = int(height / downsample)
        frame["w"] = int(width / downsample)
        file_path = os.path.join('images', "rgb", basename) if downsample==1 else os.path.join(f'images_{downsample}', "rgb", basename)
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
    data["frames"] = frames

    return data
                                          
    
if __name__ == "__main__":
    args = parse_args()
    INPUT_PATH=args.input_path
    INPUT_CSV = os.path.join(INPUT_PATH, args.input_csv)
    OUTPUT_PATH = INPUT_PATH
    DOWNSAMPLE = args.downsample
    
    results = read_rc_csv(INPUT_CSV, OUTPUT_PATH, DOWNSAMPLE)
    file_name = "transforms.json" if DOWNSAMPLE == 1 else f"transforms_{DOWNSAMPLE}.json"
    with open(os.path.join(OUTPUT_PATH , file_name), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)