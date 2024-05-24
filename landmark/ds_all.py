import argparse
import os
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description="downsample the image folder to source path")

    parser.add_argument("--folder_path", type=str, default='data/yinxinggushu_spring/images')
    parser.add_argument("--source_path", type=str, default='data/yinxinggushu_spring/images_5')
    args = parser.parse_args()
    return args

def downsample_images(folder_path, target_path, scale_factor):
    if not os.path.isdir(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            source_img_path = os.path.join(folder_path, filename)
            target_img_path = os.path.join(target_path, filename)
            if os.path.exists(target_img_path):
                continue
            with Image.open(source_img_path) as img:
                new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
                downsampled_img = img.resize(new_size, Image.ANTIALIAS)
                downsampled_img.save(target_img_path)
                print(f"Downsampled {filename} to {new_size}")

args = parse_args()
folder_path = args.folder_path
source_path = args.source_path
scale_factor = 5
os.makedirs(source_path,exist_ok=True)
downsample_images(folder_path, source_path, scale_factor)
