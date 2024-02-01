import os
from PIL import Image

def downsample_images(folder_path, target_path, scale_factor):
    if not os.path.isdir(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 构建完整的文件路径
            source_img_path = os.path.join(folder_path, filename)
            target_img_path = os.path.join(target_path, filename)
            # 打开并加载图片
            if os.path.exists(target_img_path):
                continue
            with Image.open(source_img_path) as img:
                # 计算下采样后的尺寸
                new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
                # 下采样图片
                downsampled_img = img.resize(new_size, Image.ANTIALIAS)
                # 保存下采样后的图片
                downsampled_img.save(target_img_path)
                print(f"Downsampled {filename} to {new_size}")

folder_path = 'data/huanghelu_4parts/images_cp/'
source_path = 'data/huanghelu_4parts/images_cp_5/'
scale_factor = 5 
# os.makedirs(source_path,exist_ok=True)
downsample_images(folder_path, source_path, scale_factor)
