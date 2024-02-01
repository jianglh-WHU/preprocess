import csv
import numpy as np
import os
import json
import pdb

import fnmatch

with open(os.path.join("data/huanghelu_4parts/transforms_20240129_5.json"), "r") as f:
    tj = json.load(f)
print(len(tj['frames']))

# pdb.set_trace()


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(root, filename)

# 指定文件夹路径和要查找的文件后缀
directory = "data/huanghelu_4parts/images"
pattern = "*.JPG"

# 使用生成器逐个获取匹配的文件路径
jpg_files = np.array(list(find_files(directory, pattern)))

file_names=[]

# 打印所有匹配的文件路径
for file in jpg_files:
    file_names.append(file.split('/')[-1])

file_names = np.array(file_names)
print(len(file_names))
print(len(np.unique(file_names)))

csv_filename = 'data/huanghelu_4parts/all/zong_20240129.csv'
with open(csv_filename, encoding="UTF-8") as file:
    reader = csv.DictReader(file)
    cameras = {}
    for row in reader:
        for column, value in row.items():
            cameras.setdefault(column, []).append(value)

camera_name = np.array(cameras["#name"])
unique_camera_name, counts = np.unique(camera_name, return_counts=True)
print(len(camera_name))
print(len(unique_camera_name))
duplicated_elements = unique_camera_name[counts > 1]
print(duplicated_elements)



