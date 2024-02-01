#!/bin/bash

# 设置源文件夹路径和目标文件夹路径
source_dir="data/huanghelu_4parts/images"
destination_dir="data/huanghelu_4parts/images_cp"

# 创建目标文件夹
mkdir -p "$destination_dir"

# 递归查找后缀为jpg的文件，并拷贝到目标文件夹中
find "$source_dir" -type f -iname "*.jpg" -exec sh -c 'cp "$0" "$1/$(basename "$0")"' {} "$destination_dir" \;

echo "拷贝完成！"