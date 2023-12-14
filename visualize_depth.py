import argparse
import numpy as np
import OpenEXR
import Imath
import imageio
import glob
import os
import pdb
from PIL import Image

def exr_to_jpg(exr_file, jpg_file):
    # 打开EXR文件
    exr = OpenEXR.InputFile(exr_file)

    # 获取数据窗口大小
    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # 读取RGB通道
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    redstr = exr.channel('R', pt)
    greenstr = exr.channel('G', pt)
    bluestr = exr.channel('B', pt)

    # 将字符串转换为numpy数组
    red = np.frombuffer(redstr, dtype=np.float32)
    green = np.frombuffer(greenstr, dtype=np.float32)
    blue = np.frombuffer(bluestr, dtype=np.float32)
    red.shape = (size[1], size[0])  # 注意numpy使用行列顺序
    green.shape = (size[1], size[0])
    blue.shape = (size[1], size[0])

    # 合并RGB通道并转换为8位颜色
    rgb = np.stack([red, green, blue], axis=2)
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))  # 归一化
    rgb = (255 * rgb).astype(np.uint8)

    # 使用Pillow创建并保存JPG文件
    img = Image.fromarray(rgb, 'RGB')
    img.save(jpg_file)
    
def list_exr_channels(exr_file):
    # 打开EXR文件
    exr = OpenEXR.InputFile(exr_file)

    # 获取并打印所有通道名称
    channels = exr.header()['channels']
    print("Channels in the EXR file:")
    for channel in channels:
        print(channel)

def exr_to_jpg_greyscale(exr_file, jpg_file):
    # 打开EXR文件
    exr = OpenEXR.InputFile(exr_file)

    # 获取数据窗口大小
    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # 读取Y通道
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    ystr = exr.channel('Y', pt)

    # 将字符串转换为numpy数组
    y = np.frombuffer(ystr, dtype=np.float32)
    y.shape = (size[1], size[0])  # 注意numpy使用行列顺序

    # 转换为8位颜色并保存为JPG
    y = (y - np.min(y)) / (np.max(y) - np.min(y))  # 归一化
    y = (255 * y).astype(np.uint8)
    img = Image.fromarray(y, mode='L')
    img.save(jpg_file)


def main():
    files = glob.glob('data/deyilou/project/outdoor/am/images_4/depth/*.exr')
    savepath = '.'
    for file in files:
        filename,file_ext = os.path.splitext(file)
        filename = os.path.basename(filename)
        filename = filename + '.png'
        curpath = os.path.join(savepath,filename)
        exr_to_jpg_greyscale(file, curpath)
        # list_exr_channels(file)
        break

if __name__ == "__main__":
    main()
