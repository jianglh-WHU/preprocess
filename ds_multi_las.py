import argparse
import numpy as np
import laspy
import os
from plyfile import PlyData, PlyElement
from typing import NamedTuple
import pdb
from tqdm import tqdm

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    
def read_las_file(path, ratio):
    las = laspy.read(path)
    positions = np.vstack((las.x, las.y, las.z)).transpose()
    max_color_value = max(las.red.max(), las.green.max(), las.blue.max())
    color_scale = 255 / max_color_value
    colors = np.vstack([las.red, las.green, las.blue]).T * color_scale
    # try:
    #     colors = np.vstack((las.red, las.green, las.blue)).transpose()
    # except:
    #     colors = np.random.rand(positions.shape[0], positions.shape[1])
    # normals = np.random.rand(positions.shape[0], positions.shape[1])

    return positions, colors

def read_multiple_las_files(basepath, paths, ply_path, ratio):
    all_positions = []
    all_colors = []
    import pdb;pdb.set_trace()
    for i,path in tqdm(enumerate(paths), desc='read las...'):
        path = os.path.join(basepath, path)
        positions, colors = read_las_file(path, ratio)
        all_positions.append(positions)
        all_colors.append(colors)
        
        # if i>1:
        #     break

    all_positions = np.vstack(all_positions)[::ratio]
    all_colors = np.vstack(all_colors)[::ratio]

    print("Saving point cloud to .ply file...")
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(all_positions.shape[0], dtype=dtype)
    attributes = np.concatenate((all_positions, all_colors), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="merge multi las and ds")

    parser.add_argument("--ratio", type=int, default=5)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    las_path = os.path.join("data/cuhk/point_cloud/per_tile")
    las_list = os.listdir(las_path)
    # las_dir_list = os.listdir(las_path)
    # las_list = []
    # for i in las_dir_list:
    #     dir_path = os.path.join(las_path,i)
    #     if os.path.isdir(dir_path):

    #         las_list.append(os.path.join(i,os.listdir(dir_path)[0]))
    # import pdb;pdb.set_trace()  
    RATIO = args.ratio
    save_path = os.path.join(f"data/cuhk/point_cloud/all_{RATIO}.ply")
    read_multiple_las_files(las_path, las_list, save_path,RATIO)
    