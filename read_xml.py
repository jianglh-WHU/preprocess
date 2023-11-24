import argparse
from xml.dom import minidom
import numpy as np
import open3d as o3d
from tqdm import tqdm
import json
import os
import csv
import pdb


def get_camera_frustum(c2w=None, frustum_length=30, color=[0., 1., 0.]):
    
    vfov = 20
    hfov = 20 * 600/400

    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    half_w = frustum_length * np.tan(35.7/35/2.)
    half_h = frustum_length * np.tan(23.8/35/2.)

    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, -frustum_length],    # top-left image corner
                               [half_w, -half_h, -frustum_length],     # top-right image corner
                               [half_w, half_h, -frustum_length],      # bottom-right image corner
                               [-half_w, half_h, -frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    
    frustum_points_c2w = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), c2w.T)

    frustum_points_c2w = frustum_points_c2w[:, :3] / frustum_points_c2w[:, 3:4] #/ 100

    return frustum_points_c2w, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def parse_args():
    parser = argparse.ArgumentParser(
        description="convert xml to json for nerf")

    parser.add_argument("--output_json",
                        type=str,
                        default='shizi.json')
    parser.add_argument("--output_csv",
                        type=str,
                        default='1_48.csv')
    parser.add_argument("--input_xml",
                        type=str,
                        default='shizi.xml')
    # parser.add_argument("--input_transforms",
    #                     type=str,
    #                     default='china_museum.json')
    args = parser.parse_args()
    return args


def read_xml(input_xml):
    doc = minidom.parse(input_xml)

    photos = doc.getElementsByTagName("Block")[0]
    photos = photos.getElementsByTagName("Photogroups")[0]
    photos = photos.getElementsByTagName("Photogroup") # five groups

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0., 0., 0.])
    things_to_draw = [coord_frame]
    # things_to_draw = []

    color = [1, 0, 0]
    img_size = [600,400] 


    frustums = []
    Rt_e = []

    new_poses_bounds = []

    csv_data = [
        ["x", "y", "z","filename"]
    ]
    results = {}
    photogroup = doc.getElementsByTagName("Photogroup")

    for g_id in range(len(photogroup)):
        all_poses = photos[g_id]
        ImageDimensions = all_poses.getElementsByTagName('ImageDimensions')[0]
        width = float(ImageDimensions.getElementsByTagName('Width')[0].firstChild.data)
        height = float(ImageDimensions.getElementsByTagName('Height')[0].firstChild.data)
        
        try:
            focal_px = float(all_poses.getElementsByTagName('FocalLengthPixels')[0].firstChild.data)
            print(f'{focal_px}px')
        except:
            focal = float(all_poses.getElementsByTagName('FocalLength')[0].firstChild.data)
            sensor = float(all_poses.getElementsByTagName('SensorSize')[0].firstChild.data)
            focal_px = focal / sensor * width
            print(f'focal {focal}mm, {focal_px}px')


        photo_subgroup = all_poses.getElementsByTagName('Photo')

        for photo in tqdm(photo_subgroup):

            id = int(photo.getElementsByTagName('Id')[0].firstChild.data)
            path=photo.getElementsByTagName('ImagePath')[0].firstChild.data.split('/')[-3:]
            # path = path[0] + '/' + path[1] + '/' + path[2]
            path = os.path.join(*path)

            folder = path[:1]

            if len(photo.getElementsByTagName('Pose')) == 0:
                print('pose not found', id, path)
                continue
            pose = photo.getElementsByTagName('Pose')[0]

            if len(pose.getElementsByTagName('Rotation')) == 0 or len(pose.getElementsByTagName('Center')) == 0:
                print('pose not found', id, path)
                continue

            rot = pose.getElementsByTagName('Rotation')[0]
            cet = pose.getElementsByTagName('Center')[0]

            rot_mat = np.zeros((4,4))
            rot_mat[-1,-1] = 1

            for i in range(3):
                for j in range(3):
                    rot_mat[i][j] = float(rot.getElementsByTagName(f"M_{i}{j}")[0].firstChild.data)


            rot_mat[:3,:3] = np.linalg.inv(rot_mat[:3,:3])

            rot_mat[:3,1] = -rot_mat[:3,1]
            rot_mat[:3,2] = -rot_mat[:3,2]


            x = float(cet.getElementsByTagName("x")[0].firstChild.data)
            y = float(cet.getElementsByTagName("y")[0].firstChild.data)
            z = float(cet.getElementsByTagName("z")[0].firstChild.data)
            rot_mat[0,-1] = x
            rot_mat[1,-1] = y
            rot_mat[2,-1] = z
            # pdb.set_trace()
            csv_data.append([x,y,z,path.split("\\")[-1]])
            

            Rt_e.append(rot_mat)

            c2w = np.hstack([rot_mat[:3,:4], np.array([[height, width, focal_px]]).T])

            results[id] = {'path': path, 'rot_mat': c2w.tolist()}

    return results
    # o3d.visualization.draw_geometries(things_to_draw)

    # with open(args.output_csv, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(csv_data)
if __name__ == '__main__':
    args = parse_args()
    INPUT_XML=args.input_xml
    OUTPUT_JSON = args.output_json
    results = read_xml(input_xml=INPUT_XML)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=4)