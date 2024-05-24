import json
import pdb
from typing import NamedTuple

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


def storePly(path, xyz, rgb, normals):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    # normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # positions = np.vstack([vertices['z'], vertices['x'], vertices['y']]).T

    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.zeros_like(positions)
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(colors)

    colors = colors[::50]
    positions = positions[::50]
    normals = normals[::50]
    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def get_camera_frustum(img_size, K=None, c2w=None, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    
    vfov = 20
    # hfov = 20*1280/720
    hfov = 20 * 720 /1280

    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    frustum_points = np.array([[0., 0., 0.],                           # frustum origin
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

def visualize_cameras(colored_camera_dicts, camera_size=0.1):
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [coord_frame]
    idx = 0
    for color, data in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []

        for i in range(len(data)):
            c2w = data[i]
            img_size = [1280, 720]
            frustums.append(get_camera_frustum(img_size, None, c2w, frustum_length=camera_size, color=color))
            cnt += 1

        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)
    return things_to_draw

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


# def fetchPly(path):
    # plydata = PlyData.read(path)
    # vertices = plydata['vertex']
    # positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # # colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # colors=None
    # # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # normals = None
    # return BasicPointCloud(points=positions, colors=colors, normals=normals)



if __name__ == "__main__":

    geometries = []

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0.])
    geometries.append(mesh_frame)


    
    # pos_path = "/Users/lnxu/Downloads/debug/cc/transforms_train.json"
    # with open(pos_path, 'rb') as f:
    #     poses_json = json.load(f)

    # Rt_e = []
    # for i in range(len(poses_json["frames"])):
    #     pose = np.array(poses_json["frames"][i]["transform_matrix"])
    #     Rt_e.append(pose)

    # poses = np.array(Rt_e)
    # xyz = poses[:,:3,-1]

    # center = poses[:,:2,-1].mean(0)

    # # recenter poses and points clouds
    # # poses[:,:2,-1] -= center

    # poses[:,2,-1] += 2

    # colored_camera_dicts = [([1, 0, 0], [poses[i] for i in range(len(poses))])]
    # geometries.extend(visualize_cameras(colored_camera_dicts, camera_size=0.1))



    # ply_path = "mesh2pt.ply"
    scene = "apartment"
    # ply_path = f"/Users/lnxu/Downloads/{scene}/mesh.ply"
    ply_path = f"cuhk/all_30.ply"
    # ply_path = f"sby/all_30.ply"
    # ply_path = "point_cloud.ply"
    ply = fetchPly(ply_path)
    
    # pcd.colors = o3d.utility.Vector3dVector(ply.colors)
    

    # pos_path = "transforms.json"
    # with open(pos_path, 'rb') as f:
    #     poses_json = json.load(f)['KRT']

    # poses_json = [f for f in poses_json if eval(f['cameraId'].split('/')[0]) in [19]]
    # Rt_e = []
    # for i in range(len(poses_json)):
    #     pose = np.linalg.inv(np.array(poses_json[i]["T"]).T)
    #     Rt_e.append(pose)

    # pos_path = f"/Users/lnxu/Downloads/{scene}/transforms.json"
    # pos_path = f"block_small/transforms_train_depth.json"
    pos_path = f"cuhk/transforms_cuhk.json"
    # pos_path = f"sby/transforms_5.json"
    with open(pos_path, 'rb') as f:
        poses_json = json.load(f)['frames']

    # poses_json = [f for f in poses_json if eval(f['file_path'].split('/')[0]) in [19]]
    Rt_e = []

    for i in range(len(poses_json)):
        pose = np.array(poses_json[i]["transform_matrix"])
        if "35 degree" not in poses_json[i]["file_path"]:
            continue
        # pose[:, 1:3] *= -1
        Rt_e.append(pose)
        # if i > 50:
        #     break

    poses = np.array(Rt_e)
    xyz = poses[:,:3,-1]

    # center = poses[:,:2,-1].mean(0)
    center = ply.points[:,:-1].mean(0)

    # recenter poses and points clouds
    poses[:,:2,-1] -= center
    pcd = o3d.geometry.PointCloud()
    ply.points[:,:-1] -= center
    print(f"poses num:{poses.shape}")
    pcd.points = o3d.utility.Vector3dVector(ply.points[::10])
    colored_camera_dicts = [([0, 0, 0], [poses[i] for i in range(len(poses))])]
    geometries.extend(visualize_cameras(colored_camera_dicts, camera_size=10))
    
    
    geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries)
