import argparse
import numpy as np
import open3d
import pdb
from plyfile import PlyData, PlyElement
from typing import NamedTuple
import laspy

from read_xml import read_xml
from transforms3d.quaternions import mat2quat

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

class Model:
    def __init__(self):
        self.tj = None
        self.pcd =None
        self.__vis = None
    
    def read_model(self, xml_path, pcd_path):
        self.tj = read_xml(input_xml=xml_path)
        self.pcd = laspy.read(pcd_path)
        
    def add_pcd(self, remove_statistical_outlier=False):
        pcd = open3d.geometry.PointCloud()
        xyz = []
        rgb = []

        xyz = np.stack([self.pcd.x, self.pcd.y, self.pcd.z]).T.tolist()
        # normalize color to [0-255]
        max_color_value = max(self.pcd.red.max(), self.pcd.green.max(), self.pcd.blue.max())
        color_scale = 255 / max_color_value
        rgb = np.vstack([self.pcd.red, self.pcd.green, self.pcd.blue]).T * color_scale / 255
        
        rgb = rgb.tolist()
            
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)
        
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()
    
    def fetchPly(self, path):
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        try:
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        except:
            colors = np.zeros_like(positions)
        try:
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        except:
            normals = np.zeros_like(colors)
        return BasicPointCloud(points=positions, colors=colors, normals=normals)
    
    def add_points_gs(self, min_track_len=3, remove_statistical_outlier=True):
        self.gs_pcd = self.fetchPly('data/block_c_sparse/block_c_point_cloud.ply')
        gs_pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        # pdb.set_trace()
        # for point3D in self.gs_pcd:
            
        #     xyz.append(point3D.points)
        #     rgb.append(point3D.colors / 255)

        gs_pcd.points = open3d.utility.Vector3dVector(self.gs_pcd.points)
        gs_pcd.colors = open3d.utility.Vector3dVector(self.gs_pcd.colors)

        # # remove obvious outliers
        # if remove_statistical_outlier:
        #     [pcd, _] = pcd.remove_statistical_outlier(
        #         nb_neighbors=20, std_ratio=2.0
        #     )

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(gs_pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    def add_cameras(self, scale=1):
        frames = []
        keys = self.tj.keys()
        
        for key in keys:
            frame = self.tj[key]
            c2w = np.array(frame['rot_mat'])
            c2w = c2w[...,:-1]
            c2w[:, 1:3] *= -1
            # c2w = np.vstack((c2w, (0, 0, 0, 1)))
            # w2c = np.linalg.inv(c2w)
            # R = qvec2rotmat(mat2quat(w2c[:3,:3]))
            # t = w2c[:,3]

            # # invert
            # t = -R.T @ t[:-1]
            # R = R.T
            R = c2w[...,:-1]
            t = c2w[...,-1]
            
            frame = self.tj[key]
            file_path = frame['path']
            rot_mat = np.array(frame['rot_mat'])
            
            w = int(rot_mat[1,-1])
            h = int(rot_mat[0,-1])
            fx = rot_mat[2,-1]
            fy = rot_mat[2,-1]
            
            cx = w / 2 
            cy = h / 2 
            
            # intrinsics
            K = np.identity(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            
            cam_model = draw_camera(K, R, t, w, h, scale)
            frames.extend(cam_model)

        # add geometries to visualizer
        for i in frames:
            self.__vis.add_geometry(i)

    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()


def draw_camera(K, R, t, w, h, scale=1, color=[0.8, 0.2, 0.8]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5 * scale
    )
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize cc xml"
    )
    parser.add_argument(
        "--input_xml", help="path to cc xml", default='shizi.xml'
    )
    parser.add_argument(
        "--input_pcd", help="path to point cloud (format:las)", default='shizi.las'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # read xml
    model = Model()
    model.read_model(args.input_xml, args.input_pcd)
    print("num_cameras:", len(model.tj.keys()))
    print("num_pcd:", len(model.pcd))

    # display using Open3D visualization tools
    model.create_window()
    model.add_cameras(scale=0.25)
    model.add_pcd()
    model.show()


if __name__ == "__main__":
    main()
