import argparse
import os
import json
import imageio
from tqdm import tqdm
import numpy as np
from kornia import create_meshgrid
from typing import NamedTuple
import torch
from plyfile import PlyData, PlyElement
import os
import csv
import pdb

def get_ray_directions_blender(H, W, focal, center=None):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]#+0.5
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i-cent[0])/focal[0], -(j-cent[1])/focal[1], -torch.ones_like(i)],-1)  # (H, W, 3)
    directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions

def get_rays_with_directions(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d

def filtering_rays(aabb, all_rays, chunk=10240*5):
    N = torch.tensor(all_rays.shape[:-1]).prod()
    mask_filtered = []
    idx_chunks = torch.split(torch.arange(N), chunk)
    all_pts = []
    fars = []
    for idx_chunk in idx_chunks:
        rays_chunk = all_rays[idx_chunk]
        rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (aabb[1] - rays_o) / vec
        rate_b = (aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1) # least amount of steps needed to get inside bbox.
        t_max = torch.maximum(rate_a, rate_b).amin(-1) # least amount of steps needed to get outside bbox.
        mask_inbbox = t_max > t_min
        mask_filtered.append(mask_inbbox.cpu())
        d_z=rays_d[:, -1:]
        o_z=rays_o[:, -1:]
        far=-(o_z/d_z)
        pts = rays_o + rays_d*far
        all_pts.append(pts)
        fars.append(far)
    all_pts = torch.cat(all_pts)
    mask_filtered = torch.cat(mask_filtered)
    ratio = torch.sum(mask_filtered) / N
    return mask_filtered, ratio, all_pts, torch.max(torch.concat(fars))

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.zeros((positions.shape[0], 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def parse_args():
    parser = argparse.ArgumentParser(
        description="normalize and center json.")

    parser.add_argument("--type", type=str, nargs='+')

    parser.add_argument("--input_pcd",
                        type=str,
                        default='shimao_500.ply')
    
    parser.add_argument("--input_path",
                        type=str,
                        default='data/shimao/all')
    
    parser.add_argument("--input_json",
                        type=str,
                        default='transforms_5.json')

    parser.add_argument("--scene_scale",
                        type=int,
                        default='100')

    parser.add_argument("--pad",
                        type=int,
                        default=300)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    INPUT_PATH=args.input_path
    # OUTPUT_PATH=args.output_path
    OUTPUT_PATH = INPUT_PATH
    scale = args.scene_scale
    # aabb = [i/scale for i in args.aabb]
    # xmin,ymin,xmax,ymax = aabb
    pcd = fetchPly(os.path.join(args.input_path, args.input_pcd))
    xmin,ymin = np.min(pcd.points,0)[:-1] / scale
    xmax,ymax = np.max(pcd.points,0)[:-1] / scale
    print(xmin,ymin,xmax,ymax)

    with open(os.path.join(INPUT_PATH,f"{args.input_json}"), "r") as f:
        meta = json.load(f)

    transforms = {
            "camera_model": "SIMPLE_PINHOLE",
            "orientation_override": "none",
            "frames": []
        }
    all_frames=[]

    os.makedirs(os.path.join(INPUT_PATH,'filter'),exist_ok=True)
    filtered_dict = {
            "aabb":[xmin,ymin,xmax,ymax],
            "pad":args.pad,
            "frames": []
        }
    fnames=[]
    poses=[]
    all_ground_pts = []
    frames = meta['frames']
    
    fars_max=0
    for i,frame in tqdm(enumerate(frames)):
        pose = np.array(frames[i]["transform_matrix"])
        focal = frames[i]["fl_x"]
        H = frames[i]["h"]
        W = frames[i]["w"]
        cal_scale = 10*5
        directions = get_ray_directions_blender(int(H//cal_scale),int(W//cal_scale),(focal/cal_scale,focal/cal_scale))
        
        # import pdb;pdb.set_trace()
        pose = torch.FloatTensor(pose[:3,:4])
        pose [...,-1] /=scale
        rays_o, rays_d = get_rays_with_directions(directions, pose)
        mask_filtered, ratio, pts,far_max = filtering_rays([xmin,ymin,xmax,ymax], torch.cat([rays_o, rays_d], 1))
        fars_max = far_max if far_max > fars_max else fars_max
        # pts_in_ground = (pts[:,:2] > ground_bbox[0]).sum(-1) + (pts[:,:2] < ground_bbox[1]).sum(-1)
        # import pdb;pdb.set_trace()
        # pts_in_ground = (pts[:, :2] > aabb[0]).sum(-1) + (pts[:, :2] < aabb[1]).sum(-1)
        pts_min,pts_max = pts.min(0),pts.max(0)
        if pts_min[0][0]>xmin and pts_max[0][0]<xmax and pts_min[0][1]>ymin and pts_max[0][1]<ymax:
            # import pdb;pdb.set_trace()
            filtered_dict['frames'].append(frame)
            # pose = frames[i]["transform_matrix"]

            w = frames[i]["w"]
            h = frames[i]["h"]
            fl_x = frames[i]["fl_x"]
            fl_y = frames[i]["fl_y"]
            
            cx = w / 2 
            cy = h / 2 
            
            # import pdb;pdb.set_trace()
            # c2w = np.concatenate((pose[...,:-1],np.array([[0,0,0,1]])),axis=0)
            frame_dict = {
                'fl_x':float(fl_x),
                'fl_y':float(fl_y),
                'cx':cx,
                'cy':cy,
                'w':w,
                'h':h,
                'file_path':frames[i]["file_path"],
                'transform_matrix':frames[i]["transform_matrix"],
                }
            all_frames.append(frame_dict)
            
        # if i > 500:
        #     break
    # import pdb;pdb.set_trace()
    filtered_dict['far_max']=float(fars_max)
    print(f"far max:{fars_max}")
    for i,frame in enumerate(all_frames):
        transforms['frames'].append(frame)

    # import pdb;pdb.set_trace()
    # with open(os.path.join(INPUT_PATH,'filter',f'filter_{xmin}_{xmax}_{ymin}_{ymax}.json'), 'w') as fp: 
    #     json.dump(filtered_dict, fp, indent=4)

    with open(os.path.join(INPUT_PATH,'filter', f"transforms_{xmin}_{xmax}_{ymin}_{ymax}.json"),"w") as outfile:
        json.dump(transforms, outfile, indent=2)

    print('file saved to', os.path.join(INPUT_PATH,'filter',f'filter_{xmin}_{xmax}_{ymin}_{ymax}.json'))