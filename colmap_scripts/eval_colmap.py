import argparse
import shutil
import subprocess
from pathlib import Path
import os
import numpy as np

from skimage.io import imread, imsave
from read_all_pose import read_all_pose
from transforms3d.quaternions import mat2quat

from colmap.database import COLMAPDatabase
from colmap.read_write_model import CAMERA_MODEL_NAMES
import open3d as o3d

# from ldm.base_utils import read_pickle
# from read_pose import *
from read_synthetic import read_pose

# K, _, _, _, POSES = read_pickle(f'temp/meta-fixed-16.pkl')
# H, W, NUM_IMAGES = 256, 256, 16
# K, POSES, H, W, NUM_IMAGES, img_names= read_pose(f'data/blockdemo')
# K, POSES, H, W, NUM_IMAGES, img_names= read_pose(f'data/small_city_img/new_high/block_3')
# K, POSES, H, W, NUM_IMAGES, img_names= read_all_pose(f'data/small_city_img/new_high/all/')
K, POSES, H, W, NUM_IMAGES, img_names= read_pose('data/nerf_synthetic/lego')

def extract_and_match_sift(colmap_path, database_path, image_dir):
    cmd = [
        str(colmap_path), 'feature_extractor',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    cmd = [
        str(colmap_path), 'exhaustive_matcher',
        '--database_path', str(database_path),
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

def run_triangulation(colmap_path, model_path, in_sparse_model, database_path, image_dir):
    print('Running the triangulation...')
    model_path.mkdir(exist_ok=True, parents=True)
    cmd = [
        str(colmap_path), 'point_triangulator',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--input_path', str(in_sparse_model),
        '--output_path', str(model_path),
        '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_principal_point', '0',
        '--Mapper.ba_refine_extra_params', '0']
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

def run_patch_match(colmap_path, sparse_model: Path, image_dir: Path, dense_model: Path):
    print('Running patch match...')
    assert sparse_model.exists()
    dense_model.mkdir(parents=True, exist_ok=True)
    cmd = [str(colmap_path), 'image_undistorter', '--input_path', str(sparse_model), '--image_path', str(image_dir), '--output_path', str(dense_model),]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    cmd = [str(colmap_path), 'patch_match_stereo','--workspace_path', str(dense_model),]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

def dump_images(in_image_dir, image_dir, names):
    new_names=[]
    for name in names:
    #     # img = imread(f'{in_image_dir}/{name}')
    #     # imsave(f'{str(image_dir)}/{name}', img)
    #     # if "train" in name:
    #     #     img = imread(f'{in_image_dir}/train/images/{name}')
    #     # elif "test" in  name:
    #     #     img = imread(f'{in_image_dir}/test/images/{name}')
    #     # imsave(f'{str(image_dir)}/{name}', img)
    #     # name_list = name.split('/')
    #     # path = os.path.join(*name_list)
    #     # file_path = f'{in_image_dir}/{path}'
    #     # import pdb;pdb.set_trace()
    #     # new_name = name_list[-3]+'-'+name_list[-2]+'-'+name_list[-1]
    #     # dest_path = f'{str(image_dir)}/{new_name}'
    #     # shutil.copy2(file_path, dest_path)
    #     # new_names.append(new_name)
    # return new_names
    # for index in range(NUM_IMAGES):
        # import pdb;pdb.set_trace()
        # img = imread(f'{in_image_dir}/{names[index]}')
        file_path = f'{in_image_dir}/{name}'
        name_ = name.split('/')[-1]
        dest_path = f'{str(image_dir)}/{name_}'
        shutil.copy2(file_path, dest_path)
        new_names.append(name_)
    return new_names
        # imsave(f'{str(image_dir)}/{names[index]}', img)

def build_db_known_poses_fixed(db_path, in_sparse_path, names):
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    # insert intrinsics
    with open(f'{str(in_sparse_path)}/cameras.txt', 'w') as f:
        for index in range(NUM_IMAGES):
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
            model, width, height, params = CAMERA_MODEL_NAMES['PINHOLE'].model_id, W, H, np.array((fx, fy, cx, cy),np.float32)
            # import pdb;pdb.set_trace()
            db.add_camera(model, width, height, params, prior_focal_length=(fx+fy)/2, camera_id=index+1)
            f.write(f'{index+1} PINHOLE {W} {H} {fx:.3f} {fy:.3f} {cx:.3f} {cy:.3f}\n')

    with open(f'{str(in_sparse_path)}/images.txt','w') as f:
        for index in range(NUM_IMAGES):
            pose = POSES[index]
            q = mat2quat(pose[:3,:3])
            t = pose[:,3]
            img_id = db.add_image(f"{names[index]}", camera_id=index+1, prior_q=q, prior_t=t)
            # img_id = db.add_image(f"{index:03}.png", camera_id=index+1, prior_q=q, prior_t=t)
            f.write(f'{img_id} {q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f} {t[0]:.5f} {t[1]:.5f} {t[2]:.5f} {index+1} {names[index]}\n\n')
            # f.write(f'{img_id} {q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f} {t[0]:.5f} {t[1]:.5f} {t[2]:.5f} {index+1} {index:03}.png\n\n')

    db.commit()
    db.close()

    with open(f'{in_sparse_path}/points3D.txt','w') as f:
        f.write('\n')

def patch_match_with_known_poses(in_image_dir, project_dir, colmap_path='colmap'):
    Path(project_dir).mkdir(exist_ok=True, parents=True)
    if os.path.exists(f'{str(project_dir)}/dense/stereo/depth_maps'): return

    # output poses
    db_path = f'{str(project_dir)}/database.db'
    image_dir = Path(f'{str(project_dir)}/images')
    sparse_dir = Path(f'{str(project_dir)}/sparse')
    in_sparse_dir = Path(f'{str(project_dir)}/sparse_in')
    dense_dir = Path(f'{str(project_dir)}/dense')

    image_dir.mkdir(exist_ok=True,parents=True)
    sparse_dir.mkdir(exist_ok=True,parents=True)
    in_sparse_dir.mkdir(exist_ok=True,parents=True)
    dense_dir.mkdir(exist_ok=True,parents=True)

    new_names=dump_images(in_image_dir, image_dir, img_names)
    build_db_known_poses_fixed(db_path, in_sparse_dir, new_names)
    # dump_images(in_image_dir, image_dir, img_names)
    # build_db_known_poses_fixed(db_path, in_sparse_dir, img_names)
    extract_and_match_sift(colmap_path, db_path, image_dir)
    run_triangulation(colmap_path,sparse_dir,  in_sparse_dir, db_path, image_dir)
    run_patch_match(colmap_path, sparse_dir, image_dir, dense_dir)

    # # fuse
    cmd = [str(colmap_path), 'stereo_fusion',
           '--workspace_path', f'{project_dir}/dense',
           '--workspace_format', 'COLMAP',
           '--input_type', 'geometric',
           '--output_path', f'{project_dir}/points.ply',]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str)
    parser.add_argument('--project',type=str)
    parser.add_argument('--name',type=str)
    args = parser.parse_args()

    if not os.path.exists(f'{args.project}/points.ply'):
        patch_match_with_known_poses(args.dir, args.project)

    mesh = o3d.io.read_triangle_mesh(f'{args.project}/points.ply',)
    vn = len(mesh.vertices)
    with open('colmap-results.log', 'a') as f:
        f.write(f'{args.name}\t{vn}\n')

if __name__=="__main__":
    main()