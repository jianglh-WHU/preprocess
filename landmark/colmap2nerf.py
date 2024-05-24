import argparse
import os
import json
import imageio
import numpy as np
from pathlib import Path
import torch
from colmap_parsing_utils import read_cameras_binary, parse_colmap_camera_params, read_images_binary, qvec2rotmat

def parse_args():
    parser = argparse.ArgumentParser(
        description="convert cc json to transforms.json for nerfstudio")

    # parser.add_argument("--output_path",
    #                     type=str,
    #                     default='china_museum')
    parser.add_argument("--recon_dir",
                        type=str,
                        default='demo/apple_2/sparse/0')
    parser.add_argument("--output_dir",
                        type=str,
                        default='demo/apple_2')

    
    
    args = parser.parse_args()
    return args

def colmap_to_json(
    recon_dir,
    output_dir,
):
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db

    Returns:
        The number of registered images.
    """

    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        # TODO(1480) BEGIN use pycolmap API
        # rotation = im_data.rotation_matrix()
        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = im_data.name

        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }

        frames.append(frame)

    if set(cam_id_to_camera.keys()) != {1}:
        raise RuntimeError("Only single camera shared for all images is supported.")
    out = parse_colmap_camera_params(cam_id_to_camera[1])
    out["frames"] = frames

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([1, 0, 2]), :]
    applied_transform[2, :] *= -1
    out["applied_transform"] = applied_transform.tolist()

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)

if __name__ == "__main__":
    args = parse_args()
    recon_dir = Path(args.recon_dir)
    output_dir = Path(args.output_dir)
    colmap_to_json(recon_dir,output_dir)
    