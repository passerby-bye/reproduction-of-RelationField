# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""
Convert the rio dataset into nerfstudio format.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
import json
import os
import pymeshlab
import yaml

import rio

from nerfstudio.process_data import process_data_utils
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS

from nerfstudio.cameras import camera_utils

def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def read_intrinsic(intrinsic_path, mode='rgb'):
    with open(intrinsic_path, "r") as f:
        data = f.readlines()

    m_versionNumber = data[0].strip().split(' ')[-1]
    m_sensorName = data[1].strip().split(' ')[-2]

    if mode == 'rgb':
        m_Width = int(data[2].strip().split(' ')[-1])
        m_Height = int(data[3].strip().split(' ')[-1])
        m_Shift = None
        m_intrinsic = np.array([float(x) for x in data[7].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    else:
        m_Width = int(float(data[4].strip().split(' ')[-1]))
        m_Height = int(float(data[5].strip().split(' ')[-1]))
        m_Shift = int(float(data[6].strip().split(' ')[-1]))
        m_intrinsic = np.array([float(x) for x in data[9].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))

    m_frames_size = int(float(data[11].strip().split(' ')[-1]))

    return dict(
        m_versionNumber=m_versionNumber,
        m_sensorName=m_sensorName,
        m_Width=m_Width,
        m_Height=m_Height,
        m_Shift=m_Shift,
        m_intrinsic=np.matrix(m_intrinsic),
        m_frames_size=m_frames_size
    )

def process_rio(data: Path, output_dir: Path):
    """Process rio data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Record3D poses into the nerfstudio format.
    """

    # convert mesh to triangle mesh (open3d can only read triangle meshes)
    mesh_path = data / f'labels.ply'
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    ms.apply_filter('meshing_poly_to_tri')
    os.makedirs(output_dir, exist_ok=True)
    ms.save_current_mesh(str(output_dir / mesh_path.name), save_vertex_normal=True)

    verbose = True
    num_downscales = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size = 300
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)

    summary_log = []

    rio_image_dir = data 

    if not rio_image_dir.exists():
        raise ValueError(f"Image directory {rio_image_dir} doesn't exist")

    rio_image_filenames = []
    rio_depth_filenames = []
    rio_pose_filenames = []
    
    for f in rio_image_dir.iterdir():
        # if f.stem.startswith('frame'):  # removes possible duplicate images (for example, 123(3).jpg)
        if f.suffix.lower() in [".jpg"]:
            rio_image_filenames.append(f)
        # if f.stem.startswith('depth'):  # removes possible duplicate images (for example, 123(3).jpg)
        if f.suffix.lower() in [".png"] and 'depth' in f.stem:
            rio_depth_filenames.append(f)
        # if f.stem.startswith('pose'):
        if f.suffix.lower() in [".txt"] and 'pose' in f.stem:
            rio_pose_filenames.append(f)

    rio_image_filenames = sorted(rio_image_filenames)
    rio_depth_filenames = sorted(rio_depth_filenames)
    rio_pose_filenames = sorted(rio_pose_filenames)
    assert len(rio_image_filenames) == len(rio_depth_filenames) == len(rio_pose_filenames), f"{data}, Number of images {len(rio_image_filenames)}, depths {len(rio_depth_filenames)}, and poses {len(rio_pose_filenames)} must match"
    num_images = len(rio_image_filenames)

    idx = np.arange(num_images)
    if max_dataset_size != -1 and num_images > max_dataset_size:
        idx = np.round(np.linspace(0, num_images - 1, max_dataset_size)).astype(int)

    rio_image_filenames = list(np.array(rio_image_filenames)[idx])
    rio_depth_filenames = list(np.array(rio_depth_filenames)[idx])
    rio_pose_filenames = list(np.array(rio_pose_filenames)[idx])

    # Copy images to output directory
    copied_image_paths = process_data_utils.copy_images_list(
        rio_image_filenames,
        image_dir=image_dir,
        verbose=verbose,
        num_downscales=num_downscales,
    )
    copied_depth_paths = process_data_utils.copy_images_list(
        rio_depth_filenames,
        image_dir=depth_dir,
        verbose=verbose,
        num_downscales=num_downscales,
    )
    
    assert(len(copied_image_paths) == len(copied_depth_paths))
    num_frames = len(copied_image_paths)

    copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
    summary_log.append(f"Used {num_frames} images out of {num_images} total")
    if max_dataset_size > 0:
        summary_log.append(
            "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
            f"larger than the current value ({max_dataset_size}), or -1 to use all images."
        )

    rio_to_json(copied_image_paths, copied_depth_paths, rio_pose_filenames, output_dir, mesh_path, indices=idx)


def rio_to_json(images_paths: List[Path], depths_paths: List[Path], trajectory: List[Path], output_dir: Path, mesh_path: Path, indices: np.ndarray) -> int:
    """Converts rio's metadata and image paths to a JSON file.

    Args:
        images_paths: list if image paths.
        traj_path: Path to the rio trajectory file.
        output_dir: Path to the output directory.
        indices: Indices to sample the metadata_path. Should be the same length as images_paths.

    Returns:
        The number of registered images.
    """

    assert len(images_paths) == len(indices)
    poses_data = np.array([np.loadtxt(t_path, delimiter=' ') for t_path in trajectory]).astype(np.float32)

    poses_data[:,:3, 1] *= -1
    poses_data[:,:3, 2] *= -1

    poses_data = torch.from_numpy(poses_data)
    camera_to_worlds = poses_data

    frames = []
    for i, im_path in enumerate(images_paths):
        c2w = camera_to_worlds[i]
        frame = {
            "file_path": im_path.as_posix(),
            "depth_file_path": depths_paths[i].as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    intrinisics = yaml.safe_load(open(trajectory[0].parent / "camera.yaml",'r'))
    height = intrinisics['camera_intrinsics']['height']
    width = intrinisics['camera_intrinsics']['width']
    model = intrinisics['camera_intrinsics']['model']
    fx, fy, cx, cy = model
    distor = intrinisics['camera_intrinsics']['distortion']

    # Camera intrinsics


    out = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "k1": distor[0],
        "k2": distor[1],
        "k3": distor[2],
        "ply_file_path": mesh_path.as_posix(),
        "camera_model": CAMERA_MODELS["perspective"].name,
    }

    out["frames"] = frames
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    return len(frames)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the data directory.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()
    return args

def is_sequence_dir(path: Path) -> bool:
    """Return True when `path` already points at a single sequence folder."""
    return (
        path.is_dir()
        and (path / "labels.ply").exists()
        and (path / "camera.yaml").exists()
    )

if __name__ == "__main__":      
    args = get_args()
    data_root = Path(args.data)
    output_root = Path(args.output_dir)

    if is_sequence_dir(data_root):
        process_rio(data_root, output_root / data_root.name)
    else:
        for scene in rio.scenes:
            data = data_root / scene
            output_dir = output_root / f"rio_{scene}"
            process_rio(data, output_dir)
