#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from matplotlib import cm

def apply_depth_colormap(depth, cmap="turbo", min=None, max=None):
    """
    Apply a colormap to a normalized depth tensor.
    """
    # Normalize depth to range [0, 1]
    near_plane = float(torch.min(depth)) if min is None else min
    far_plane = float(torch.max(depth)) if max is None else max
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    # Convert depth to NumPy for color mapping
    depth_np = depth.squeeze().cpu().numpy()  # Ensure it's 2D (H x W)

    # Apply colormap
    colormap = cm.get_cmap(cmap)  # Get the colormap
    depth_colored = colormap(depth_np)  # Returns (H x W x 4) RGBA array

    # Drop the alpha channel
    depth_colored = depth_colored[:, :, :3]  # Keep only RGB (H x W x 3)

    # Convert back to PyTorch tensor
    depth_colored = torch.from_numpy(depth_colored).float()  # Convert to tensor
    depth_colored = depth_colored.permute(2, 0, 1)  # Rearrange to CHW format

    return depth_colored
    
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        color = rendering["render"]
        depth = apply_depth_colormap(rendering["median_depth"], cmap="turbo")
        normal = 0.5 + 0.5 * rendering["normal"]
        alpha = rendering["mask"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(color, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normal, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(alpha, os.path.join(alpha_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.kernel_size)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.kernel_size)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)