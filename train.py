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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, predicted_normal_loss, delta_normal_loss, zero_one_loss
from gaussian_renderer import render, network_gui, render_lighting
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import point_double_to_normal, depth_double_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
print(TENSORBOARD_FOUND)

from scene.cameras import Camera
import matplotlib.pyplot as plt
from utils.vis_utils import apply_depth_colormap

# function L1_loss_appearance is fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/train.py
def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    
    viewpoint_stack = None
    ema_loss_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0

    require_depth = not dataset.use_coord_map
    require_coord = dataset.use_coord_map
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, kernel_size, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        #gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        #brdf
        if pipe.brdf:
            gaussians.set_requires_grad("normal", state=iteration >= opt.normal_reg_from_iter)
            gaussians.set_requires_grad("normal2", state=iteration >= opt.normal_reg_from_iter)
            if gaussians.brdf_mode=="envmap":
                gaussians.brdf_mlp.build_mips()


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        reg_kick_on = iteration >= opt.regularization_from_iter
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size, require_coord = require_coord and reg_kick_on, require_depth = require_depth and reg_kick_on)
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
                                                                    render_pkg["render"], 
                                                                    render_pkg["viewspace_points"], 
                                                                    render_pkg["visibility_filter"], 
                                                                    render_pkg["radii"])


        #brdf extra losses
        #Check!!! I am only going to use Lsparse and Lreg, because RaDeGs already has the normalfromdepth and expected normal loss
        losses_extra = {}
        if pipe.brdf and iteration > opt.normal_reg_from_iter and not opt.disable_reg_loss:
            #if iteration<opt.normal_reg_util_iter:
            #    losses_extra['predicted_normal'] = predicted_normal_loss(render_pkg["normal"], render_pkg["normal_ref"], render_pkg["alpha"])
            losses_extra['zero_one'] = zero_one_loss(render_pkg["mask"])
            if "delta_normal_norm" not in render_pkg.keys() and opt.lambda_delta_reg>0: assert()
            if "delta_normal_norm" in render_pkg.keys():
                losses_extra['delta_reg'] = delta_normal_loss(render_pkg["delta_normal_norm"], render_pkg["mask"])    
        
        gt_image = viewpoint_cam.original_image.cuda()

        if dataset.use_decoupled_appearance:
            Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1_render = l1_loss(rendered_image, gt_image)

        
        if reg_kick_on:
            lambda_depth_normal = opt.lambda_predicted_normal
            if require_depth:
                rendered_expected_depth: torch.Tensor = render_pkg["expected_depth"]
                rendered_median_depth: torch.Tensor = render_pkg["median_depth"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = depth_double_to_normal(viewpoint_cam, rendered_expected_depth, rendered_median_depth)
            else:
                rendered_expected_coord: torch.Tensor = render_pkg["expected_coord"]
                rendered_median_coord: torch.Tensor = render_pkg["median_coord"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = point_double_to_normal(viewpoint_cam, rendered_expected_coord, rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
        else:
            lambda_depth_normal = 0
            depth_normal_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
            
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image.unsqueeze(0)))
        loss = rgb_loss  + depth_normal_loss * lambda_depth_normal
        if not opt.disable_reg_loss:
            for k in losses_extra.keys():
                loss += getattr(opt, f'lambda_{k}')* losses_extra[k]
      
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "loss_normal": f"{ema_normal_loss_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Log and save
            training_report(tb_writer, iteration, Ll1_render, loss, depth_normal_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, kernel_size))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = opt.size_threshold if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration)

            if pipe.brdf:
                gaussians.brdf_mlp.clamp_(min=0.0, max=1.0)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    print("all good")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, normal_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({
            'name': 'test', 
            'cameras' : scene.getTestCameras()
        }, {
            'name': 'train', 
            'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]
        })

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    # Optionally log a few images
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                
                # Average the test scores
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test:.4f}")
                
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint-l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint-psnr", psnr_test, iteration)
                
                # Save PSNR to file for the test configuration only (adjust if needed)
                if config["name"] == "test":
                    with open(os.path.join(scene.model_path, "psnr_scores.txt"), "a") as f:
                        f.write(f"Iteration {iteration}: PSNR {psnr_test.item():.4f}\n")
                        f.write(f"Iteration {iteration}: Loss {l1_test.item():.4f}, PSNR {psnr_test.item():.4f}\n")
        torch.cuda.empty_cache()


def training_report_old(tb_writer, iteration, Ll1, loss, normal_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        #for k in losses_extra.keys():
        #    tb_writer.add_scalar(f'train_loss_patches/{k}_loss', losses_extra[k].item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        #for config in validation_configs:
        #     if config['cameras'] and len(config['cameras']) > 0:
        #         l1_test = 0.0
        #         psnr_test = 0.0
        #         for idx, viewpoint in enumerate(config['cameras']):
        #             render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
        #             image = torch.clamp(render_result["render"], 0.0, 1.0)
        #             gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
        #             if tb_writer and (idx < 5):
        #                 tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
        #                 if iteration == testing_iterations[0]:
        #                     tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
        #                 for k in render_result.keys():
        #                     if render_result[k].dim()<3 or k=="render" or k=="delta_normal_norm":
        #                         continue
        #                     if k == "median_depth":
        #                         image_k = apply_depth_colormap(-render_result[k][0][...,None])
        #                         image_k = image_k.permute(2,0,1)
        #                     elif k == "mask":
        #                         image_k = apply_depth_colormap(render_result[k][0][...,None])
        #                         image_k = image_k.permute(2,0,1)
        #                     else:
        #                         if "normal" in k:
        #                             render_result[k] = 0.5 + (0.5*render_result[k]) # (-1, 1) -> (0, 1)
        #                         image_k = torch.clamp(render_result[k], 0.0, 1.0)
        #                     tb_writer.add_images(config['name'] + "_view_{}/{}".format(viewpoint.image_name, k), image_k[None], global_step=iteration)

        #                 if renderArgs[0].brdf:
        #                     lighting = render_lighting(scene.gaussians, resolution=(512, 1024))
        #                     if tb_writer:
        #                         tb_writer.add_images(config['name'] + "/lighting", lighting[None], global_step=iteration)
        #         l1_test += l1_loss(image, gt_image).mean().double()
        #         psnr_test += psnr(image, gt_image).mean().double()
        #         psnr_test /= len(config['cameras'])
        #         l1_test /= len(config['cameras'])
        #         print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        #         if config["name"] == "test":
        #             with open(scene.model_path + "/chkpnt" + str(iteration) + ".txt", "w") as file_object:
        #                 print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), file=file_object)
        #         if tb_writer:
        #             tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
        #             tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        # torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset=lp.extract(args), 
             opt=op.extract(args), 
             pipe=pp.extract(args), 
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             checkpoint_iterations=args.checkpoint_iterations, 
             checkpoint=args.start_checkpoint, 
             debug_from=args.debug_from)

    # All done
    print("\nTraining complete.")
